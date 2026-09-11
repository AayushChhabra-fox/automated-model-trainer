"""Harness metric math, checked against values computed by hand.

These are the numbers every scientific claim in a campaign rests on, so they
are pinned to closed-form expectations rather than to the implementation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from amt.jobs.eval import cohorts as cohort_mod
from amt.jobs.eval.harness import (
    HARNESS_VERSION,
    evaluate,
    hitrate_at_k,
    item_coverage,
    metric_names,
    mrr,
    ndcg_at_k,
    per_user_metrics,
    recall_at_k,
)


def test_harness_version_is_pinned():
    # The contract records this string; bumping it must be deliberate.
    assert HARNESS_VERSION == "1.0.0"


def test_recall_counts_retrieved_fraction():
    ranked = [[1, 2, 3, 4], [9, 8, 7, 6]]
    truth = [{2, 4}, {1}]
    got = recall_at_k(ranked, truth, 4)
    assert got.tolist() == [1.0, 0.0]
    assert recall_at_k(ranked, truth, 2).tolist() == [0.5, 0.0]


def test_recall_does_not_exceed_one_with_duplicate_ranks():
    # A buggy model that emits the same item twice must not score twice.
    assert recall_at_k([[5, 5, 5]], [{5}], 3).tolist() == [1.0]


def test_hitrate_is_binary():
    ranked = [[1, 2], [3, 4], [5, 6]]
    truth = [{2}, {99}, {5, 6}]
    assert hitrate_at_k(ranked, truth, 2).tolist() == [1.0, 0.0, 1.0]


def test_ndcg_matches_closed_form():
    # One relevant item at rank 2 -> DCG = 1/log2(3), IDCG = 1/log2(2) = 1.
    got = ndcg_at_k([[7, 3, 9]], [{3}], 3)
    assert got[0] == pytest.approx(1.0 / math.log2(3))

    # Two relevant items at ranks 1 and 3.
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    got = ndcg_at_k([[1, 5, 2]], [{1, 2}], 3)
    assert got[0] == pytest.approx(dcg / idcg)


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([[1, 2, 3]], [{1, 2, 3}], 3)[0] == pytest.approx(1.0)


def test_mrr_uses_first_relevant_rank():
    ranked = [[4, 5, 6], [7, 8, 9]]
    truth = [{6}, {7, 9}]
    assert mrr(ranked, truth, 3).tolist() == [1.0 / 3.0, 1.0]


def test_mrr_is_zero_when_nothing_is_retrieved():
    assert mrr([[1, 2]], [{99}], 2).tolist() == [0.0]


def test_item_coverage_is_distinct_items_over_catalog():
    ranked = [[1, 2], [2, 3], [1, 2]]
    assert item_coverage(ranked, catalog_size=10, k=2) == pytest.approx(0.3)
    assert item_coverage(ranked, catalog_size=0, k=2) == 0.0


def test_users_without_truth_are_excluded_not_counted_as_zero():
    ranked = [[1, 2], [3, 4]]
    truth = [{1}, set()]
    per_user = recall_at_k(ranked, truth, 2)
    assert per_user[0] == 1.0
    assert np.isnan(per_user[1])

    result = evaluate(ranked, truth, ks=(2,), catalog_size=10)
    # Mean over the single scoreable user, not 0.5 over both.
    assert result["recall@2"] == pytest.approx(1.0)
    assert result["n_users"] == 2.0
    assert result["n_scored"] == 1.0


def test_evaluate_emits_every_metric_family_per_k():
    ranked = [[1, 2, 3], [3, 2, 1]]
    truth = [{1}, {2}]
    result = evaluate(ranked, truth, ks=(1, 3), catalog_size=5)
    for name in metric_names((1, 3)):
        assert name in result
    assert "coverage@1" in result and "coverage@3" in result


def test_evaluate_can_return_aligned_per_user_arrays():
    ranked = [[1], [2], [3]]
    truth = [{1}, {9}, {3}]
    result = evaluate(ranked, truth, ks=(1,), catalog_size=3, include_per_user=True)
    per_user = result["per_user"]
    assert per_user["recall@1"].tolist() == [1.0, 0.0, 1.0]
    assert len(per_user["ndcg@1"]) == len(ranked)


def test_mismatched_input_lengths_are_rejected():
    with pytest.raises(ValueError):
        recall_at_k([[1]], [{1}, {2}], 1)


def test_per_user_metrics_keys_match_metric_names():
    per_user = per_user_metrics([[1]], [{1}], ks=(1,))
    assert sorted(per_user) == sorted(metric_names((1,)))


# -- cohorts ---------------------------------------------------------------


def test_cohort_buckets_split_cold_start_out():
    assert cohort_mod.history_length_cohort(0) == "cold_start"
    assert cohort_mod.history_length_cohort(3) == "hist_1_4"
    assert cohort_mod.history_length_cohort(19) == "hist_5_19"
    assert cohort_mod.history_length_cohort(500) == "hist_20_plus"


def test_recency_buckets():
    assert cohort_mod.recency_cohort(1) == "recency_0_7d"
    assert cohort_mod.recency_cohort(20) == "recency_8_30d"
    assert cohort_mod.recency_cohort(400) == "recency_31d_plus"
    assert cohort_mod.recency_cohort(None) == "recency_unknown"


def test_cohort_aggregation_averages_within_bucket():
    per_user = {"recall@5": np.array([1.0, 0.0, 0.5, np.nan])}
    labels = ["a", "a", "b", "b"]
    agg = cohort_mod.aggregate_by_cohort(per_user, labels)
    assert agg["a"]["recall@5"] == pytest.approx(0.5)
    assert agg["a"]["n_users"] == 2.0
    # The nan user is ignored in the mean but still counted in n_users.
    assert agg["b"]["recall@5"] == pytest.approx(0.5)
    assert agg["b"]["n_users"] == 2.0


def test_cohort_metrics_namespaces_dimensions():
    histories = [[], [1, 2], [1] * 30]
    assignments = cohort_mod.assign_cohorts(histories, [1.0, 40.0, 10.0])
    per_user = {"ndcg@10": np.array([0.1, 0.2, 0.3])}
    out = cohort_mod.cohort_metrics(per_user, assignments)
    assert "history_length=cold_start" in out
    assert "recency=recency_31d_plus" in out


def test_cohort_labels_must_align_with_metrics():
    with pytest.raises(ValueError):
        cohort_mod.aggregate_by_cohort({"m": np.array([1.0, 2.0])}, ["a"])
