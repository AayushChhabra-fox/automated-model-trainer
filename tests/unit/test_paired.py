"""Paired comparison: deltas, seeded bootstrap CIs, fraction improved."""

from __future__ import annotations

import numpy as np
import pytest

from amt.jobs.eval.paired import (
    DEFAULT_BOOTSTRAP_SEED,
    paired_bootstrap,
    paired_comparison,
    per_user_deltas,
)


def test_per_user_deltas_are_elementwise():
    got = per_user_deltas([0.5, 0.2, 1.0], [0.1, 0.4, 1.0])
    assert got.tolist() == pytest.approx([0.4, -0.2, 0.0])


def test_users_either_side_cannot_score_are_dropped_pairwise():
    cand = [0.5, np.nan, 0.3]
    base = [0.1, 0.2, np.nan]
    assert per_user_deltas(cand, base).tolist() == pytest.approx([0.4])


def test_misaligned_arrays_are_rejected():
    with pytest.raises(ValueError):
        per_user_deltas([1.0, 2.0], [1.0])


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    rng = np.random.default_rng(0)
    cand = rng.random(200)
    base = rng.random(200)
    a = paired_bootstrap(cand, base, n_boot=200, seed=DEFAULT_BOOTSTRAP_SEED)
    b = paired_bootstrap(cand, base, n_boot=200, seed=DEFAULT_BOOTSTRAP_SEED)
    assert a == b


def test_ci_brackets_the_mean_and_excludes_zero_for_a_clear_win():
    cand = np.full(400, 0.4)
    base = np.full(400, 0.1)
    stats = paired_bootstrap(cand, base, n_boot=300)
    assert stats["mean"] == pytest.approx(0.3)
    assert stats["ci_low"] <= stats["mean"] <= stats["ci_high"]
    assert stats["ci_low"] > 0.0
    assert stats["frac_improved"] == pytest.approx(1.0)
    assert stats["n_users"] == 400


def test_ci_includes_zero_for_pure_noise():
    rng = np.random.default_rng(11)
    values = rng.normal(size=500)
    stats = paired_bootstrap(values, values.copy(), n_boot=300)
    assert stats["mean"] == pytest.approx(0.0)
    assert stats["ci_low"] <= 0.0 <= stats["ci_high"]
    # Ties are not improvements.
    assert stats["frac_improved"] == 0.0


def test_frac_improved_counts_strictly_better_users():
    cand = [1.0, 0.0, 0.5, 0.5]
    base = [0.0, 1.0, 0.5, 0.4]
    stats = paired_bootstrap(cand, base, n_boot=50)
    assert stats["frac_improved"] == pytest.approx(0.5)


def test_empty_comparison_is_safe():
    stats = paired_bootstrap([], [], n_boot=10)
    assert stats["n_users"] == 0
    assert stats["mean"] == 0.0
    assert stats["ci_low"] == stats["ci_high"] == 0.0


def test_paired_comparison_covers_the_shared_metrics_only():
    cand = {"ndcg@10": [0.4, 0.5], "recall@10": [0.2, 0.3], "extra": [1.0, 1.0]}
    base = {"ndcg@10": [0.1, 0.2], "recall@10": [0.2, 0.1]}
    out = paired_comparison(cand, base, n_boot=100)
    assert sorted(out) == ["ndcg@10", "recall@10"]
    assert set(out["ndcg@10"]) == {
        "mean",
        "ci_low",
        "ci_high",
        "frac_improved",
        "n_users",
    }
    assert out["ndcg@10"]["mean"] == pytest.approx(0.3)


def test_sign_of_a_regression_is_negative():
    stats = paired_bootstrap([0.1] * 100, [0.5] * 100, n_boot=100)
    assert stats["mean"] < 0
    assert stats["ci_high"] < 0
    assert stats["frac_improved"] == 0.0
