"""Grouped re-ranking: one render is one evaluation unit.

The product question for a container is "should this item have been first",
and the reference point is the order the incumbent actually chose. These tests
pin the three pieces that makes possible: the render-grain split, the
logged-order baseline, and the position-bias weighting without which replay is
rigged for the incumbent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from amt.jobs.data.features import build_eval_frame
from amt.jobs.eval.harness import binary_metrics, evaluate, ndcg_at_k
from amt.jobs.materialize_splits import (
    SplitPlan,
    estimate_propensities,
    materialize,
    parse_position,
    relevance_weight,
)
from amt.jobs.models import logged_order

# Three renders per split window. `item_index` is the source's
# "rail_rank,column" form, and the conversion sits at a different slot in each.
IMPRESSIONS = pd.DataFrame(
    [
        # train window
        ("s1", "p1", "c1", "2026-08-01", "ROKU", "a", "0,0", True),
        ("s1", "p1", "c1", "2026-08-01", "ROKU", "b", "0,1", False),
        ("s1", "p1", "c1", "2026-08-01", "ROKU", "c", "0,2", False),
        ("s2", "p1", "c1", "2026-08-02", "ROKU", "a", "0,0", False),
        ("s2", "p1", "c1", "2026-08-02", "ROKU", "d", "0,1", True),
        ("s2", "p1", "c1", "2026-08-02", "ROKU", "e", "0,2", False),
        ("s3", "p1", "c1", "2026-08-03", "ROKU", "a", "0,0", True),
        ("s3", "p1", "c1", "2026-08-03", "ROKU", "f", "0,1", False),
        # validation window
        ("s4", "p1", "c1", "2026-08-09", "ROKU", "b", "0,0", False),
        ("s4", "p1", "c1", "2026-08-09", "ROKU", "a", "0,1", True),
        ("s5", "p1", "c1", "2026-08-10", "ROKU", "c", "0,0", True),
        ("s5", "p1", "c1", "2026-08-10", "ROKU", "d", "0,1", False),
        # qa window
        ("s6", "p1", "c1", "2026-08-16", "ROKU", "e", "0,0", True),
        ("s6", "p1", "c1", "2026-08-16", "ROKU", "f", "0,1", False),
        # dropped: a single-candidate render cannot be re-ranked
        ("s7", "p1", "c1", "2026-08-16", "ROKU", "a", "0,0", False),
    ],
    columns=[
        "session_id",
        "page_id",
        "parent_id",
        "ts",
        "client_name",
        "item_id",
        "item_index",
        "conversion_flag",
    ],
)

#: A placeholder split reference: materialization fills in the real one.
_SPLIT = {
    "name": "train",
    "table": "",
    "delta_version": 0,
    "row_count": 0,
    "membership_hash": "",
}

CONTRACT = {
    "contract_hash": "h",
    "evaluation_mode": "grouped_rerank",
    "cohort_columns": ["client_name"],
    "dataset": {
        "source_table": "cat.sch.impressions",
        "columns": {"user": "session_id", "item": "item_id", "timestamp": "ts"},
        "group_key": ["session_id", "page_id", "parent_id"],
        "position_column": "item_index",
        # The label lives on `dataset` beside the other executed fields. It was
        # briefly read from a `label_spec` block the contract does not have,
        # which made every label FALSE and every propensity 1.0 -- shaped
        # perfectly and completely useless.
        "label_column": "conversion_flag",
        "label_positive_values": [True],
        "snapshot_id": "s",
        "dataset_version": "0",
    },
    "split_policy": {
        "strategy": "temporal",
        "train_end": "2026-08-07",
        "validation_end": "2026-08-14",
        "qa_end": "2026-08-21",
    },
}


def _materialize(tmp_path, **contract_overrides):
    source = tmp_path / "impressions.parquet"
    IMPRESSIONS.to_parquet(source)
    contract = {**CONTRACT, **contract_overrides}
    contract["dataset"] = {**CONTRACT["dataset"], "source_path": str(source)}
    return materialize(contract, output_dir=str(tmp_path), catalog="cat", schema="sch")


def _split(tmp_path, name: str) -> pd.DataFrame:
    return pd.read_parquet(f"{tmp_path}/{name}.parquet")


# --- the slot ------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0,2", 2.0), ("3", 3.0), ("12,0", 0.0), ("", float("inf")), (None, float("inf"))],
)
def test_the_slot_is_the_last_element_of_item_index(raw, expected):
    """The source records "rail_rank,column"; the tile's column is the slot."""
    assert parse_position(raw) == expected


def test_an_unparseable_slot_sorts_last_rather_than_first():
    assert parse_position("nonsense") == float("inf")


# --- render-grain materialization ----------------------------------------


def test_each_render_becomes_one_row_in_the_incumbents_order(tmp_path):
    result = _materialize(tmp_path)

    train = _split(tmp_path, "split_train")
    assert result["evaluation_mode"] == "grouped_rerank"
    assert len(train) == 3, "three renders in the train window"
    first = train[train["group_id"] == "s1|p1|c1"].iloc[0]
    assert list(first["candidates"]) == ["a", "b", "c"]
    assert list(first["positions"]) == [0.0, 1.0, 2.0]
    assert list(first["labels"]) == [1.0, 0.0, 0.0]


def test_row_counts_are_group_counts_not_impression_counts(tmp_path):
    result = _materialize(tmp_path)

    counts = {name: ref["row_count"] for name, ref in result["splits"].items()}
    assert counts == {"train": 3, "validation": 2, "qa": 1}


def test_a_render_with_one_candidate_is_dropped(tmp_path):
    """There is nothing to re-rank, so it can only flatter whoever is scored."""
    _materialize(tmp_path)

    qa = _split(tmp_path, "split_qa")
    assert list(qa["group_id"]) == ["s6|p1|c1"], "s7 showed a single tile"


def test_the_cohort_column_rides_along(tmp_path):
    _materialize(tmp_path)
    assert set(_split(tmp_path, "split_train")["client_name"]) == {"ROKU"}


def test_membership_is_hashed_over_renders(tmp_path):
    result = _materialize(tmp_path)
    hashes = {ref["membership_hash"] for ref in result["splits"].values()}
    assert len(hashes) == 3 and all(hashes)


# --- the logged-order baseline -------------------------------------------


def test_the_incumbent_ranks_by_the_slot_it_showed(tmp_path):
    _materialize(tmp_path)
    frame = build_eval_frame(_split(tmp_path, "split_train"))

    ranked = logged_order.fit().recommend(
        frame.histories, k=3, candidates=frame.candidates, positions=frame.positions
    )

    assert ranked[0] == ["a", "b", "c"]


def test_the_incumbent_recovers_its_order_even_from_shuffled_candidates():
    model = logged_order.fit()

    ranked = model.recommend([[]], k=3, candidates=[["c", "a", "b"]], positions=[[2, 0, 1]])

    assert ranked[0] == ["a", "b", "c"]


def test_the_incumbent_has_no_catalog_to_cover():
    """Candidates are per render, so `coverage@k` comes from the split."""
    assert logged_order.fit().catalog_size == 0


def test_the_incumbent_round_trips_through_the_registry(tmp_path):
    from amt.jobs.models import registry

    model = registry.fit_model("logged_order", None, None, {})
    registry.save_model(model, str(tmp_path / "m"))
    reloaded = registry.load_model(str(tmp_path / "m"))

    assert reloaded.family == "logged_order"
    assert reloaded.recommend([[]], k=2, candidates=[["x", "y", "z"]])[0] == ["x", "y"]


# --- position-bias weighting ---------------------------------------------


def test_propensities_are_slot_rates_relative_to_slot_zero():
    """FYL's real shape: slot 0 converts ~40%, slot 1 ~5%, slot 2 ~2%."""
    rows = (
        [{"slot": 0.0, "label": True, "client_name": "ROKU"}] * 40
        + [{"slot": 0.0, "label": False, "client_name": "ROKU"}] * 60
        + [{"slot": 1.0, "label": True, "client_name": "ROKU"}] * 5
        + [{"slot": 1.0, "label": False, "client_name": "ROKU"}] * 95
    )

    theta = estimate_propensities(rows, floor=0.02, by="client_name")

    assert theta["ROKU|0"] == pytest.approx(1.0)
    assert theta["ROKU|1"] == pytest.approx(0.125)


def test_propensities_are_estimated_per_client_because_a_dpad_is_not_a_swipe():
    rows = (
        [{"slot": 0.0, "label": True, "client_name": "ROKU"}] * 50
        + [{"slot": 0.0, "label": False, "client_name": "ROKU"}] * 50
        + [{"slot": 1.0, "label": True, "client_name": "ROKU"}] * 5
        + [{"slot": 1.0, "label": False, "client_name": "ROKU"}] * 95
        + [{"slot": 0.0, "label": True, "client_name": "IOS"}] * 24
        + [{"slot": 0.0, "label": False, "client_name": "IOS"}] * 76
        + [{"slot": 1.0, "label": True, "client_name": "IOS"}] * 8
        + [{"slot": 1.0, "label": False, "client_name": "IOS"}] * 92
    )

    theta = estimate_propensities(rows, floor=0.02, by="client_name")

    assert theta["ROKU|1"] == pytest.approx(0.1)
    assert theta["IOS|1"] == pytest.approx(0.3333333, rel=1e-4)


def test_the_floor_caps_how_large_a_weight_can_get():
    rows = [{"slot": 0.0, "label": True}] * 100 + [{"slot": 9.0, "label": False}] * 100

    theta = estimate_propensities(rows, floor=0.02)

    assert theta["|9"] == 0.02
    assert relevance_weight(theta, "", 9.0, 0.02) == 50.0


def test_a_slot_with_no_reference_rate_gets_no_correction():
    """No slot-0 rate means no examination probability to divide by."""
    theta = estimate_propensities([{"slot": 3.0, "label": True}], floor=0.02)
    assert theta["|3"] == 1.0


def test_weights_are_written_next_to_the_labels(tmp_path):
    result = _materialize(tmp_path, relevance_weighting="pbm")

    train = _split(tmp_path, "split_train")
    assert "weights" in train.columns
    row = train[train["group_id"] == "s2|p1|c1"].iloc[0]
    # s2's conversion is at slot 1, which the train window says is rarer than
    # slot 0, so it carries more than unit weight.
    weights = list(row["weights"])
    assert weights[list(row["labels"]).index(1.0)] > 1.0
    assert result["propensities"]


def test_no_weighting_means_no_weights_column(tmp_path):
    _materialize(tmp_path)
    assert "weights" not in _split(tmp_path, "split_train").columns


def test_the_cutoffs_come_from_the_contracts_own_metrics():
    """`ndcg@4` must actually compute `ndcg@4`.

    The job used to fall back to 5/10/20 whenever `metrics` was a list -- which
    it always is -- so a campaign's primary metric was never computed and
    champion selection had nothing to select on. On a rail with 3.5 candidates
    per render, `@10` is also trivially 1.0.
    """
    from amt.jobs.train import contract_ks

    assert contract_ks(
        {
            "metrics": [{"name": "ndcg@4"}, {"name": "hitrate@1"}, {"name": "auc"}],
            "primary_metric": "ndcg@4",
        }
    ) == (1, 4)


def test_a_cutoff_only_the_primary_metric_names_is_still_computed():
    from amt.jobs.train import contract_ks

    assert contract_ks({"metrics": [{"name": "auc"}], "primary_metric": "mrr@7"}) == (7,)


def test_a_contract_naming_no_cutoffs_keeps_the_harness_defaults():
    from amt.jobs.eval.harness import DEFAULT_KS
    from amt.jobs.train import contract_ks

    assert contract_ks({"metrics": [{"name": "auc"}]}) == DEFAULT_KS


def test_a_dry_run_can_still_override_the_cutoffs():
    from amt.jobs.train import contract_ks

    assert contract_ks({"metrics": {"ks": [2, 3]}}) == (2, 3)
    assert contract_ks({"ks": [9]}) == (9,)


def test_the_generated_sql_matches_a_boolean_positive_value_case_insensitively():
    """Python writes `True`; Spark reads `true`. The mismatch cost two cluster runs.

    The pandas path compares Python to Python and matched fine, so every local
    test passed while the SQL matched nothing -- which turned the entire label
    column into FALSE on a cluster and collapsed every propensity to 1.0.
    """
    from amt.jobs.materialize_splits import _label_expr

    plan = SplitPlan(label_column="conversion_flag", positive_values=[True])

    expression = _label_expr(plan)

    assert expression == "(lower(cast(conversion_flag AS string)) IN ('true'))"
    assert "'True'" not in expression


def test_the_two_backends_agree_on_what_counts_as_a_conversion(tmp_path):
    """The pandas path is this module's reference semantics, so it has to."""
    _materialize(tmp_path)
    train = _split(tmp_path, "split_train")

    positives = sum(sum(1 for x in row if x > 0) for row in train["labels"])

    # Three train renders, one conversion each (s1 slot 0, s2 slot 1, s3 slot 0).
    assert positives == 3


def test_a_grouped_contract_without_a_label_column_is_refused(tmp_path):
    """The failure this prevents shipped once: FALSE labels, theta all 1.0.

    Every row's label resolved to FALSE, so the propensities collapsed to 1.0,
    the pbm weighting silently became no weighting, and three perfectly-shaped
    useless splits came back after eleven minutes of cluster time.
    """
    from amt.jobs.materialize_splits import plan_from_contract

    contract = {**CONTRACT, "dataset": {**CONTRACT["dataset"]}}
    contract["dataset"].pop("label_column")

    with pytest.raises(ValueError, match=r"dataset\.label_column"):
        plan_from_contract(contract)


def test_the_contract_schema_refuses_it_too():
    """Caught at HC3 as well as in the job, because HC3 is cheaper."""
    from pydantic import ValidationError

    from amt.schemas.evaluation_contract import EvaluationContract

    with pytest.raises(ValidationError, match="label_column"):
        EvaluationContract.model_validate(
            {
                "spec_hash": "s",
                "evaluation_mode": "grouped_rerank",
                "dataset": {
                    "source_table": "cat.sch.t",
                    "group_key": ["a"],
                    "position_column": "p",
                    "snapshot_id": "s",
                    "dataset_version": "1",
                },
                "train_split": _SPLIT,
                "validation_split": _SPLIT,
                "qa_split": _SPLIT,
                "split_policy": {
                    "strategy": "temporal",
                    "train_end": "2026-01-01",
                    "validation_end": "2026-01-08",
                },
                "primary_metric": "ndcg@4",
                "acceptance_criteria": {"primary_metric": "ndcg@4"},
            }
        )


# --- what the weighting does to a score ----------------------------------


def test_a_conversion_the_incumbent_buried_rewards_the_ranker_that_surfaces_it():
    """With theta = (1, 0.1) the logged order must not win by copying itself."""
    truth = [{"b"}]
    gains = [{"b": 1.0 / 0.1}]

    incumbent = ndcg_at_k([["a", "b"]], truth, 2, gains)
    reordered = ndcg_at_k([["b", "a"]], truth, 2, gains)

    assert reordered[0] > incumbent[0]


def test_without_weighting_the_two_orders_are_judged_the_same_way():
    """The unweighted metric cannot tell a good re-ordering from a lucky one."""
    truth = [{"b"}]

    assert ndcg_at_k([["b", "a"]], truth, 2)[0] == pytest.approx(1.0)
    assert ndcg_at_k([["a", "b"]], truth, 2)[0] == pytest.approx(0.6309298, rel=1e-6)


def test_every_metric_reduces_exactly_to_its_unweighted_form():
    """`gains` of 1.0 everywhere must be byte-identical to passing nothing."""
    ranked = [["a", "b", "c"], ["c", "a", "b"], ["x", "y", "z"]]
    truth = [{"b", "c"}, {"c"}, set()]
    unit = [dict.fromkeys(row, 1.0) for row in truth]

    plain = evaluate(ranked, truth, ks=(1, 2, 3), catalog_size=5)
    weighted = evaluate(ranked, truth, ks=(1, 2, 3), catalog_size=5, gains=unit)

    assert plain == weighted


def test_a_weighting_leaves_the_metric_names_alone():
    """`ndcg@4` is `ndcg@4`: the weighting is a contract fact, not a new metric."""
    scored = evaluate([["a"]], [{"a"}], ks=(4,), gains=[{"a": 50.0}])
    assert "ndcg@4" in scored and not any("pbm" in name for name in scored)


# --- the pointwise secondaries -------------------------------------------


def test_binary_metrics_separate_a_good_scorer_from_a_coin_flip():
    labels = [1, 0, 1, 0, 1, 0]
    good = binary_metrics([0.9, 0.1, 0.8, 0.2, 0.7, 0.3], labels)
    inverted = binary_metrics([0.1, 0.9, 0.2, 0.8, 0.3, 0.7], labels)

    assert good["auc"] == pytest.approx(1.0)
    assert inverted["auc"] == pytest.approx(0.0)
    assert good["logloss"] < inverted["logloss"]


def test_binary_metrics_score_ties_as_half():
    assert binary_metrics([0.5, 0.5], [1, 0])["auc"] == pytest.approx(0.5)


def test_binary_metrics_refuse_a_single_class():
    """AUC over one class is undefined, and a number there would be a lie."""
    assert binary_metrics([0.9, 0.1], [1, 1]) == {}
    assert binary_metrics([], []) == {}


def test_a_ranker_margin_is_squashed_before_logloss():
    """A raw margin is not a probability; logloss on one is meaningless."""
    scored = binary_metrics([4.0, -4.0], [1, 0])
    assert 0.0 < scored["logloss"] < 0.1


def test_calibration_error_is_zero_for_a_perfectly_calibrated_scorer():
    scores = [0.05] * 100 + [0.95] * 100
    labels = [0] * 95 + [1] * 5 + [0] * 5 + [1] * 95
    assert binary_metrics(scores, labels)["ece"] == pytest.approx(0.0, abs=0.01)


# --- fitting a booster on render-grain rows ------------------------------


GROUPED_RECIPE = {
    "sources": [{"name": "impressions", "role": "interactions", "split": "train"}],
    "feature_columns": [
        {"name": "pop", "kind": "count", "of": "item_id", "over": "train"},
        {"name": "rank", "kind": "column", "source": "item_train_rank"},
    ],
    "label": {
        "strategy": "label_column",
        "source_column": "conversion_flag",
        "positive_values": [True],
    },
    "group_key": "group_id",
    "candidate_generation": {"strategy": "logged_impressions", "n": 1},
    "filters": [],
}


def _rendered(n_groups: int = 6) -> pd.DataFrame:
    """Render-grain rows: three candidates each, the conversion rotating slots.

    Exactly the columns materialization writes -- and therefore *no* ``item_id``
    column. The first real campaign died on ``KeyError: 'item_id'`` because an
    earlier version of this fixture carried one and hid the assumption.
    """
    rows = []
    for g in range(n_groups):
        items = [f"i{(g + j) % 5}" for j in range(3)]
        labels = [0.0, 0.0, 0.0]
        labels[g % 3] = 1.0
        rows.append(
            {
                "group_id": f"g{g}",
                "candidates": items,
                "positions": [0.0, 1.0, 2.0],
                "labels": labels,
                "user_id": f"u{g}",
                "ts": g,
            }
        )
    return pd.DataFrame(rows)


def test_a_grouped_split_has_no_item_column_and_the_state_is_built_from_candidates():
    from amt.jobs.models import tabular_common

    frame = _rendered(6)
    assert "item_id" not in frame.columns
    state = tabular_common.build_state(GROUPED_RECIPE, frame, "item_id")
    assert set(state.catalog) == {f"i{k}" for k in range(5)}
    # i0..i4 each appear in 18 candidate slots spread over 5 titles
    assert sum(state.counts.values()) == 18.0, "a count is an impression"


def test_a_grouped_fit_uses_the_impressed_candidates_and_samples_no_negatives(stub_backend):
    """Every candidate is a real impression, so the non-converters are the negatives."""
    from amt.jobs.models import tabular_common

    model = tabular_common.fit_family(
        "lightgbm_reranker",
        _rendered(),
        _rendered(3),
        {"feature_recipe": GROUPED_RECIPE, "fit_spec": {"num_boost_round": 4}},
    )

    call = stub_backend.calls[-1]
    assert set(call["group_sizes"]) == {3}, "three impressed candidates per render"
    assert call["positives"] == len(call["group_sizes"]), "one conversion per render"
    assert len(call["group_sizes"]) == 6
    assert model.cfg["n_features"] == 2


def test_a_render_nobody_converted_in_carries_no_signal(stub_backend):
    from amt.jobs.models import tabular_common

    frame = _rendered(4)
    frame.at[0, "labels"] = [0.0, 0.0, 0.0]

    tabular_common.fit_family(
        "xgboost_reranker",
        frame,
        None,
        {"feature_recipe": GROUPED_RECIPE, "fit_spec": {"num_boost_round": 4}},
    )

    assert len(stub_backend.calls[-1]["group_sizes"]) == 3, "the flat render is dropped"


def test_a_grouped_model_reranks_only_what_the_container_showed(stub_backend):
    from amt.jobs.models import tabular_common

    model = tabular_common.fit_family(
        "lightgbm_reranker",
        _rendered(),
        None,
        {"feature_recipe": GROUPED_RECIPE, "fit_spec": {"num_boost_round": 4}},
    )

    ranked = model.recommend([["i0", "i1"]], k=5, candidates=[["i0", "i1"]])

    assert sorted(ranked[0]) == ["i0", "i1"], "no item outside the impression set"


def test_generating_logged_candidates_is_a_bug_not_a_fallback():
    """If the group row's candidates were lost, failing loudly is the only safe move."""
    from amt.jobs.models import tabular_common

    state = tabular_common.build_state(GROUPED_RECIPE, _rendered())

    with pytest.raises(tabular_common.UnsupportedRecipe, match="group row"):
        tabular_common._candidates_for(GROUPED_RECIPE, state, ["i0"])


def test_the_harness_reports_nan_free_totals_for_grouped_units():
    ranked = [["a", "b"], ["c", "d"]]
    truth = [{"a"}, set()]
    scored = evaluate(ranked, truth, ks=(1,), catalog_size=4)

    assert scored["n_users"] == 2.0
    assert scored["n_scored"] == 1.0
    assert not np.isnan(scored["ndcg@1"])


def test_an_attribute_source_is_read_under_the_contracts_item_column():
    """The split says ``item_id``; ``pr_item_feature`` says ``entity_id``. Bridge them."""
    from amt.jobs import train as train_mod

    class _Spark:
        def __init__(self) -> None:
            self.sql_text = ""

        def sql(self, text: str):
            self.sql_text = text
            return pd.DataFrame({"item_id": ["i0"], "content_type": ["news"]})

    spark = _Spark()
    contract = {
        "dataset": {"columns": {"user": "account_id", "item": "entity_id", "timestamp": "ts"}},
        "qa_split": {"table": "cat.sch.qa"},
    }
    config = {
        "feature_recipe": {
            "sources": [
                {
                    "name": "traits",
                    "table": "pnr_prod.silver.pr_item_feature",
                    "columns": ["entity_id", "content_type"],
                }
            ]
        }
    }
    frames = train_mod._recipe_attribute_frames(contract, config, "item_id", spark)

    assert "entity_id AS item_id" in spark.sql_text
    assert spark.sql_text.count("entity_id") == 1, "the key is not selected twice"
    assert list(frames["traits"].columns) == ["item_id", "content_type"]


# --- what a grouped recipe can read, and what reaches the booster -----------


def _rendered_with_context(n_groups: int = 6) -> pd.DataFrame:
    """A grouped split as materialization writes it: a cohort column and weights."""
    frame = _rendered(n_groups)
    frame["client_name"] = ["ROKU" if g % 2 == 0 else "FIRETV" for g in range(n_groups)]
    frame["weights"] = [
        [w if flag > 0 else 0.0 for flag, w in zip(labels, (1.0, 4.0, 9.0), strict=True)]
        for labels in frame["labels"]
    ]
    return frame


CONTEXT_RECIPE = {
    **GROUPED_RECIPE,
    "feature_columns": [
        {"name": "client_code", "kind": "cast", "source": "client_name", "to": "code"},
        {"name": "rank", "kind": "column", "source": "item_train_rank"},
    ],
}


def test_a_render_level_column_is_a_feature_read_from_the_row(stub_backend):
    from amt.jobs.models import tabular_common

    model = tabular_common.fit_family(
        "lightgbm_classifier",
        _rendered_with_context(),
        None,
        {"feature_recipe": CONTEXT_RECIPE, "fit_spec": {"num_boost_round": 4}},
    )

    assert model.state.context_columns == ["client_name"]
    assert model.state.attribute_columns == []
    assert set(model.state.category_codes["client_name"]) == {"FIRETV", "ROKU"}
    call = stub_backend.calls[-1]
    assert call["columns"] == ["client_code", "rank"]


def test_scoring_hands_the_render_context_to_the_model(stub_backend):
    from amt.jobs.models import tabular_common

    frame = _rendered_with_context(4)
    model = tabular_common.fit_family(
        "lightgbm_classifier",
        frame,
        None,
        {"feature_recipe": CONTEXT_RECIPE, "fit_spec": {"num_boost_round": 4}},
    )
    eval_frame = build_eval_frame(frame)

    # g0..g3 sort in render order, so the contexts line up with the rows.
    assert eval_frame.contexts == [
        {"client_name": "ROKU"},
        {"client_name": "FIRETV"},
        {"client_name": "ROKU"},
        {"client_name": "FIRETV"},
    ]
    scores, _groups, _items = model.score(
        eval_frame.histories, eval_frame.candidates, contexts=eval_frame.contexts
    )
    assert len(scores) == sum(len(c) for c in eval_frame.candidates)


def test_a_feature_whose_source_resolves_to_nothing_is_refused(stub_backend):
    """The first LFY recipe read `content_type` off a split that has no such column.

    Training on it would have produced constant features and a model that
    reproduces the tie rule. Refuse at fit time, naming the columns.
    """
    from amt.jobs.models import tabular_common

    recipe = {
        **GROUPED_RECIPE,
        "feature_columns": [
            {"name": "ct", "kind": "cast", "source": "content_type", "to": "code"},
        ],
    }
    with pytest.raises(tabular_common.UnsupportedRecipe, match="content_type"):
        tabular_common.fit_family(
            "lightgbm_classifier",
            _rendered(),
            None,
            {"feature_recipe": recipe, "fit_spec": {"num_boost_round": 4}},
        )


def test_the_contracts_weights_reach_a_classifier_as_sample_weights(stub_backend):
    from amt.jobs.models import tabular_common

    tabular_common.fit_family(
        "lightgbm_classifier",
        _rendered_with_context(),
        None,
        {"feature_recipe": CONTEXT_RECIPE, "fit_spec": {"num_boost_round": 4}},
    )

    call = stub_backend.calls[-1]
    assert call["weight"] is not None
    positives = [w for y, w in zip(call["y"], call["weight"], strict=True) if y > 0]
    negatives = [w for y, w in zip(call["y"], call["weight"], strict=True) if y == 0]
    assert set(positives) <= {1.0, 4.0, 9.0}, "a positive weighs 1/theta of its slot"
    assert set(negatives) == {1.0}, "a negative weighs one"
    assert set(call["y"]) == {0.0, 1.0}, "labels stay binary for a classifier"


def test_the_contracts_weights_reach_a_reranker_as_linear_grades(stub_backend):
    from amt.jobs.models import tabular_common

    tabular_common.fit_family(
        "lightgbm_reranker",
        _rendered_with_context(),
        None,
        {"feature_recipe": CONTEXT_RECIPE, "fit_spec": {"num_boost_round": 4}},
    )

    call = stub_backend.calls[-1]
    assert call["weight"] is None, "a ranker takes grades, not weights"
    assert set(call["y"]) <= {0.0, 1.0, 4.0, 9.0}
    assert call["params"]["label_gain"] == list(range(10)), "linear gain, not 2^grade - 1"


def test_lightgbm_gets_ndcg_and_eval_at_when_the_fit_spec_says_ndcg_at_4(stub_backend):
    """`ndcg@4` is xgboost's spelling; LightGBM rejects it as a metric name."""
    from amt.jobs.models import tabular_common

    tabular_common.fit_family(
        "lightgbm_reranker",
        _rendered_with_context(),
        None,
        {"feature_recipe": CONTEXT_RECIPE, "fit_spec": {"num_boost_round": 4, "eval_metric": "ndcg@4"}},
    )

    params = stub_backend.calls[-1]["params"]
    assert params["metric"] == "ndcg" and params["eval_at"] == [4]


def test_a_fit_survives_the_array_columns_spark_actually_hands_over(stub_backend):
    """Spark -> pandas yields numpy arrays, whose truth value is ambiguous.

    The second real LFY training job died on `candidates or ()`. Every fixture
    above uses Python lists, which is why nothing caught it; this one uses what
    `to_pandas` produces.
    """
    from amt.jobs.models import tabular_common

    frame = _rendered_with_context(6)
    for column in ("candidates", "positions", "labels", "weights"):
        frame[column] = [np.asarray(v, dtype=object if column == "candidates" else float) for v in frame[column]]

    model = tabular_common.fit_family(
        "lightgbm_classifier",
        frame,
        frame.iloc[:3],
        {"feature_recipe": CONTEXT_RECIPE, "fit_spec": {"num_boost_round": 4}},
    )

    assert len(stub_backend.calls[-1]["group_sizes"]) == 6
    eval_frame = build_eval_frame(frame)
    ranked = model.recommend(
        eval_frame.histories, 3, candidates=eval_frame.candidates, contexts=eval_frame.contexts
    )
    assert all(sorted(r) == sorted(c) for r, c in zip(ranked, eval_frame.candidates, strict=True))


def test_training_controls_never_reach_the_booster_as_hyperparameters(stub_backend):
    """`training: {early_stopping: true}` resolved to LightGBM's early_stopping_round=True."""
    from amt.jobs.models import tabular_common

    tabular_common.fit_family(
        "lightgbm_classifier",
        _rendered_with_context(),
        _rendered_with_context(3),
        {
            "feature_recipe": CONTEXT_RECIPE,
            "fit_spec": {"num_boost_round": 4, "early_stopping_rounds": 2, "seed": 7},
            "training": {"early_stopping": True, "num_iterations": 999, "learning_rate": 0.2},
        },
    )

    call = stub_backend.calls[-1]
    params = call["params"]
    assert "early_stopping" not in params and "early_stopping_round" not in params
    assert "num_iterations" not in params
    assert params["learning_rate"] == 0.2, "a real hyperparameter still gets through"
    assert params["seed"] == 7 and call["rounds"] == 4 and call["early_stopping_rounds"] == 2
