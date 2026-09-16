"""What the runtime can express, and what happens to a hypothesis that needs more.

The previous campaign's backlog asked for transforms that do not exist and was
approved anyway; the cost landed three stages later, on a cluster.
"""

from __future__ import annotations

import pytest

from amt.graph.nodes.planner import _duplicate_shapes, _mark_feasibility
from amt.graph.nodes.trainer import _repeated_configs
from amt.schemas.feature_recipe import (
    BASE_COLUMNS,
    runtime_vocabulary,
    unsupported_requirements,
)
from amt.schemas.hypothesis import Hypothesis
from amt.schemas.run_result import RunSummary


def _hypothesis(requires=(), family="lightgbm_reranker", hypothesis_id="hyp-1") -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        title="t",
        proposed_change="c",
        why_it_should_improve="w",
        motivating_evidence="e",
        experiment_design="d",
        metric_expected_to_improve="ndcg@10",
        falsification_criteria="f",
        family=family,
        requires=list(requires),
    )


# --- the vocabulary -------------------------------------------------------


def test_the_vocabulary_is_derived_from_the_types_not_restated():
    """A literal added to `TransformKind` must show up here without an edit."""
    from typing import get_args

    from amt.schemas.feature_recipe import TransformKind

    vocabulary = runtime_vocabulary()

    assert set(vocabulary["transform_kinds"]) == set(get_args(TransformKind))
    assert vocabulary["base_columns"] == list(BASE_COLUMNS)
    assert "lightgbm_reranker" in vocabulary["families"]
    assert "popularity" in vocabulary["families"]


def test_everything_in_the_vocabulary_is_supported():
    every = [value for values in runtime_vocabulary().values() for value in values]
    assert unsupported_requirements(every) == []


def test_a_requirement_outside_the_vocabulary_is_named():
    assert unsupported_requirements(["column", "embedding_lookup", "rolling_window"]) == [
        "embedding_lookup",
        "rolling_window",
    ]


def test_blanks_are_not_requirements():
    assert unsupported_requirements(["", "  ", "column"]) == []


# --- striking out the infeasible -----------------------------------------


def test_a_buildable_hypothesis_is_left_alone():
    hypothesis = _hypothesis(requires=["lightgbm_reranker", "count", "popularity_topn"])
    assert _mark_feasibility(hypothesis).status == "pending"


def test_a_hypothesis_needing_something_unbuildable_is_struck_out_with_a_reason():
    marked = _mark_feasibility(_hypothesis(requires=["count", "sequence_encoder"]))

    assert marked.status == "infeasible"
    assert "sequence_encoder" in marked.infeasible_reason


def test_an_unknown_family_is_infeasible_even_when_requires_is_empty():
    marked = _mark_feasibility(_hypothesis(family="two_tower_dnn"))

    assert marked.status == "infeasible"
    assert "two_tower_dnn" in marked.infeasible_reason


def test_a_hypothesis_with_no_family_named_is_not_struck_out():
    """A blank family is a fixed-shape default, not a claim about a missing one."""
    assert _mark_feasibility(_hypothesis(family="")).status == "pending"


def test_an_infeasible_hypothesis_is_not_pending_so_selection_skips_it():
    marked = _mark_feasibility(_hypothesis(requires=["embedding_lookup"]))
    backlog = [marked, _hypothesis(hypothesis_id="hyp-2")]

    assert [h.hypothesis_id for h in backlog if h.status == "pending"] == ["hyp-2"]


def test_infeasible_is_a_real_status(evaluation_contract):
    from amt.graph.routing import route_after_hc_hypotheses

    state = {
        "hypotheses_approved": True,
        "hypotheses": [_mark_feasibility(_hypothesis(requires=["embedding_lookup"]))],
    }
    # The only tabular hypothesis cannot be built, so there is no shape to review.
    assert route_after_hc_hypotheses(state) == "planner_select_hypothesis"


# --- duplicate shapes ----------------------------------------------------


class _Proposal:
    def __init__(self, shape_hash: str) -> None:
        self.feature_fit_hash = shape_hash


def test_two_proposals_with_one_shape_point_at_each_other():
    duplicates = _duplicate_shapes({"hyp-1": _Proposal("a"), "hyp-2": _Proposal("a")})

    assert duplicates == {"hyp-1": ["hyp-2"], "hyp-2": ["hyp-1"]}


def test_distinct_shapes_are_not_flagged():
    assert _duplicate_shapes({"hyp-1": _Proposal("a"), "hyp-2": _Proposal("b")}) == {}


def test_a_shape_the_campaign_already_measured_is_the_expensive_duplicate():
    history = [
        RunSummary(run_id="r1", mlflow_run_id="m1", hypothesis_id="hyp-9", feature_fit_hash="a")
    ]

    duplicates = _duplicate_shapes({"hyp-1": _Proposal("a")}, history)

    assert duplicates == {"hyp-1": ["already run as hyp-9"]}


def test_history_without_a_shape_hash_is_ignored():
    history = [RunSummary(run_id="r1", mlflow_run_id="m1", hypothesis_id="hyp-9")]
    assert _duplicate_shapes({"hyp-1": _Proposal("a")}, history) == {}


# --- no-op iterations ----------------------------------------------------


def _summary(iteration: int, config_hash: str) -> RunSummary:
    return RunSummary(
        run_id=f"r{iteration}",
        mlflow_run_id=f"m{iteration}",
        iteration=iteration,
        config_hash=config_hash,
    )


def test_an_identical_retune_is_reported_as_a_repeat():
    """Identical configs produce identical metrics, which read as a plateau."""
    repeats = _repeated_configs([_summary(1, "abc"), _summary(2, "abc")])

    assert len(repeats) == 1
    assert "iteration 2 repeats iteration 1" in repeats[0]


def test_genuinely_different_iterations_are_not_repeats():
    assert _repeated_configs([_summary(1, "abc"), _summary(2, "def")]) == []


def test_runs_without_a_config_hash_are_skipped():
    assert _repeated_configs([_summary(1, ""), _summary(2, "")]) == []


# --- the join block is gone ----------------------------------------------


def test_a_recipe_can_no_longer_declare_a_join():
    """Validated, never executed, and the planner kept writing join graphs."""
    from pydantic import ValidationError

    from amt.schemas.feature_recipe import FeatureRecipe, FeatureTransform, RecipeSource

    with pytest.raises(ValidationError):
        FeatureRecipe(
            sources=[RecipeSource(name="i", role="interactions", split="train")],
            feature_columns=[
                FeatureTransform(name="pop", kind="count", of="item_id", over="train")
            ],
            joins=[{"source": "i", "keys": ["item_id"]}],
        )


def test_an_attribute_source_no_longer_declares_join_keys():
    from pydantic import ValidationError

    from amt.schemas.feature_recipe import RecipeSource

    with pytest.raises(ValidationError):
        RecipeSource(
            name="attrs",
            role="item_attributes",
            table="cat.sch.items",
            columns=["genre"],
            join_keys=["item_id"],
        )
