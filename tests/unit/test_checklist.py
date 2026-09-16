"""Checklist verbs at a gate, and the clamp that replaced the bounce loop.

These are the two halves of the gate redesign: a human edits the list instead
of rejecting it, and the Trainer's structural asks are refused in place instead
of costing a review round.
"""

from __future__ import annotations

import pytest

from amt.graph.human import (
    HumanDecision,
    UnknownHypothesis,
    apply_checklist,
    parse_resume,
)
from amt.graph.nodes.trainer import _bind_feature_fit, _structural_asks
from amt.graph.routing import ContractViolation
from amt.schemas.experiment_config import ExperimentConfig
from amt.schemas.feature_recipe import (
    CandidateGeneration,
    FeatureRecipe,
    FeatureTransform,
    FitSpec,
    LabelSpec,
    RecipeSource,
    feature_fit_hash,
)
from amt.schemas.hypothesis import Hypothesis


def _hypothesis(hypothesis_id: str, priority: int = 1, family: str = "") -> Hypothesis:
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        title=f"hypothesis {hypothesis_id}",
        proposed_change="c",
        why_it_should_improve="w",
        motivating_evidence="e",
        experiment_design="d",
        metric_expected_to_improve="ndcg@10",
        falsification_criteria="f",
        priority=priority,
        family=family,
    )


BACKLOG = [
    _hypothesis("H1", 1, "lightgbm_reranker"),
    _hypothesis("H2", 2, "lightgbm_reranker"),
    _hypothesis("H3", 3, "sasrec"),
]


def _statuses(hypotheses):
    return {h.hypothesis_id: h.status for h in hypotheses}


# --- the verbs ------------------------------------------------------------


def test_a_plain_approval_changes_nothing():
    assert apply_checklist(BACKLOG, HumanDecision(approved=True)) == BACKLOG


def test_dropping_one_hypothesis_abandons_only_that_one():
    kept = apply_checklist(BACKLOG, HumanDecision(approved=True, drop=["H2"]))

    assert _statuses(kept) == {"H1": "pending", "H2": "abandoned", "H3": "pending"}


def test_keeping_a_subset_abandons_the_rest_and_sets_the_order():
    """`--keep H3,H1` is both a strike-out and a re-prioritisation."""
    kept = apply_checklist(BACKLOG, HumanDecision(approved=True, keep=["H3", "H1"]))

    assert _statuses(kept) == {"H1": "pending", "H2": "abandoned", "H3": "pending"}
    priorities = {h.hypothesis_id: h.priority for h in kept}
    assert priorities["H3"] == 1 and priorities["H1"] == 2


def test_a_typo_is_refused_rather_than_silently_abandoning_the_backlog():
    with pytest.raises(UnknownHypothesis, match="H9"):
        apply_checklist(BACKLOG, HumanDecision(approved=True, keep=["H1", "H9"]))


def test_keep_only_speaks_for_the_hypotheses_under_review():
    """HC_FEATURE reviews the tabular shapes; a SASRec hypothesis is not on trial."""
    kept = apply_checklist(BACKLOG, HumanDecision(approved=True, keep=["H1"]), scope=["H1", "H2"])

    assert _statuses(kept) == {"H1": "pending", "H2": "abandoned", "H3": "pending"}


def test_the_cli_comma_form_and_the_list_form_mean_the_same_thing():
    assert parse_resume({"approved": True, "keep": "H1, H3", "drop": "H2"}) == HumanDecision(
        approved=True, keep=["H1", "H3"], drop=["H2"]
    )
    assert parse_resume({"approved": True, "keep": ["H1"], "drop": []}).keep == ["H1"]
    assert parse_resume({"approved": True}).keep == []


def test_typing_what_the_form_placeholder_shows_is_not_a_typo():
    # The web form's placeholders read "keep H1,H3" / "drop H2"; an operator who
    # types exactly that must not be told H1 does not exist.
    decision = parse_resume({"approved": True, "keep": "keep H1,H2,H4", "drop": "drop H3"})
    assert decision.keep == ["H1", "H2", "H4"]
    assert decision.drop == ["H3"]
    assert parse_resume({"approved": True, "keep": "H1 H3; H4"}).keep == ["H1", "H3", "H4"]


# --- the clamp ------------------------------------------------------------


def _recipe(extra: FeatureTransform | None = None) -> FeatureRecipe:
    columns = [FeatureTransform(name="pop", kind="count", of="item_id", over="train")]
    if extra is not None:
        columns.append(extra)
    return FeatureRecipe(
        sources=[RecipeSource(name="interactions", role="interactions", split="train")],
        feature_columns=columns,
        label=LabelSpec(strategy="holdout_next_item"),
        candidate_generation=CandidateGeneration(strategy="popularity_topn", n=200),
    )


APPROVED_FIT = FitSpec(
    objective="lambdarank",
    eval_metric="ndcg",
    num_boost_round=200,
    negative_samples_per_positive=4,
    booster_params={"learning_rate": 0.1},
)


def _state(**overrides) -> dict:
    state = {
        "approved_feature_recipe": _recipe(),
        "approved_fit_spec": APPROVED_FIT,
        "feature_fit_hash": feature_fit_hash(_recipe(), APPROVED_FIT),
        "feature_fit_approved": True,
    }
    state.update(overrides)
    return state


def test_a_retune_is_bound_to_the_approved_shape():
    config = ExperimentConfig(
        family="lightgbm_reranker",
        fit_spec=APPROVED_FIT.model_copy(update={"booster_params": {"learning_rate": 0.01}}),
    )

    bound, clamped = _bind_feature_fit(_state(), config)

    assert clamped == []
    assert bound.fit_spec.booster_params == {"learning_rate": 0.01}
    assert bound.feature_fit_hash == feature_fit_hash(_recipe(), APPROVED_FIT)


def test_a_structural_ask_is_refused_in_place_and_recorded():
    """This is the bounce loop's replacement: no gate, no rewrite, one line."""
    config = ExperimentConfig(
        family="lightgbm_reranker",
        feature_recipe=_recipe(FeatureTransform(name="bias", kind="literal", value=1.0)),
        fit_spec=APPROVED_FIT.model_copy(
            update={"num_boost_round": 900, "objective": "rank:pairwise"}
        ),
    )

    bound, clamped = _bind_feature_fit(_state(), config)

    assert bound.feature_recipe.structural_payload == _recipe().structural_payload
    assert bound.fit_spec.num_boost_round == 200
    assert bound.fit_spec.objective == "lambdarank"
    assert bound.feature_fit_hash == feature_fit_hash(_recipe(), APPROVED_FIT)
    assert clamped[0].startswith("feature_recipe")
    assert any("num_boost_round=900" in ask for ask in clamped)
    assert any("rank:pairwise" in ask for ask in clamped)


def test_a_blank_objective_inherits_the_approved_one_and_is_not_drift():
    """B2: the Trainer routinely omits these, and omission is not an ask."""
    config = ExperimentConfig(
        family="lightgbm_reranker",
        fit_spec=FitSpec(objective="", eval_metric="", booster_params={"max_depth": 9}),
    )

    bound, clamped = _bind_feature_fit(_state(), config)

    assert bound.fit_spec.objective == "lambdarank"
    assert bound.fit_spec.eval_metric == "ndcg"
    assert [ask for ask in clamped if "objective" in ask or "eval_metric" in ask] == []


def test_a_fixed_shape_family_carries_no_recipe_at_all():
    bound, clamped = _bind_feature_fit(_state(), ExperimentConfig(family="sasrec"))

    assert (bound.feature_recipe, bound.fit_spec, bound.feature_fit_hash) == (None, None, "")
    assert clamped == []


def test_binding_without_an_approved_shape_is_a_violation():
    config = ExperimentConfig(family="lightgbm_reranker")

    with pytest.raises(ContractViolation, match="never legitimately"):
        _bind_feature_fit(_state(feature_fit_approved=False), config)


def test_only_non_blank_differences_count_as_asks():
    assert _structural_asks(APPROVED_FIT, None) == []
    assert _structural_asks(APPROVED_FIT, APPROVED_FIT) == []
    assert _structural_asks(APPROVED_FIT, APPROVED_FIT.model_copy(update={"seed": 7})) == [
        "fit_spec.seed=7 (kept 42)"
    ]


# --- rewinding an approved gate -------------------------------------------


def _mid_flight_state():
    from amt.schemas.common import ExperimentBudget

    hypotheses = [
        BACKLOG[0].model_copy(update={"status": "active"}),
        BACKLOG[1],
        BACKLOG[2].model_copy(update={"status": "abandoned"}),
    ]
    return {
        "hypotheses": hypotheses,
        "hypotheses_approved": True,
        "budget": ExperimentBudget(max_hypotheses=3, hypotheses_used=1, iterations_used=2),
        "approved_shapes": {"H1": object()},
        "approved_feature_recipe": object(),
        "feature_fit_hash": "abc",
        "feature_fit_approved": True,
        "current_hypothesis_id": "H1",
        "experiment_plan": object(),
        "experiment_history": ["kept"],
        "frozen_champion": None,
    }


def test_rewinding_to_the_feature_gate_reopens_the_shapes_and_returns_the_budget_slot():
    from amt.graph.human import rewind_update

    as_node, values = rewind_update(_mid_flight_state(), "feature", "recipe read missing columns")

    assert as_node == "hc_hypotheses", "re-enter through the approved backlog"
    assert values["approved_shapes"] == {} and values["approved_feature_recipe"] is None
    assert values["feature_fit_approved"] is False and values["feature_fit_hash"] == ""
    assert values["current_hypothesis_id"] is None and values["experiment_plan"] is None
    assert _statuses(values["hypotheses"]) == {"H1": "pending", "H2": "pending", "H3": "abandoned"}
    assert values["budget"].hypotheses_used == 0 and values["budget"].iterations_used == 0
    assert values["human_feedback"] == "recipe read missing columns"
    assert "hypotheses_approved" not in values, "the backlog approval stands"
    assert "experiment_history" not in values, "what ran is not erased"


def test_rewinding_to_the_backlog_unapproves_it():
    from amt.graph.human import rewind_update

    as_node, values = rewind_update(_mid_flight_state(), "backlog", "wrong ideas")

    assert as_node == "run_baseline"
    assert values["hypotheses_approved"] is False


def test_a_frozen_champion_cannot_be_rewound():
    from amt.graph.human import rewind_update

    state = {**_mid_flight_state(), "frozen_champion": object()}
    with pytest.raises(ValueError, match="frozen"):
        rewind_update(state, "feature", "too late")


def test_an_unknown_rewind_target_is_refused():
    from amt.graph.human import rewind_update

    with pytest.raises(ValueError, match="backlog"):
        rewind_update(_mid_flight_state(), "hc3", "no")


def test_rewinding_to_qa_keeps_the_frozen_champion_and_clears_only_qa():
    from amt.graph.human import rewind_update

    state = {**_mid_flight_state(), "frozen_champion": object(), "qa_decision": "INVALID_EXPERIMENT"}
    as_node, values = rewind_update(state, "qa", "pyfunc signature dropped the candidates")

    assert as_node == "freeze_champion"
    assert values["qa_phase0"] is None and values["qa_decision"] is None and values["conclusion"] is None
    assert values["phase0_attempts"] == 0
    assert "frozen_champion" not in values and "hypotheses" not in values, "the artifact is untouched"


def test_rewinding_to_qa_without_a_freeze_is_refused():
    from amt.graph.human import rewind_update

    with pytest.raises(ValueError, match="nothing is frozen"):
        rewind_update(_mid_flight_state(), "qa", "too early")
