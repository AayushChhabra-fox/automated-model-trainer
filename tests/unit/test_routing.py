"""Routing is where the scientific constraints are actually enforced."""

from __future__ import annotations

import pytest

from amt.graph.routing import (
    PHASE0_MAX_ATTEMPTS,
    ContractViolation,
    assert_contract_hash,
    assert_not_frozen,
    assert_ready_for_qa,
    route_after_hc_contract,
    route_after_hc_hypotheses,
    route_after_hc_problem_spec,
    route_after_planner_select,
    route_after_qa_decision,
    route_after_qa_phase0,
    route_after_trainer_interpret,
    trainer_should_continue,
)
from amt.schemas.common import ExperimentBudget
from amt.schemas.hypothesis import ExperimentPlan
from amt.schemas.qa import Phase0Result, QADecision


def base_state(contract, **overrides):
    state = {
        "evaluation_contract": contract,
        "contract_hash": contract.contract_hash,
        "budget": ExperimentBudget(),
        "iteration": 0,
    }
    state.update(overrides)
    return state


def plan(max_iterations=3):
    return ExperimentPlan(
        hypothesis_id="hyp-1",
        hypothesis_summary="s",
        rationale="r",
        primary_metric="ndcg@10",
        max_iterations=max_iterations,
    )


# --- human checkpoints -------------------------------------------------------------
def test_hc2_rejection_returns_to_problem_spec():
    assert route_after_hc_problem_spec({"problem_spec_approved": False}) == "problem_spec"
    assert route_after_hc_problem_spec({"problem_spec_approved": True}) == "evaluation_contract"


def test_hc3_rejection_returns_to_contract():
    assert route_after_hc_contract({"contract_approved": False}) == "evaluation_contract"
    assert route_after_hc_contract({"contract_approved": True}) == "run_baseline"


def test_a_rejected_backlog_is_revised_before_anything_is_planned():
    assert route_after_hc_hypotheses({"hypotheses_approved": False}) == "planner_hypotheses"
    assert route_after_hc_hypotheses({"hypotheses_approved": True}) == "planner_select_hypothesis"


# --- contract hash ------------------------------------------------------------------
def test_contract_hash_must_match_the_value_stamped_at_hc3(evaluation_contract):
    state = base_state(evaluation_contract)
    assert assert_contract_hash(state) == evaluation_contract.contract_hash

    tampered = evaluation_contract.model_copy(update={"label_definition": "changed"})
    with pytest.raises(ContractViolation):
        assert_contract_hash(state, tampered)


def test_contract_hash_must_have_been_stamped(evaluation_contract):
    with pytest.raises(ContractViolation):
        assert_contract_hash({"evaluation_contract": evaluation_contract, "contract_hash": ""})
    with pytest.raises(ContractViolation):
        assert_contract_hash({"contract_hash": "abc"})


# --- trainer loop ---------------------------------------------------------------------
def test_trainer_loop_stops_at_max_iterations(evaluation_contract):
    state = base_state(evaluation_contract, experiment_plan=plan(3), iteration=3)
    should_continue, reason = trainer_should_continue(state)
    assert should_continue is False
    assert "max iterations" in reason
    assert route_after_trainer_interpret(state) == "select_hypothesis_champion"


def test_trainer_loop_continues_below_the_cap(evaluation_contract):
    state = base_state(evaluation_contract, experiment_plan=plan(3), iteration=1)
    assert route_after_trainer_interpret(state) == "trainer_propose"


def test_trainer_loop_stops_when_compute_budget_is_exhausted(evaluation_contract):
    budget = ExperimentBudget(max_compute_minutes=10, compute_minutes_used=10)
    state = base_state(evaluation_contract, experiment_plan=plan(5), iteration=1, budget=budget)
    should_continue, reason = trainer_should_continue(state)
    assert should_continue is False and "compute" in reason


def test_trainer_cannot_resume_after_freeze(evaluation_contract, frozen_champion):
    state = base_state(evaluation_contract, frozen_champion=frozen_champion)
    with pytest.raises(ContractViolation):
        assert_not_frozen(state)
    assert trainer_should_continue(state)[0] is False


def test_planner_routes_to_final_champion_when_no_hypothesis_is_selected():
    assert route_after_planner_select({"current_hypothesis_id": None}) == "planner_final_champion"
    assert route_after_planner_select({"current_hypothesis_id": "hyp-1"}) == "trainer_propose"


# --- QA gating --------------------------------------------------------------------------
def test_qa_cannot_run_before_freeze(evaluation_contract):
    with pytest.raises(ContractViolation):
        assert_ready_for_qa(base_state(evaluation_contract, frozen_champion=None))


def test_qa_rejects_a_champion_from_another_contract(evaluation_contract, frozen_champion):
    state = base_state(evaluation_contract, frozen_champion=frozen_champion)
    assert_ready_for_qa(state)  # consistent case

    state["contract_hash"] = "0" * 64
    with pytest.raises(ContractViolation):
        assert_ready_for_qa(state)


def test_phase0_retries_only_fixable_defects_and_only_up_to_the_cap():
    fixable = Phase0Result(passed=False, fixable=True, decision=QADecision.INVALID_EXPERIMENT)
    assert route_after_qa_phase0({"qa_phase0": fixable, "phase0_attempts": 1}) == "qa_phase0"
    assert (
        route_after_qa_phase0({"qa_phase0": fixable, "phase0_attempts": PHASE0_MAX_ATTEMPTS})
        == "conclusion"
    )

    unfixable = Phase0Result(passed=False, fixable=False, decision=QADecision.INVALID_EXPERIMENT)
    assert route_after_qa_phase0({"qa_phase0": unfixable, "phase0_attempts": 1}) == "conclusion"

    assert route_after_qa_phase0({"qa_phase0": Phase0Result(passed=True)}) == "qa_phase1"


def test_every_qa_decision_ends_at_the_conclusion():
    for decision in (QADecision.PASS, QADecision.FAIL_HYPOTHESIS, QADecision.SPEC_REVIEW):
        assert route_after_qa_decision({"qa_decision": decision}) == "conclusion"
    with pytest.raises(ContractViolation):
        route_after_qa_decision({})


# --- feature/fit checkpoint (HC_FEATURE) ----------------------------------------------
def _recipe():
    from amt.schemas.feature_recipe import (
        CandidateGeneration,
        FeatureRecipe,
        FeatureTransform,
        LabelSpec,
        RecipeSource,
    )

    return FeatureRecipe(
        sources=[RecipeSource(name="interactions", role="interactions", split="train")],
        feature_columns=[
            FeatureTransform(name="pop", kind="count", of="item_id", over="train"),
        ],
        label=LabelSpec(strategy="holdout_next_item"),
        candidate_generation=CandidateGeneration(strategy="popularity_topn", n=200),
    )


def _approved_state(contract, family="lightgbm_reranker", **overrides):
    from amt.schemas.experiment_config import ExperimentConfig
    from amt.schemas.feature_recipe import FitSpec, feature_fit_hash

    recipe, fit = _recipe(), FitSpec(booster_params={"learning_rate": 0.1})
    approved = feature_fit_hash(recipe, fit)
    state = base_state(
        contract,
        current_hypothesis_id="hyp-1",
        experiment_plan=plan().model_copy(update={"family": family}),
        approved_feature_recipe=recipe,
        approved_fit_spec=fit,
        feature_fit_hash=approved,
        feature_fit_approved=True,
        current_config=ExperimentConfig(
            family=family,
            feature_recipe=recipe,
            fit_spec=fit,
            feature_fit_hash=approved,
            contract_hash=contract.contract_hash,
        ),
    )
    state.update(overrides)
    return state


def test_a_tabular_plan_goes_to_the_feature_checkpoint_before_training(evaluation_contract):
    from amt.graph.routing import route_after_planner_select

    state = base_state(
        evaluation_contract,
        current_hypothesis_id="hyp-1",
        experiment_plan=plan().model_copy(update={"family": "xgboost_classifier"}),
    )
    assert route_after_planner_select(state) == "planner_feature_fit"


def test_the_fixed_shape_families_skip_the_feature_checkpoint(evaluation_contract):
    from amt.graph.routing import route_after_planner_select

    for family in ("", "popularity", "sasrec"):
        state = base_state(
            evaluation_contract,
            current_hypothesis_id="hyp-1",
            experiment_plan=plan().model_copy(update={"family": family}),
        )
        assert route_after_planner_select(state) == "trainer_propose"


def test_an_approved_shape_goes_straight_to_the_trainer(evaluation_contract):
    from amt.graph.routing import route_after_planner_select, route_after_trainer_propose

    state = _approved_state(evaluation_contract, current_config=None)
    assert route_after_planner_select(state) == "trainer_propose"
    assert route_after_trainer_propose(_approved_state(evaluation_contract)) == "execute_training"


def test_retuning_hyperparameters_does_not_need_re_approval(evaluation_contract):
    from amt.graph.routing import route_after_trainer_propose

    state = _approved_state(evaluation_contract)
    config = state["current_config"]
    state["current_config"] = config.model_copy(
        update={
            "fit_spec": config.fit_spec.model_copy(
                update={"booster_params": {"learning_rate": 0.01, "max_depth": 9}}
            ),
            "model": {"max_depth": 9},
        }
    )
    assert route_after_trainer_propose(state) == "execute_training"


def test_changing_the_shape_returns_to_the_feature_checkpoint(evaluation_contract):
    from amt.graph.routing import route_after_trainer_propose
    from amt.schemas.feature_recipe import FeatureTransform, feature_fit_hash

    state = _approved_state(evaluation_contract)
    config = state["current_config"]
    wider = config.feature_recipe.model_copy(
        update={
            "feature_columns": [
                *config.feature_recipe.feature_columns,
                FeatureTransform(name="bias", kind="literal", value=1.0),
            ]
        }
    )
    state["current_config"] = config.model_copy(
        update={
            "feature_recipe": wider,
            "feature_fit_hash": feature_fit_hash(wider, config.fit_spec),
        }
    )
    assert route_after_trainer_propose(state) == "planner_feature_fit"


def test_hc_feature_rejection_redrafts_the_shape():
    from amt.graph.routing import route_after_hc_feature_fit

    assert route_after_hc_feature_fit({"feature_fit_approved": False}) == "planner_feature_fit"
    assert route_after_hc_feature_fit({"feature_fit_approved": True}) == "trainer_propose"


def test_a_booster_cannot_train_on_an_unapproved_shape(evaluation_contract):
    from amt.graph.routing import assert_feature_fit_approved

    state = _approved_state(evaluation_contract)
    assert_feature_fit_approved(state)  # consistent case

    unapproved = _approved_state(evaluation_contract, feature_fit_approved=False)
    with pytest.raises(ContractViolation):
        assert_feature_fit_approved(unapproved)

    tampered = _approved_state(evaluation_contract)
    tampered["current_config"] = tampered["current_config"].model_copy(
        update={"feature_fit_hash": "0" * 64}
    )
    with pytest.raises(ContractViolation):
        assert_feature_fit_approved(tampered)


def test_the_fixed_shape_families_are_not_gated_on_a_recipe(evaluation_contract):
    from amt.graph.routing import assert_feature_fit_approved
    from amt.schemas.experiment_config import ExperimentConfig

    state = base_state(
        evaluation_contract,
        current_config=ExperimentConfig(family="sasrec"),
        feature_fit_approved=False,
    )
    assert_feature_fit_approved(state)
