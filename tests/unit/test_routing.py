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
