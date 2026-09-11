"""Conditional edges and hard guards (PLAN §4, §9).

These rules are code, not prompts. An agent cannot talk its way past them:

* the trainer loop cannot exceed ``plan.max_iterations`` (or the campaign budget);
* every stage after HC3 asserts the frozen ``contract_hash``;
* QA cannot run before champion freeze, and never with more than one champion;
* Phase 0 may be retried at most :data:`PHASE0_MAX_ATTEMPTS` times;
* ``SPEC_REVIEW`` routes to the conclusion/human checkpoint, never back into training.
"""

from __future__ import annotations

from amt.graph.state import CampaignState
from amt.schemas.champion import FrozenChampion
from amt.schemas.common import ExperimentBudget
from amt.schemas.evaluation_contract import EvaluationContract
from amt.schemas.qa import QADecision

PHASE0_MAX_ATTEMPTS = 3


class ContractViolation(RuntimeError):
    """Raised when a scientific constraint from SPEC.md would be broken."""


# --- guards -------------------------------------------------------------------------
def assert_contract_hash(state: CampaignState, contract: EvaluationContract | None = None) -> str:
    """Assert the live contract still hashes to the value stamped at HC3."""
    frozen_hash = state.get("contract_hash") or ""
    contract = contract if contract is not None else state.get("evaluation_contract")
    if contract is None:
        raise ContractViolation("No EvaluationContract in state.")
    if not frozen_hash:
        raise ContractViolation("contract_hash was never stamped (HC3 not reached).")
    if contract.contract_hash != frozen_hash:
        raise ContractViolation(
            "EvaluationContract changed after approval: "
            f"{contract.contract_hash[:12]} != {frozen_hash[:12]}. "
            "Ground truth must not change inside a campaign; return to Stage 1/2."
        )
    return frozen_hash


def assert_not_frozen(state: CampaignState) -> None:
    """Trainer work is forbidden once a champion is frozen."""
    if state.get("frozen_champion") is not None:
        raise ContractViolation("Training cannot resume after champion freeze.")


def assert_ready_for_qa(state: CampaignState) -> None:
    """QA runs only on exactly one frozen champion, under the frozen contract."""
    frozen = state.get("frozen_champion")
    if frozen is None:
        raise ContractViolation("QA cannot run before champion freeze.")
    if not isinstance(frozen, FrozenChampion):
        raise ContractViolation(
            "QA accepts exactly one immutable FrozenChampion, never a collection of candidates."
        )
    stamped = state.get("contract_hash") or ""
    if frozen.contract_hash != stamped:
        raise ContractViolation(
            "Frozen champion references a different EvaluationContract than the campaign."
        )
    assert_contract_hash(state)


def assert_single_champion(state: CampaignState) -> None:
    """Never submit multiple candidates to QA."""
    if state.get("frozen_champion") is None:
        raise ContractViolation("No frozen champion.")


# --- conditional edges ---------------------------------------------------------------
def route_after_hc_problem_spec(state: CampaignState) -> str:
    """HC2: rejected → redraft the spec; approved → build the EvaluationContract."""
    return "evaluation_contract" if state.get("problem_spec_approved") else "problem_spec"


def route_after_hc_contract(state: CampaignState) -> str:
    """HC3: rejected → redraft the contract; approved → baseline, then autonomous."""
    return "run_baseline" if state.get("contract_approved") else "evaluation_contract"


def route_after_planner_select(state: CampaignState) -> str:
    """No hypothesis left (or budget spent) → pick the final champion."""
    if state.get("current_hypothesis_id"):
        return "trainer_propose"
    return "planner_final_champion"


def _plan_max_iterations(state: CampaignState) -> int:
    plan = state.get("experiment_plan")
    budget: ExperimentBudget = state.get("budget") or ExperimentBudget()
    return plan.max_iterations if plan else budget.max_iterations


def trainer_should_continue(state: CampaignState) -> tuple[bool, str]:
    """Convergence check. Never based on training loss alone."""
    if state.get("frozen_champion") is not None:
        return False, "champion already frozen"
    if state.get("trainer_converged"):
        return False, state.get("convergence_reason") or "trainer reported convergence"
    iteration = int(state.get("iteration", 0) or 0)
    max_iterations = _plan_max_iterations(state)
    if iteration >= max_iterations:
        return False, f"max iterations reached ({iteration}/{max_iterations})"
    budget: ExperimentBudget = state.get("budget") or ExperimentBudget()
    if budget.compute_exhausted:
        return False, "compute budget exhausted"
    return True, ""


def route_after_trainer_interpret(state: CampaignState) -> str:
    should_continue, _ = trainer_should_continue(state)
    return "trainer_propose" if should_continue else "select_hypothesis_champion"


def route_after_qa_phase0(state: CampaignState) -> str:
    """Phase 0 gates Phase 1. Fixable defects may be retried up to the attempt cap."""
    result = state.get("qa_phase0")
    if result is None:
        raise ContractViolation("qa_phase0 produced no result.")
    if result.passed:
        return "qa_phase1"
    attempts = int(state.get("phase0_attempts", 0) or 0)
    if result.fixable and attempts < PHASE0_MAX_ATTEMPTS:
        return "qa_phase0"
    return "conclusion"


def route_after_qa_decision(state: CampaignState) -> str:
    """Every QA decision — PASS, FAIL_HYPOTHESIS, SPEC_REVIEW — ends at the conclusion (HC4)."""
    decision = state.get("qa_decision")
    if decision is None:
        raise ContractViolation("qa_decision produced no decision.")
    if decision not in set(QADecision):
        raise ContractViolation(f"Unknown QA decision: {decision!r}")
    return "conclusion"
