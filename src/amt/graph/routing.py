"""Conditional edges and hard guards (PLAN §4, §9).

These rules are code, not prompts. An agent cannot talk its way past them:

* no hypothesis is planned before a human approves the backlog;
* the trainer loop cannot exceed ``plan.max_iterations`` (or the campaign budget);
* a tabular booster never trains on a feature/fit shape a human has not approved;
* every stage after HC3 asserts the frozen ``contract_hash``;
* QA cannot run before champion freeze, and never with more than one champion;
* Phase 0 may be retried at most ``settings.phase0_max_attempts`` times;
* ``SPEC_REVIEW`` routes to the conclusion/human checkpoint, never back into training.

Nothing here reads settings or the environment: every rule is a function of state, so a
router can be unit-tested by writing the state that should trigger it. Caps that come from
configuration are stamped into state by the node they govern.
"""

from __future__ import annotations

from amt.graph.state import CampaignState
from amt.schemas.champion import FrozenChampion
from amt.schemas.common import ExperimentBudget
from amt.schemas.evaluation_contract import EvaluationContract
from amt.schemas.feature_recipe import is_tabular_family


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


def target_family(state: CampaignState) -> str:
    """The family about to run: what the Trainer asked for, else what the plan declares."""
    config = state.get("current_config")
    if config is not None and config.family:
        return str(config.family)
    plan = state.get("experiment_plan")
    return str(getattr(plan, "family", "") or "")


def feature_fit_needed(state: CampaignState) -> bool:
    """True when a tabular family has no approved feature/fit shape to run under.

    Nothing routes on this any more. HC_FEATURE approves every tabular shape in
    the backlog before the first hypothesis is planned, and Trainer drift is
    clamped to the approved shape rather than bounced back to the gate, so a
    true answer here is a bug and the caller raises.
    """
    if not is_tabular_family(target_family(state)):
        return False
    return not (state.get("feature_fit_approved") and state.get("feature_fit_hash"))


def assert_feature_fit_approved(state: CampaignState) -> None:
    """A booster job may only be set up on the exact shape a human approved."""
    config = state.get("current_config")
    family = str(getattr(config, "family", "") or "") or target_family(state)
    if not is_tabular_family(family):
        return
    if not state.get("feature_fit_approved"):
        raise ContractViolation(
            f"family {family!r} requires an approved FeatureRecipe/FitSpec; "
            "the feature/fit checkpoint has not been passed."
        )
    approved = str(state.get("feature_fit_hash") or "")
    if config is None or not approved:
        raise ContractViolation("No approved feature/fit shape to train under.")
    if config.feature_fit_hash != approved:
        raise ContractViolation(
            "ExperimentConfig runs a different feature/fit shape than the approved one: "
            f"{config.feature_fit_hash[:12]} != {approved[:12]}."
        )
    if config.feature_recipe is None:
        raise ContractViolation("Tabular config carries no FeatureRecipe.")


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


def route_after_hc_hypotheses(state: CampaignState) -> str:
    """HC_BACKLOG: rejected → revise; approved → review shapes, or plan straight away.

    A backlog with nothing tabular in it never sees HC_FEATURE: there is no
    shape to approve, and interrupting on an empty document is how an operator
    learns to approve without reading.
    """
    if not state.get("hypotheses_approved"):
        return "planner_hypotheses"
    tabular = any(
        h.status == "pending" and is_tabular_family(h.family) for h in state.get("hypotheses") or []
    )
    return "planner_feature_fit" if tabular else "planner_select_hypothesis"


def route_after_planner_select(state: CampaignState) -> str:
    """No hypothesis left (or budget spent) → pick the final champion."""
    if not state.get("current_hypothesis_id"):
        return "planner_final_champion"
    if feature_fit_needed(state):
        raise ContractViolation(
            "A tabular hypothesis reached the Trainer with no approved shape. "
            "HC_FEATURE approves every shape before selection, so there is "
            "nothing a human can approve to fix this; the selection node is wrong."
        )
    return "trainer_propose"


def route_after_hc_feature_fit(state: CampaignState) -> str:
    """HC_FEATURE: rejected → redraft every shape; approved → plan the first hypothesis.

    Approval covers the whole backlog, so the next step is selection rather
    than the Trainer: which hypothesis runs first is still the Planner's call.
    Striking every shape out is an approval, not a rejection — the operator
    decided none of them is worth compute, and the campaign continues with
    whatever non-tabular hypotheses are left.
    """
    return "planner_select_hypothesis" if state.get("shapes_approved") else "planner_feature_fit"


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


def route_after_hc_champion(state: CampaignState) -> str:
    """HC_CHAMPION: ``freeze`` stops exploring; anything else plans the next hypothesis.

    ``freeze`` is a shortcut through selection, not around it: the node has
    already abandoned every pending hypothesis, so routing to
    `planner_select_hypothesis` would reach the final champion anyway. Going
    straight there makes the operator's intent legible in the graph.
    """
    if state.get("freeze_requested"):
        return "planner_final_champion"
    return "planner_select_hypothesis"


def phase0_max_attempts(state: CampaignState) -> int:
    """The Phase 0 retry cap, as ``qa_phase0_node`` stamped it from settings.

    Read from state rather than from ``settings`` so this module stays pure and
    so a resumed campaign keeps the cap it started under -- editing
    ``AMT_PHASE0_MAX_ATTEMPTS`` mid-campaign must not change how many repairs a
    champion already part-way through QA is allowed.
    """
    cap = int(state.get("phase0_max_attempts", 0) or 0)
    if cap < 1:
        raise ContractViolation(
            "phase0_max_attempts is not in state; qa_phase0 stamps it from "
            "settings alongside the result this router reads."
        )
    return cap


def route_after_qa_phase0(state: CampaignState) -> str:
    """Phase 0 gates Phase 1. Fixable defects may be retried up to the attempt cap."""
    result = state.get("qa_phase0")
    if result is None:
        raise ContractViolation("qa_phase0 produced no result.")
    if result.passed:
        return "qa_phase1"
    attempts = int(state.get("phase0_attempts", 0) or 0)
    if result.fixable and attempts < phase0_max_attempts(state):
        return "qa_phase0"
    return "conclusion"
