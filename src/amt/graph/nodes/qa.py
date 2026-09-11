"""Stage 9/10 — independent QA. Runs only on the frozen champion, never before freeze."""

from __future__ import annotations

from collections.abc import Callable

from amt.graph.deps import NodeDeps
from amt.graph.routing import (
    PHASE0_MAX_ATTEMPTS,
    ContractViolation,
    assert_contract_hash,
    assert_ready_for_qa,
)
from amt.graph.state import CampaignState, Stage
from amt.schemas.qa import QADecision


def make_qa_phase0_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Phase 0 — integrity & reproducibility, before any QA performance is observed."""

    def qa_phase0_node(state: CampaignState) -> dict:
        assert_ready_for_qa(state)
        contract = state.get("evaluation_contract")
        frozen = state.get("frozen_champion")
        attempts = int(state.get("phase0_attempts", 0) or 0) + 1

        result = deps.executor.run_qa_phase0(frozen, contract)
        result = result.model_copy(update={"attempt": attempts})
        if result.contract_hash and result.contract_hash != state.get("contract_hash"):
            raise ContractViolation("QA Phase 0 ran against a different EvaluationContract.")

        update: dict = {
            "stage": Stage.QA_PHASE0.value,
            "qa_phase0": result,
            "phase0_attempts": attempts,
        }
        if result.passed:
            return update

        triage = deps.qa_agent.triage_phase0(result, frozen, contract)
        result = result.model_copy(
            update={
                "fixable": triage.fixable,
                "defects": triage.defects or result.defects,
                "fix_applied": triage.fix_plan if triage.fixable else "",
                "decision": QADecision.INVALID_EXPERIMENT,
            }
        )
        update["qa_phase0"] = result
        if not triage.fixable or attempts >= PHASE0_MAX_ATTEMPTS:
            # Out of repairs: the experiment is invalid, not the hypothesis.
            update["qa_decision"] = QADecision.INVALID_EXPERIMENT
            update["status"] = "aborted"
        return update

    return qa_phase0_node


def make_qa_phase1_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Phase 1 — holdout evaluation. Only reachable once Phase 0 has passed."""

    def qa_phase1_node(state: CampaignState) -> dict:
        assert_ready_for_qa(state)
        phase0 = state.get("qa_phase0")
        if phase0 is None or not phase0.passed:
            raise ContractViolation("QA Phase 1 requires a passing Phase 0.")
        contract = state.get("evaluation_contract")
        frozen = state.get("frozen_champion")
        baseline_run_id = state.get("baseline_run_id") or ""
        if not baseline_run_id:
            # Phase 1 derives `runs:/{id}/model` from this; without it the job
            # would have to refit the baseline, which needs the train split it
            # is deliberately not granted.
            raise ContractViolation(
                "QA Phase 1 needs the baseline MLflow run id; the baseline was "
                "never evaluated under this contract."
            )
        result = deps.executor.run_qa_phase1(frozen, baseline_run_id, contract)
        if result.contract_hash and result.contract_hash != state.get("contract_hash"):
            raise ContractViolation("QA Phase 1 ran against a different EvaluationContract.")
        return {"stage": Stage.QA_PHASE1.value, "qa_phase1": result}

    return qa_phase1_node


def make_qa_decision_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Stage 10 — PASS / FAIL_HYPOTHESIS / SPEC_REVIEW. QA never optimizes."""

    def qa_decision_node(state: CampaignState) -> dict:
        assert_ready_for_qa(state)
        assert_contract_hash(state)
        phase0 = state.get("qa_phase0")
        phase1 = state.get("qa_phase1")
        if phase1 is None:
            raise ContractViolation("No Phase 1 evidence to decide on.")
        assessment = deps.qa_agent.decide(
            phase0=phase0, phase1=phase1, contract=state.get("evaluation_contract"),
            frozen=state.get("frozen_champion"),
        )
        if assessment.decision == QADecision.INVALID_EXPERIMENT:
            raise ContractViolation(
                "Only Phase 0 may return INVALID_EXPERIMENT; Phase 1 evidence cannot."
            )
        update: dict = {
            "stage": Stage.QA_DECISION.value,
            "qa_assessment": assessment,
            "qa_decision": assessment.decision,
        }
        if assessment.decision == QADecision.SPEC_REVIEW:
            # Ground truth is suspect: stop, do not reopen the parameter search.
            update["status"] = "spec_review"
        return update

    return qa_decision_node
