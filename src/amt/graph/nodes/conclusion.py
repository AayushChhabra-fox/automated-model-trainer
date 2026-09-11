"""Stage 11/12 — scientific conclusion, final MLflow run, HC4."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.types import interrupt

from amt.graph.deps import NodeDeps
from amt.graph.human import parse_resume
from amt.graph.state import CampaignState, Stage
from amt.schemas.qa import QADecision


def make_conclusion_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Write the report. A negative result is a valid outcome."""

    def conclusion_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        decision = state.get("qa_decision") or QADecision.INVALID_EXPERIMENT
        conclusion = deps.planner_agent.write_conclusion(
            campaign_id=state.get("campaign_id", ""),
            spec=state.get("problem_spec"),
            contract=contract,
            history=list(state.get("experiment_history") or []),
            champions=list(state.get("hypothesis_champions") or []),
            assessment=state.get("qa_assessment"),
            phase0=state.get("qa_phase0"),
            phase1=state.get("qa_phase1"),
        )
        conclusion = conclusion.model_copy(
            update={
                "campaign_id": state.get("campaign_id", ""),
                "qa_decision": decision,
                "contract_hash": state.get("contract_hash", ""),
                "mlflow_experiment_id": state.get("mlflow_experiment_id", ""),
                "negative_result": decision != QADecision.PASS,
            }
        )
        status = state.get("status")
        if decision == QADecision.SPEC_REVIEW:
            status = "spec_review"
        return {
            "stage": Stage.CONCLUSION.value,
            "conclusion": conclusion,
            "status": status or "running",
        }

    return conclusion_node


def make_log_final_mlflow_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Stage 12 — log the final run with full lineage (skipped when MLflow is disabled)."""

    def log_final_mlflow_node(state: CampaignState) -> dict:
        if not deps.use_mlflow:
            return {}
        frozen = state.get("frozen_champion")
        conclusion = state.get("conclusion")
        if frozen is None or conclusion is None:
            return {}
        from amt.databricks.mlflow_client import log_final_run

        run_id = log_final_run(
            campaign_id=state.get("campaign_id", ""),
            experiment_id=state.get("mlflow_experiment_id", ""),
            frozen=frozen,
            conclusion=conclusion,
            problem_spec=state.get("problem_spec"),
            evaluation_contract=state.get("evaluation_contract"),
            phase1=state.get("qa_phase1"),
            code_version=deps.code_version,
            settings=deps.settings,
        )
        return {
            "mlflow_final_run_id": run_id,
            "conclusion": conclusion.model_copy(update={"mlflow_final_run_id": run_id}),
        }

    return log_final_mlflow_node


def make_hc_conclusion_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """HC4 — present the conclusion and record the human's promotion decision."""

    def hc_conclusion_node(state: CampaignState) -> dict:
        conclusion = state.get("conclusion")
        payload = interrupt(
            {
                "checkpoint": "HC4",
                "title": "Scientific conclusion & promotion decision",
                "campaign_id": state.get("campaign_id"),
                "qa_decision": state.get("qa_decision").value
                if state.get("qa_decision")
                else None,
                "conclusion": conclusion.model_dump(mode="json") if conclusion else None,
                "instructions": "amt promote <campaign> | amt reject <campaign> -m '<reason>'",
            }
        )
        decision = parse_resume(payload)
        status = "spec_review" if state.get("status") == "spec_review" else "done"
        return {
            "stage": Stage.DONE.value,
            "status": status,
            "awaiting": None,
            "promotion_decision": decision.decision
            or ("promote" if decision.approved else "hold"),
            "human_feedback": decision.feedback,
        }

    return hc_conclusion_node
