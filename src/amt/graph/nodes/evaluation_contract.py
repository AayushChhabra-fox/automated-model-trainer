"""Stage 2 nodes: draft the contract, materialize splits, HC3 freeze, baseline run."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.types import interrupt

from amt.graph.deps import NodeDeps
from amt.graph.human import gate_update, parse_resume
from amt.graph.routing import ContractViolation, assert_contract_hash
from amt.graph.state import CampaignState, Stage


def make_evaluation_contract_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Draft (or redraft) the single EvaluationContract for the campaign."""

    def evaluation_contract_node(state: CampaignState) -> dict:
        spec = state.get("problem_spec")
        if spec is None:
            raise ContractViolation("EvaluationContract requires an approved ProblemSpec.")
        contract = deps.problem_spec_agent.draft_contract(
            spec=spec,
            previous=state.get("evaluation_contract"),
            human_feedback=state.get("human_feedback"),
        )
        return {
            "stage": Stage.EVALUATION_CONTRACT.value,
            "status": "running",
            "evaluation_contract": contract,
            "contract_approved": False,
            "contract_hash": "",
            "human_feedback": None,
        }

    return evaluation_contract_node


def make_materialize_splits_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Materialize immutable splits and fold their references back into the contract."""

    def materialize_splits_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        if contract is None:
            raise ContractViolation("No EvaluationContract to materialize.")
        materialization = deps.executor.materialize_splits(contract)
        contract = contract.model_copy(
            update={
                "train_split": materialization.train,
                "validation_split": materialization.validation,
                "qa_split": materialization.qa,
            }
        )
        return {
            "evaluation_contract": contract,
            "split_materialization": materialization,
        }

    return materialize_splits_node


def make_hc_contract_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """HC3 — approve the contract. Approval stamps the campaign's ``contract_hash``."""

    def hc_contract_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        payload = interrupt(
            {
                "checkpoint": "HC3",
                "title": "EvaluationContract approval",
                "campaign_id": state.get("campaign_id"),
                "evaluation_contract": contract.model_dump(mode="json") if contract else None,
                "contract_hash": contract.contract_hash if contract else None,
                "instructions": "amt approve <campaign> | amt reject <campaign> -m '<feedback>'",
            }
        )
        decision = parse_resume(payload)
        update: dict = {
            "contract_approved": decision.approved,
            **gate_update(state, decision),
        }
        if decision.approved and contract is not None:
            # Ground truth is frozen here; every later node and job asserts this hash.
            update["contract_hash"] = contract.contract_hash
        return update

    return hc_contract_node


def make_run_baseline_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Evaluate the baseline under the frozen contract (same ground truth as candidates)."""

    def run_baseline_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        spec = state.get("problem_spec")
        assert_contract_hash(state, contract)
        result = deps.executor.run_baseline(spec, contract, "validation")
        update: dict = {
            "stage": Stage.BASELINE.value,
            "baseline_run_id": result.mlflow_run_id,
        }
        if deps.use_mlflow and not state.get("mlflow_experiment_id"):
            from amt.databricks.mlflow_client import get_or_create_experiment

            update["mlflow_experiment_id"] = get_or_create_experiment(
                state.get("campaign_id", ""), deps.settings
            )
        elif not state.get("mlflow_experiment_id"):
            update["mlflow_experiment_id"] = result.mlflow_experiment_id
        return update

    return run_baseline_node
