"""Stage 7a (hypothesis champion) and Stage 8 (champion freeze)."""

from __future__ import annotations

from collections.abc import Callable

from amt.graph.deps import NodeDeps
from amt.graph.routing import ContractViolation, assert_contract_hash
from amt.graph.state import CampaignState, Stage, current_hypothesis
from amt.schemas.champion import FrozenChampion, HypothesisChampion


def make_select_hypothesis_champion_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Exactly one best configuration per hypothesis, by predefined validation criteria."""

    def select_hypothesis_champion_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        assert_contract_hash(state, contract)
        hypothesis = current_hypothesis(state)
        plan = state.get("experiment_plan")
        if hypothesis is None or plan is None:
            raise ContractViolation("No hypothesis in flight to select a champion for.")

        hid = hypothesis.hypothesis_id
        runs = [r for r in (state.get("experiment_history") or []) if r.hypothesis_id == hid]
        successful = [
            r for r in runs if r.verdict != "failed" and r.primary_metric_value is not None
        ]
        champions = list(state.get("hypothesis_champions") or [])
        hypotheses = list(state.get("hypotheses") or [])

        def close_hypothesis(status: str) -> list:
            return [
                h.model_copy(update={"status": status}) if h.hypothesis_id == hid else h
                for h in hypotheses
            ]

        if not successful:
            return {
                "stage": Stage.HYPOTHESIS_CHAMPION.value,
                "hypotheses": close_hypothesis("abandoned"),
                "current_hypothesis_id": None,
                "experiment_plan": None,
                "run_results": {},
                "last_result": None,
                "last_interpretation": None,
            }

        selection = deps.trainer_agent.select_champion(hypothesis, plan, successful, contract)
        chosen = next((r for r in successful if r.run_id == selection.run_id), None)
        if chosen is None:
            # The agent named a run outside this hypothesis: fall back to the best validated run.
            chosen = max(successful, key=lambda r: r.primary_metric_value or float("-inf"))

        results = state.get("run_results") or {}
        result = results.get(chosen.run_id)
        champion = HypothesisChampion(
            hypothesis_id=hid,
            run_id=chosen.run_id,
            mlflow_run_id=chosen.mlflow_run_id,
            config=result.config if result is not None and result.config is not None
            else state.get("current_config"),
            config_hash=chosen.config_hash,
            primary_metric=contract.primary_metric,
            primary_metric_value=chosen.primary_metric_value or 0.0,
            secondary_metrics=dict(result.validation_metrics) if result is not None else {},
            cohort_metrics=dict(result.cohort_metrics) if result is not None else {},
            cohort_regressions=selection.cohort_regressions,
            guardrails_passed=selection.guardrails_passed,
            complexity_notes=selection.complexity_notes,
            operational_notes=selection.operational_notes,
            selection_rationale=selection.selection_rationale,
            model_artifact_uri=result.model_artifact_uri if result is not None else None,
        )
        return {
            "stage": Stage.HYPOTHESIS_CHAMPION.value,
            "hypothesis_champions": [*champions, champion],
            "hypotheses": close_hypothesis("converged"),
            "current_hypothesis_id": None,
            "experiment_plan": None,
            "run_results": {},
            "last_result": None,
            "last_interpretation": None,
        }

    return select_hypothesis_champion_node


def make_freeze_champion_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Stage 8 — freeze exactly one candidate. Nothing about it may change before QA."""

    def freeze_champion_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        assert_contract_hash(state, contract)
        if state.get("frozen_champion") is not None:
            raise ContractViolation("A champion is already frozen for this campaign.")
        champions = list(state.get("hypothesis_champions") or [])
        selected_id = state.get("current_hypothesis_id")
        champion = next((c for c in champions if c.hypothesis_id == selected_id), None)
        if champion is None:
            raise ContractViolation("Final champion selection did not resolve to a champion.")

        config = champion.config
        frozen = FrozenChampion(
            hypothesis_id=champion.hypothesis_id,
            run_id=champion.run_id,
            mlflow_run_id=champion.mlflow_run_id,
            config=config,
            config_hash=champion.config_hash or config.config_hash,
            family=config.family,
            architecture=dict(config.model),
            features=dict(config.features),
            hyperparameters={**config.training, **config.loss},
            objective=str(config.loss.get("type", "")),
            preprocessing_ref=f"{champion.mlflow_run_id}:preprocessing",
            checkpoint_uri=champion.model_artifact_uri,
            model_artifact_uri=champion.model_artifact_uri,
            validation_metrics=dict(champion.secondary_metrics),
            cohort_metrics=dict(champion.cohort_metrics),
            contract_hash=state.get("contract_hash", ""),
            spec_hash=state.get("spec_hash", ""),
            harness_version=contract.harness_version,
        )
        return {
            "stage": Stage.FREEZE.value,
            "frozen_champion": frozen,
        }

    return freeze_champion_node
