"""Stage 7a (hypothesis champion) and Stage 8 (champion freeze)."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.types import interrupt

from amt.graph.deps import NodeDeps
from amt.graph.human import apply_checklist, gate_update, parse_resume
from amt.graph.routing import ContractViolation, assert_contract_hash
from amt.graph.state import CampaignState, Stage, current_hypothesis
from amt.schemas.champion import FrozenChampion, HypothesisChampion
from amt.schemas.common import ExperimentBudget


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


def make_hc_champion_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """HC_CHAMPION — a human reads each hypothesis's champion before the next one runs.

    Per hypothesis, not per iteration: the Trainer loop inside a hypothesis
    stays autonomous, which is the point of the band between HC3 and HC4. What
    a human gets to decide is whether the *next* hypothesis is worth its
    compute, which is the decision the previous campaign could only make by
    abandoning the run.
    """

    def hc_champion_node(state: CampaignState) -> dict:
        if not deps.gate_champions:
            return {}
        champions = list(state.get("hypothesis_champions") or [])
        if not champions:
            # The hypothesis produced nothing worth comparing; there is no
            # decision to put in front of anyone.
            return {}
        contract = state.get("evaluation_contract")
        budget = state.get("budget") or ExperimentBudget()
        hypotheses = list(state.get("hypotheses") or [])
        remaining = [h.hypothesis_id for h in hypotheses if h.status == "pending"]

        payload = interrupt(
            {
                "checkpoint": "HC_CHAMPION",
                "title": f"Champion for {champions[-1].hypothesis_id}",
                "campaign_id": state.get("campaign_id"),
                "hypothesis_id": champions[-1].hypothesis_id,
                "primary_metric": contract.primary_metric if contract else "",
                "comparison": _comparison(state, champions, contract),
                "cohort_deltas": _cohort_deltas(
                    state.get("baseline_cohort_metrics") or {},
                    champions[-1].cohort_metrics or {},
                    contract.primary_metric if contract else "",
                ),
                "cohort_metrics": dict(champions[-1].cohort_metrics or {}),
                "guardrails_passed": champions[-1].guardrails_passed,
                "cohort_regressions": list(champions[-1].cohort_regressions or []),
                "selection_rationale": champions[-1].selection_rationale,
                "hypotheses_remaining": remaining,
                "budget": {
                    "hypotheses_remaining": budget.hypotheses_remaining,
                    "compute_minutes_used": round(budget.compute_minutes_used, 1),
                    "max_compute_minutes": budget.max_compute_minutes,
                },
                "instructions": (
                    "amt approve <campaign> [--drop H4,H5] (continue) "
                    "| amt approve <campaign> --freeze (stop here and go to QA) "
                    "| amt reject <campaign> -m '<feedback>'"
                ),
            }
        )
        decision = parse_resume(payload)
        update: dict = {**gate_update(state, decision), "champion_reviewed": True}
        if decision.drop or decision.keep:
            update["hypotheses"] = apply_checklist(hypotheses, decision, scope=remaining)
        if decision.freeze:
            # Skip the rest of the backlog: whatever is pending is abandoned so
            # `planner_select_hypothesis` has nothing left to choose.
            kept = update.get("hypotheses", hypotheses)
            update["hypotheses"] = [
                h.model_copy(update={"status": "abandoned"}) if h.status == "pending" else h
                for h in kept
            ]
            update["freeze_requested"] = True
        return update

    return hc_champion_node


def _cohort_deltas(
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
    primary: str,
) -> dict[str, dict[str, float]]:
    """Candidate minus incumbent, per cohort, for the metrics both report.

    The guardrail is "no cohort regresses more than X % relative", which needs
    the incumbent's number for the same cohort. Without the baseline (a
    campaign that predates this field) the deltas are empty and the payload's
    ``cohort_metrics`` still carries the candidate's absolute values.
    """
    out: dict[str, dict[str, float]] = {}
    for cohort, metrics in candidate.items():
        base = baseline.get(cohort) or {}
        deltas: dict[str, float] = {}
        for name, value in metrics.items():
            if name in base and isinstance(value, (int, float)) and isinstance(base[name], (int, float)):
                deltas[name] = float(value) - float(base[name])
                if base[name] and name == primary:
                    deltas[f"{name}_rel"] = (float(value) - float(base[name])) / float(base[name])
        if "n_users" in metrics:
            deltas["n_users"] = float(metrics["n_users"])
        if deltas:
            out[cohort] = deltas
    return out


def _comparison(state: CampaignState, champions: list, contract) -> list[dict]:
    """The baseline, every prior champion, and this one, on the same metrics.

    A champion's number means nothing alone: the question is whether it beats
    the incumbent and whether it beats the hypothesis before it.
    """
    primary = contract.primary_metric if contract else ""
    rows: list[dict] = []
    baseline = state.get("baseline_metrics") or {}
    if baseline:
        rows.append({"label": "baseline", "metrics": dict(baseline)})
    for champion in champions:
        rows.append(
            {
                "label": champion.hypothesis_id,
                "metrics": {
                    primary: champion.primary_metric_value,
                    **{
                        name: value
                        for name, value in (champion.secondary_metrics or {}).items()
                        if name != primary
                    },
                },
            }
        )
    return rows


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
