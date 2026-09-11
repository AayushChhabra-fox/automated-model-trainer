"""Stage 3 (hypothesis selection / planning) and Stage 7b (final champion)."""

from __future__ import annotations

from collections.abc import Callable

from amt.graph.deps import NodeDeps
from amt.graph.routing import ContractViolation, assert_contract_hash
from amt.graph.state import CampaignState, Stage
from amt.schemas.common import ExperimentBudget


def make_planner_select_hypothesis_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Generate the backlog once, then hand the Trainer one hypothesis at a time."""

    def planner_select_hypothesis_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        spec = state.get("problem_spec")
        assert_contract_hash(state, contract)

        hypotheses = list(state.get("hypotheses") or [])
        history = list(state.get("experiment_history") or [])
        budget: ExperimentBudget = state.get("budget") or ExperimentBudget()

        if not hypotheses:
            backlog = deps.planner_agent.propose_hypotheses(spec, contract, history)
            hypotheses = sorted(backlog.hypotheses, key=lambda h: (h.priority, -h.confidence))

        update: dict = {
            "stage": Stage.PLANNING.value,
            "hypotheses": hypotheses,
            "status": "running",
        }

        pending = [h for h in hypotheses if h.status == "pending"]
        out_of_budget = budget.hypotheses_remaining <= 0 or budget.compute_exhausted
        if not pending or out_of_budget:
            # Nothing left worth running: move on to final champion selection.
            update["current_hypothesis_id"] = None
            update["experiment_plan"] = None
            return update

        chosen = pending[0]
        plan = deps.planner_agent.build_plan(
            hypothesis=chosen,
            spec=spec,
            contract=contract,
            baseline_reference=state.get("baseline_run_id") or "",
            history=history,
        )
        update["hypotheses"] = [
            h.model_copy(update={"status": "active"}) if h.hypothesis_id == chosen.hypothesis_id
            else h
            for h in hypotheses
        ]
        update["current_hypothesis_id"] = chosen.hypothesis_id
        update["experiment_plan"] = plan
        update["iteration"] = 0
        update["trainer_converged"] = False
        update["convergence_reason"] = ""
        update["budget"] = budget.record_hypothesis()
        return update

    return planner_select_hypothesis_node


def make_planner_final_champion_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Stage 7b — exactly one candidate goes forward to freeze."""

    def planner_final_champion_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        spec = state.get("problem_spec")
        assert_contract_hash(state, contract)
        champions = list(state.get("hypothesis_champions") or [])
        if not champions:
            raise ContractViolation(
                "No hypothesis champion was produced; there is nothing to freeze."
            )
        if len(champions) == 1:
            selected = champions[0].hypothesis_id
            rationale = "Only one hypothesis champion was produced."
        else:
            selection = deps.planner_agent.select_final_champion(champions, spec, contract)
            selected = selection.hypothesis_id
            rationale = selection.rationale
            if selected not in {c.hypothesis_id for c in champions}:
                raise ContractViolation(
                    f"Planner selected an unknown hypothesis champion: {selected!r}"
                )
        return {
            "stage": Stage.FINAL_CHAMPION.value,
            "current_hypothesis_id": selected,
            "final_selection_rationale": rationale,
        }

    return planner_final_champion_node
