"""Stage 3 (hypothesis selection / planning / feature-fit HC) and Stage 7b (champion)."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.types import interrupt

from amt.graph.deps import NodeDeps
from amt.graph.human import parse_resume
from amt.graph.routing import ContractViolation, assert_contract_hash, target_family
from amt.graph.state import CampaignState, Stage, current_hypothesis
from amt.schemas.common import ExperimentBudget
from amt.schemas.feature_recipe import feature_fit_hash, is_tabular_family


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
        # Approval is per hypothesis: a new hypothesis starts with no approved shape.
        update["current_config"] = None
        update["feature_fit_proposal"] = None
        update["approved_feature_recipe"] = None
        update["approved_fit_spec"] = None
        update["feature_fit_hash"] = ""
        update["feature_fit_approved"] = False
        return update

    return planner_select_hypothesis_node


def make_planner_feature_fit_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Declare the input shape of a tabular model (features, label, candidates, fit)."""

    def planner_feature_fit_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        assert_contract_hash(state, contract)
        plan = state.get("experiment_plan")
        hypothesis = current_hypothesis(state)
        if plan is None or hypothesis is None:
            raise ContractViolation("A feature/fit proposal needs a hypothesis and a plan.")
        family = target_family(state)
        if not is_tabular_family(family):
            raise ContractViolation(
                f"family {family!r} does not take a FeatureRecipe; "
                "only the tabular booster families do."
            )
        proposal = deps.planner_agent.propose_feature_fit(
            hypothesis=hypothesis,
            plan=plan,
            spec=state.get("problem_spec"),
            contract=contract,
            family=family,
            previous=state.get("feature_fit_proposal"),
            human_feedback=state.get("human_feedback"),
        )
        return {
            "stage": Stage.FEATURE_FIT.value,
            "status": "running",
            "feature_fit_proposal": proposal,
            "feature_fit_approved": False,
            "human_feedback": None,
        }

    return planner_feature_fit_node


def make_hc_feature_fit_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """HC_FEATURE — a human approves what the model is allowed to see, once per shape."""

    def hc_feature_fit_node(state: CampaignState) -> dict:
        proposal = state.get("feature_fit_proposal")
        payload = interrupt(
            {
                "checkpoint": "HC_FEATURE",
                "title": "Feature/fit approval",
                "campaign_id": state.get("campaign_id"),
                "hypothesis_id": state.get("current_hypothesis_id"),
                "feature_fit": proposal.model_dump(mode="json") if proposal else None,
                "feature_fit_hash": proposal.feature_fit_hash if proposal else None,
                "instructions": "amt approve <campaign> | amt reject <campaign> -m '<feedback>'",
            }
        )
        decision = parse_resume(payload)
        update: dict = {
            "feature_fit_approved": decision.approved,
            "human_feedback": decision.feedback,
            "awaiting": None,
            "status": "running",
        }
        if not decision.approved or proposal is None:
            return update

        recipe, fit = proposal.feature_recipe, proposal.fit_spec
        approved_hash = feature_fit_hash(recipe, fit)
        update["approved_feature_recipe"] = recipe
        update["approved_fit_spec"] = fit
        update["feature_fit_hash"] = approved_hash
        plan = state.get("experiment_plan")
        if plan is not None:
            update["experiment_plan"] = plan.model_copy(
                update={
                    "family": proposal.family,
                    "feature_recipe": recipe,
                    "fit_spec": fit,
                    "feature_fit_hash": approved_hash,
                }
            )
        return update

    return hc_feature_fit_node


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
