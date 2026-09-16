"""Stage 3 (hypothesis selection / planning / feature-fit HC) and Stage 7b (champion)."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.types import interrupt

from amt.graph.deps import NodeDeps
from amt.graph.human import apply_checklist, parse_resume
from amt.graph.routing import ContractViolation, assert_contract_hash
from amt.graph.state import CampaignState, Stage
from amt.schemas.common import ExperimentBudget
from amt.schemas.feature_recipe import (
    ApprovedShape,
    is_tabular_family,
    runtime_vocabulary,
    unsupported_requirements,
)
from amt.schemas.hypothesis import Hypothesis


def make_planner_hypotheses_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Write the campaign's whole hypothesis backlog, for a human to read at once."""

    def planner_hypotheses_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        spec = state.get("problem_spec")
        assert_contract_hash(state, contract)
        baseline = state.get("last_result")
        backlog = deps.planner_agent.propose_hypotheses(
            spec,
            contract,
            history=list(state.get("experiment_history") or []),
            existing=list(state.get("hypotheses") or []),
            human_feedback=state.get("human_feedback"),
            baseline_metrics=dict(getattr(baseline, "validation_metrics", {}) or {}),
        )
        hypotheses = [_mark_feasibility(h) for h in backlog.hypotheses]
        return {
            "stage": Stage.HYPOTHESES.value,
            "status": "running",
            "hypotheses": sorted(hypotheses, key=lambda h: (h.priority, -h.confidence)),
            "backlog_rationale": backlog.prioritization_rationale,
            "hypotheses_approved": False,
            "human_feedback": None,
        }

    return planner_hypotheses_node


def _mark_feasibility(hypothesis: Hypothesis) -> Hypothesis:
    """Strike out a hypothesis this runtime cannot build, with the reason.

    ``infeasible`` is not in ``pending``, so selection skips it; the human still
    sees it at HC_BACKLOG, struck through, which is more useful than a backlog
    that quietly omits the idea.
    """
    unsupported = unsupported_requirements(hypothesis.requires)
    if hypothesis.family and hypothesis.family not in runtime_vocabulary()["families"]:
        unsupported = sorted({*unsupported, hypothesis.family})
    if not unsupported:
        return hypothesis
    return hypothesis.model_copy(
        update={
            "status": "infeasible",
            "infeasible_reason": (
                f"this runtime cannot provide {unsupported}; "
                "no transform, candidate strategy or family expresses it"
            ),
        }
    )


def make_hc_hypotheses_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """HC_BACKLOG — a human approves *what will be tried* before any of it is planned.

    The SPEC leaves the space between HC3 and HC4 autonomous, so this gate is not
    one of its four; it exists because the backlog decides how the whole compute
    budget is spent, and a human reading all of it at once can strike what this
    harness cannot honestly test.
    """

    def hc_hypotheses_node(state: CampaignState) -> dict:
        hypotheses = list(state.get("hypotheses") or [])
        payload = interrupt(
            {
                "checkpoint": "HC_BACKLOG",
                "title": "Hypothesis backlog approval",
                "campaign_id": state.get("campaign_id"),
                "hypotheses": [h.model_dump(mode="json") for h in hypotheses],
                "prioritization_rationale": state.get("backlog_rationale") or "",
                "baseline_run_id": state.get("baseline_run_id"),
                "instructions": (
                    "amt approve <campaign> [--keep H1,H3] [--drop H2] "
                    "| amt reject <campaign> -m '<feedback>'"
                ),
            }
        )
        decision = parse_resume(payload)
        update: dict = {
            "hypotheses_approved": decision.approved,
            "human_feedback": decision.feedback,
            "awaiting": None,
            "status": "running",
        }
        if decision.approved:
            update["hypotheses"] = apply_checklist(hypotheses, decision)
        return update

    return hc_hypotheses_node


def make_planner_select_hypothesis_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Hand the Trainer one approved hypothesis at a time, in priority order."""

    def planner_select_hypothesis_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        spec = state.get("problem_spec")
        assert_contract_hash(state, contract)

        hypotheses = list(state.get("hypotheses") or [])
        history = list(state.get("experiment_history") or [])
        budget: ExperimentBudget = state.get("budget") or ExperimentBudget()

        if not hypotheses:
            raise ContractViolation(
                "No hypothesis backlog to select from; the backlog checkpoint never ran."
            )

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
        shape = _selected_shape(state, chosen)
        plan = deps.planner_agent.build_plan(
            hypothesis=chosen,
            spec=spec,
            contract=contract,
            baseline_reference=state.get("baseline_run_id") or "",
            history=history,
            shape=shape,
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
        update["current_config"] = None
        # The shape a human approved at HC_FEATURE, copied into the keys the
        # Trainer and the executor read. Non-tabular hypotheses clear them, so a
        # booster's recipe cannot leak into the SASRec run that follows it.
        update["approved_feature_recipe"] = shape.feature_recipe if shape else None
        update["approved_fit_spec"] = shape.fit_spec if shape else None
        update["feature_fit_hash"] = shape.feature_fit_hash if shape else ""
        update["feature_fit_approved"] = shape is not None
        return update

    return planner_select_hypothesis_node


def _selected_shape(state: CampaignState, chosen: Hypothesis) -> ApprovedShape | None:
    """The approved shape for ``chosen``, or ``None`` for a fixed-shape family."""
    shape = (state.get("approved_shapes") or {}).get(chosen.hypothesis_id)
    if not is_tabular_family(chosen.family):
        return None
    if shape is None:
        raise ContractViolation(
            f"hypothesis {chosen.hypothesis_id!r} targets the tabular family "
            f"{chosen.family!r} but carries no approved feature/fit shape. "
            "HC_FEATURE approves every tabular shape in the backlog before "
            "selection, so this is a routing bug, not a missing approval."
        )
    return shape


def tabular_hypotheses(state: CampaignState) -> list[Hypothesis]:
    """The pending hypotheses whose family needs a feature/fit shape."""
    return [
        h
        for h in state.get("hypotheses") or []
        if h.status == "pending" and is_tabular_family(h.family)
    ]


def make_planner_feature_fit_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Declare the input shape of every tabular hypothesis in the approved backlog.

    One call per hypothesis, all before the first is planned, so the human sees
    every shape the campaign will spend compute on in a single sitting. The old
    per-hypothesis gate charged one review round per hypothesis and re-opened on
    Trainer drift; eight of those rounds is what this replaces.
    """

    def planner_feature_fit_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        assert_contract_hash(state, contract)
        pending = tabular_hypotheses(state)
        if not pending:
            raise ContractViolation(
                "No pending tabular hypothesis to propose a shape for; "
                "route_after_hc_hypotheses should have skipped HC_FEATURE."
            )
        previous = dict(state.get("feature_fit_proposals") or {})
        spec = state.get("problem_spec")
        feedback = state.get("human_feedback")

        proposals = {}
        for hypothesis in pending:
            proposal = deps.planner_agent.propose_feature_fit(
                hypothesis=hypothesis,
                spec=spec,
                contract=contract,
                family=hypothesis.family,
                previous=previous.get(hypothesis.hypothesis_id),
                human_feedback=feedback,
            )
            _reject_unreachable_sources(deps, proposal)
            proposals[hypothesis.hypothesis_id] = proposal
        return {
            "stage": Stage.FEATURE_FIT.value,
            "status": "running",
            "feature_fit_proposals": proposals,
            "shapes_approved": False,
            "human_feedback": None,
        }

    return planner_feature_fit_node


def _reject_unreachable_sources(deps: NodeDeps, proposal) -> None:
    """Fail a recipe naming a table that does not exist, here and not on a cluster.

    A bad table name otherwise survives approval, a wheel build and a job
    submission before Spark reports it. Skipped under ``--mock``, which has no
    warehouse to ask.
    """
    probe = getattr(deps, "sql_probe", None)
    if probe is None:
        return
    for source in proposal.feature_recipe.sources:
        if not source.table:
            continue
        try:
            probe.describe_table(source.table)
        except Exception as exc:
            raise ContractViolation(
                f"recipe source {source.name!r} names {source.table!r}, which this "
                f"identity cannot describe: {exc}"
            ) from exc


def make_hc_feature_fit_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """HC_FEATURE — one review of every shape the campaign will train.

    Dropping a shape here abandons its hypothesis, which is the decision the
    previous operator had to smuggle through ``FeatureRecipe.notes``.
    """

    def hc_feature_fit_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        proposals = dict(state.get("feature_fit_proposals") or {})
        hypotheses = list(state.get("hypotheses") or [])
        titles = {h.hypothesis_id: h.title for h in hypotheses}
        payload = interrupt(
            {
                "checkpoint": "HC_FEATURE",
                "title": f"Feature/fit approval — {len(proposals)} shape(s)",
                "campaign_id": state.get("campaign_id"),
                "hypothesis_ids": sorted(proposals),
                "shapes": {
                    hypothesis_id: {
                        "title": titles.get(hypothesis_id, ""),
                        "feature_fit_hash": proposal.feature_fit_hash,
                        **_ordered_shape(proposal),
                    }
                    for hypothesis_id, proposal in sorted(proposals.items())
                },
                "duplicate_shapes": _duplicate_shapes(
                    proposals, list(state.get("experiment_history") or [])
                ),
                "validation_rows": getattr(contract.validation_split, "row_count", 0)
                if contract
                else 0,
                "instructions": (
                    "amt approve <campaign> [--keep H1,H3] [--drop H2] "
                    "| amt reject <campaign> -m '<feedback>'"
                ),
            }
        )
        decision = parse_resume(payload)
        update: dict = {
            "shapes_approved": decision.approved,
            "human_feedback": decision.feedback,
            "awaiting": None,
            "status": "running",
        }
        if not decision.approved:
            return update

        kept = apply_checklist(hypotheses, decision, scope=proposals)
        update["hypotheses"] = kept
        alive = {h.hypothesis_id for h in kept if h.status == "pending"}
        update["approved_shapes"] = {
            hypothesis_id: ApprovedShape(
                hypothesis_id=hypothesis_id,
                family=proposal.family,
                feature_recipe=proposal.feature_recipe,
                fit_spec=proposal.fit_spec,
                approved_at_hash=state.get("contract_hash") or "",
            )
            for hypothesis_id, proposal in proposals.items()
            if hypothesis_id in alive
        }
        return update

    return hc_feature_fit_node


#: `candidate_generation` and `fit_spec` decide the cost and the objective of a
#: run, and were below the fold of an 8 000-character preview. They lead now.
_SHAPE_KEY_ORDER = ("candidate_generation", "fit_spec", "label", "feature_columns")


def _ordered_shape(proposal) -> dict:
    """The proposal as JSON, with the fields a reviewer decides on first."""
    document = proposal.model_dump(mode="json")
    recipe = document.get("feature_recipe") or {}
    lead = {key: recipe[key] for key in _SHAPE_KEY_ORDER if key in recipe}
    lead |= {key: value for key, value in recipe.items() if key not in lead}
    document["feature_recipe"] = lead
    return {
        "fit_spec": document.pop("fit_spec", None),
        "candidate_generation": lead.get("candidate_generation"),
        **document,
    }


def _duplicate_shapes(proposals: dict, history: list | None = None) -> dict[str, list[str]]:
    """Hypotheses that reduce to a shape already proposed or already run.

    Two hypotheses with one ``feature_fit_hash`` are one experiment billed
    twice; a backlog reads as five ideas and spends like three. Runs already in
    ``experiment_history`` count too — re-proposing a shape the campaign has
    measured is the most expensive kind of duplicate.
    """
    by_hash: dict[str, list[str]] = {}
    for hypothesis_id, proposal in sorted(proposals.items()):
        by_hash.setdefault(proposal.feature_fit_hash, []).append(hypothesis_id)

    already_run: dict[str, set[str]] = {}
    for summary in history or []:
        if summary.feature_fit_hash:
            already_run.setdefault(summary.feature_fit_hash, set()).add(
                summary.hypothesis_id or summary.run_id
            )

    duplicates: dict[str, list[str]] = {}
    for shape_hash, ids in by_hash.items():
        run_before = sorted(already_run.get(shape_hash, ()))
        for hypothesis_id in ids:
            others = [other for other in ids if other != hypothesis_id]
            marks = others + [f"already run as {ran}" for ran in run_before]
            if marks:
                duplicates[hypothesis_id] = marks
    return duplicates


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
