"""Stage 11/12 — scientific conclusion, final MLflow run, HC4."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path

from langgraph.types import interrupt

from amt.graph.deps import NodeDeps
from amt.graph.human import gate_update, parse_resume
from amt.graph.state import CampaignState, Stage
from amt.schemas.conclusion import (
    GROUPED_RERANK_ASSUMPTIONS,
    AbDesign,
    CohortRow,
    ExperimentationReport,
    ProductUnits,
    ProxySanity,
)
from amt.schemas.qa import QADecision


def make_conclusion_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Write the report. A negative result is a valid outcome."""

    def conclusion_node(state: CampaignState) -> dict:
        contract = state.get("evaluation_contract")
        decision = state.get("qa_decision") or QADecision.INVALID_EXPERIMENT
        # Built first and handed to the agent as context: the Planner writes
        # the narrative, never the numbers.
        report = build_report(state)
        champions = list(state.get("hypothesis_champions") or [])
        verdict = validation_verdict(
            state.get("baseline_metrics") or {},
            champions,
            contract.primary_metric if contract else "",
        )
        conclusion = deps.planner_agent.write_conclusion(
            campaign_id=state.get("campaign_id", ""),
            spec=state.get("problem_spec"),
            contract=contract,
            history=list(state.get("experiment_history") or []),
            champions=champions,
            assessment=state.get("qa_assessment"),
            phase0=state.get("qa_phase0"),
            phase1=state.get("qa_phase1"),
            report=report,
            baseline_metrics=state.get("baseline_metrics") or {},
            validation_verdict=verdict,
        )
        update: dict = {
            "campaign_id": state.get("campaign_id", ""),
            "qa_decision": decision,
            "contract_hash": state.get("contract_hash", ""),
            "mlflow_experiment_id": state.get("mlflow_experiment_id", ""),
            "negative_result": decision != QADecision.PASS,
        }
        # The recommendation is clamped to the numbers, not left to the prose.
        # A champion that lost to the incumbent on validation is a rejected
        # hypothesis whatever QA did or did not manage to run; the narrative
        # may explain, it may not upgrade.
        if verdict and not verdict["any_beats_baseline"]:
            if decision != QADecision.SPEC_REVIEW:
                update["recommendation"] = "hypothesis_rejected"
            update["negative_result"] = True
        conclusion = conclusion.model_copy(update=update)
        status = state.get("status")
        if decision == QADecision.SPEC_REVIEW:
            status = "spec_review"
        return {
            "stage": Stage.CONCLUSION.value,
            "conclusion": conclusion,
            "experimentation_report": report,
            "status": status or "running",
        }

    return conclusion_node


def validation_verdict(
    baseline: dict[str, float], champions: list, primary: str
) -> dict | None:
    """Candidate-minus-incumbent on validation, per champion, as plain numbers.

    Computed here so the conclusion cannot omit it, misread it, or claim the
    incumbent's numbers "were not produced" -- which is what the first LFY
    conclusion did while the incumbent sat 4.8 % above the champion.
    """
    if not primary or not champions:
        return None
    base = baseline.get(primary)
    rows = []
    for champion in champions:
        value = champion.primary_metric_value
        row = {"hypothesis_id": champion.hypothesis_id, primary: value}
        if isinstance(base, (int, float)) and isinstance(value, (int, float)):
            delta = float(value) - float(base)
            row.update(
                {
                    f"baseline_{primary}": float(base),
                    "delta": delta,
                    "relative_delta": delta / float(base) if base else None,
                    "beats_baseline": delta > 0,
                }
            )
        rows.append(row)
    return {
        "primary_metric": primary,
        "champions": rows,
        "any_beats_baseline": any(r.get("beats_baseline") for r in rows),
        "note": (
            "Point deltas on the validation split. Paired confidence intervals "
            "come from QA phase 1 only; if QA did not run, none exist."
        ),
    }


def build_report(state: CampaignState) -> ExperimentationReport:
    """Assemble the decision-ready evidence from what QA already computed.

    Nothing here is derived from an agent's opinion; every number comes out of
    ``Phase1Result``. A ``FAIL_HYPOTHESIS`` or ``SPEC_REVIEW`` campaign gets the
    same report as a passing one -- a negative result on the incumbent's own
    ground is still the first honest read of the rail, and it is only worth
    anything if it is presented the same way.
    """
    phase1 = state.get("qa_phase1")
    contract = state.get("evaluation_contract")
    materialization = state.get("split_materialization")
    mode = getattr(contract, "evaluation_mode", "next_item")
    primary = getattr(contract, "primary_metric", "") or (
        getattr(phase1, "primary_metric", "") if phase1 else ""
    )
    if phase1 is None:
        return ExperimentationReport(
            campaign_id=state.get("campaign_id", ""),
            primary_metric=primary,
            evaluation_mode=mode,
            contract_hash=state.get("contract_hash", ""),
        )

    delta = phase1.primary_delta
    guardrail = getattr(
        getattr(contract, "acceptance_criteria", None), "max_cohort_regression", 0.02
    )
    cohorts = _cohort_rows(phase1, primary, float(guardrail))
    units = phase1.product_units or {}

    return ExperimentationReport(
        campaign_id=state.get("campaign_id", ""),
        primary_metric=primary,
        primary_delta=delta.mean_delta if delta else 0.0,
        primary_ci=(delta.ci_low, delta.ci_high) if delta else (0.0, 0.0),
        primary_significant=bool(delta and delta.significant),
        candidate_metrics=dict(phase1.qa_metrics),
        baseline_metrics=dict(phase1.baseline_metrics),
        secondary_deltas={
            d.metric: d.mean_delta for d in phase1.paired_deltas if d.metric != primary
        },
        product_units=_product_units(units),
        cohort_table=cohorts,
        regressions=[row.name for row in cohorts if row.regressed],
        proxy_sanity=ProxySanity.model_validate(
            {k: v for k, v in (phase1.proxy_sanity or {}).items() if k in _PROXY_FIELDS}
        ),
        propensities=dict(getattr(materialization, "propensities", {}) or {}),
        ab_design=AbDesign.model_validate(
            {k: v for k, v in (phase1.ab_design or {}).items() if k in _DESIGN_FIELDS}
        ),
        operational=dict(phase1.operational),
        assumptions=list(GROUPED_RERANK_ASSUMPTIONS) if mode == "grouped_rerank" else [],
        evaluation_mode=mode,
        relevance_weighting=getattr(contract, "relevance_weighting", "none"),
        qa_decision=state.get("qa_decision"),
        contract_hash=state.get("contract_hash", ""),
    )


_PROXY_FIELDS = set(ProxySanity.model_fields)
_DESIGN_FIELDS = set(AbDesign.model_fields)


def _product_units(units: dict) -> ProductUnits:
    candidate = dict(units.get("candidate") or {})
    baseline = dict(units.get("baseline") or {})
    return ProductUnits(
        unit=str(units.get("unit") or ProductUnits.model_fields["unit"].default),
        candidate=float(candidate.get("per_1k", 0.0)),
        candidate_ci=(
            float(candidate.get("ci_low", 0.0)),
            float(candidate.get("ci_high", 0.0)),
        ),
        baseline=float(baseline.get("per_1k", 0.0)),
        baseline_ci=(
            float(baseline.get("ci_low", 0.0)),
            float(baseline.get("ci_high", 0.0)),
        ),
        n_renders=int(candidate.get("n_renders", 0)),
    )


def _cohort_rows(phase1, primary: str, guardrail: float) -> list[CohortRow]:
    """One row per reported slice, flagged when it falls past the guardrail.

    The guardrail is a *relative* regression, pre-registered in the contract:
    an overall win that costs a cohort more than that is not a win.
    """
    rows: list[CohortRow] = []
    for name, block in sorted((phase1.cohort_metrics or {}).items()):
        candidate = float(block.get(primary, block.get(f"val/{primary}", 0.0)))
        baseline = float(block.get(f"baseline/{primary}", 0.0))
        delta = float(block.get(f"delta/{primary}", candidate - baseline))
        relative = (delta / baseline) if baseline else 0.0
        rows.append(
            CohortRow(
                name=name,
                metric=primary,
                candidate=candidate,
                baseline=baseline,
                delta=delta,
                relative_delta=relative,
                n=int(block.get("n_users", 0) or 0),
                regressed=relative < -abs(guardrail),
            )
        )
    return rows


def make_log_final_mlflow_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Stage 12 — log the final run with full lineage (skipped when MLflow is disabled)."""

    def log_final_mlflow_node(state: CampaignState) -> dict:
        report = state.get("experimentation_report")
        # The report is written whether or not MLflow is on: under `--mock`
        # there is no tracking server, and the HTML is still what a reader
        # wants to see.
        report_path = _write_report(deps, state, report)
        if not deps.use_mlflow:
            return {"report_path": report_path} if report_path else {}
        frozen = state.get("frozen_champion")
        conclusion = state.get("conclusion")
        if frozen is None or conclusion is None:
            return {"report_path": report_path} if report_path else {}
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
            report_html=report_path,
        )
        return {
            "mlflow_final_run_id": run_id,
            "report_path": report_path or "",
            "conclusion": conclusion.model_copy(update={"mlflow_final_run_id": run_id}),
        }

    return log_final_mlflow_node


def _write_report(deps: NodeDeps, state: CampaignState, report) -> str:
    """Render the report beside the checkpoint database and return its path.

    Best effort. An unwritable directory is worth a warning, not the loss of a
    campaign that has already spent its whole compute budget -- and the report
    is reconstructible from state at any time.
    """
    if report is None:
        return ""
    from amt.report import render_report

    try:
        path = Path(deps.settings.reports_dir()) / (
            f"{state.get('campaign_id') or 'campaign'}.html"
        )
        path.write_text(render_report(report, state.get("conclusion")), encoding="utf-8")
        return str(path)
    except OSError as exc:
        warnings.warn(f"could not write the experimentation report: {exc}", stacklevel=2)
        return ""


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
            **gate_update(state, decision),
            "stage": Stage.DONE.value,
            "status": status,
            "promotion_decision": decision.decision
            or ("promote" if decision.approved else "hold"),
        }

    return hc_conclusion_node
