"""Read model over a compiled campaign graph.

Pure functions of ``(graph, campaign_id)``. No Rich, no Typer, no Flask: the
CLI renders these as tables and the web surface renders them as HTML, and
neither is allowed to become the only way to read a campaign.

The write side is not here. It is ``Command(resume=...)`` against the same
graph, and it goes through :func:`amt.graph.human.parse_resume` so both
surfaces build the same ``HumanDecision``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

__all__ = [
    "FIT_FIELDS",
    "PREVIEW_CHARS",
    "REWIND_EFFECTS",
    "backlog_rows",
    "champion_rows",
    "checkpoint_document",
    "checkpoint_prose",
    "clip",
    "conclusion_rows",
    "config_for",
    "list_campaigns",
    "masked_settings",
    "pending_interrupt",
    "report_rows",
    "rewind_targets",
    "shape_rows",
    "state",
    "summary_rows",
]

#: A checkpoint document can run to hundreds of KB once probe rows are included,
#: which no terminal panel can usefully show. The panel is a preview; the full
#: document is what `amt show` and the campaign page print.
PREVIEW_CHARS = 20_000

#: The six FitSpec fields a reviewer has to see. Not the booster params: those
#: are the Trainer's to change without asking.
FIT_FIELDS = (
    "objective",
    "eval_metric",
    "num_boost_round",
    "early_stopping_rounds",
    "negative_samples_per_positive",
    "seed",
)

#: High enough for the batched HC_FEATURE gate plus a full trainer loop.
RECURSION_LIMIT = 150


def config_for(campaign: str) -> dict:
    """The LangGraph config for one campaign (``thread_id == campaign_id``)."""
    return {"configurable": {"thread_id": campaign}, "recursion_limit": RECURSION_LIMIT}


def state(graph: Any, campaign: str) -> dict:
    """The campaign's committed state, or ``{}`` when it does not exist."""
    snapshot = graph.get_state(config_for(campaign))
    return dict(snapshot.values or {})


def pending_interrupt(graph: Any, campaign: str) -> dict | None:
    """The payload of the interrupt the campaign is suspended on, if any."""
    snapshot = graph.get_state(config_for(campaign))
    for task in getattr(snapshot, "tasks", ()) or ():
        for interrupt in getattr(task, "interrupts", ()) or ():
            value = getattr(interrupt, "value", None)
            if isinstance(value, dict):
                return value
    return None


def list_campaigns(db: str | Path) -> list[str]:
    """Thread ids in a SQLite checkpoint database, most recently written first.

    Read directly rather than through the saver: LangGraph has no "list
    threads" API, and the campaign list is the one question the checkpointer
    can answer that the graph cannot.
    """
    path = Path(db).expanduser()
    if not path.exists():
        return []
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT thread_id, max(checkpoint_id) AS latest FROM checkpoints "
            "GROUP BY thread_id ORDER BY latest DESC"
        ).fetchall()
    except sqlite3.DatabaseError:
        # A database that exists but has no checkpoints table yet.
        return []
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def checkpoint_document(payload: dict) -> Any:
    """The part of an interrupt payload a human actually decides on."""
    for key in (
        "problem_spec",
        "evaluation_contract",
        "hypotheses",
        "shapes",
        "champion",
        "conclusion",
    ):
        if payload.get(key):
            return payload[key]
    return payload


def backlog_rows(payload: dict) -> list[dict[str, str]]:
    """One row per hypothesis: what it targets, what it costs, whether it is alive."""
    return [
        {
            "id": str(h.get("hypothesis_id") or ""),
            "title": str(h.get("title") or ""),
            "family": str(h.get("family") or "—"),
            "metric": str(h.get("metric_expected_to_improve") or ""),
            "expected effect": str(h.get("expected_effect_size") or ""),
            "cost": f"{h.get('estimated_cost_minutes') or 0:.0f}m",
            "status": str(h.get("status") or ""),
            "note": str(h.get("infeasible_reason") or ""),
        }
        for h in payload.get("hypotheses") or []
    ]


def shape_rows(payload: dict) -> list[dict[str, str]]:
    """One row per feature/fit shape, priced.

    The candidate-row estimate is the number this gate exists to catch: a
    plausible-looking ``n`` against a 44 000-row validation split is 8.9M rows
    of feature building, and nothing else in the payload says so.
    """
    validation_rows = int(payload.get("validation_rows") or 0)
    duplicates = payload.get("duplicate_shapes") or {}
    rows = []
    for hypothesis_id, shape in sorted((payload.get("shapes") or {}).items()):
        candidates = shape.get("candidate_generation") or {}
        fit = shape.get("fit_spec") or {}
        n = int(candidates.get("n") or 0)
        strategy = str(candidates.get("strategy") or "?")
        if strategy == "logged_impressions":
            # The candidate set is closed at what the container showed -- `n`
            # is never applied, and pricing it as `n × renders` once sent a
            # planner chasing a 49.7M-row budget that did not exist.
            candidates_cell = "logged_impressions (closed set; n not applied)"
            estimated_cell = f"{validation_rows:,} renders × tiles shown" if validation_rows else "unknown"
        else:
            candidates_cell = f"{strategy} n={n}"
            estimated = validation_rows * n
            estimated_cell = f"{estimated:,}" if estimated else "unknown"
        notes = []
        if duplicates.get(hypothesis_id):
            notes.append(f"same shape as {', '.join(duplicates[hypothesis_id])}")
        if not (shape.get("feature_recipe") or {}).get("feature_columns"):
            notes.append("no feature columns")
        rows.append(
            {
                "id": hypothesis_id,
                "family": str(shape.get("family") or ""),
                "candidates": candidates_cell,
                "est. rows": estimated_cell,
                "fit": "\n".join(
                    f"{key}={fit.get(key)}" for key in FIT_FIELDS if fit.get(key) not in ("", None)
                ),
                "notes": "\n".join(notes),
            }
        )
    return rows


#: Metric names that are bookkeeping, not evidence. Everything under
#: `reproduction/` is the phase-0 subset (200 renders); the counts describe the
#: split, not the model.
_NOT_EVIDENCE_PREFIXES = ("reproduction/",)
_NOT_EVIDENCE = frozenset({"n_users", "n_scored", "catalog_size"})


def champion_rows(payload: dict) -> list[dict[str, str]]:
    """This hypothesis's champion against the baseline and its predecessors.

    Metrics down the side, models across — the primary metric first, then the
    rest the contract asked for, then the pointwise diagnostics — and a delta
    column for the newest champion against the baseline. A 27-column table with
    `reproduction/mrr@1` in it is how a −5 % primary metric went unread.
    """
    comparison = list(payload.get("comparison") or [])
    if not comparison:
        return []
    primary = str(payload.get("primary_metric") or "")
    labels = [str(e.get("label") or "") for e in comparison]
    metrics_by_label = {
        str(e.get("label") or ""): dict(e.get("metrics") or {}) for e in comparison
    }
    baseline = metrics_by_label.get("baseline") or {}
    latest = labels[-1] if labels[-1] != "baseline" else ""

    def evidence(name: str) -> bool:
        return name not in _NOT_EVIDENCE and not name.startswith(_NOT_EVIDENCE_PREFIXES)

    seen: list[str] = []
    for label in labels:
        for name in metrics_by_label[label]:
            if evidence(name) and name not in seen:
                seen.append(name)
    ranking = [n for n in seen if "@" in n]
    pointwise = [n for n in seen if "@" not in n]
    ordered = ([primary] if primary in seen else []) + [
        n for n in ranking if n != primary
    ] + pointwise

    rows = []
    for name in ordered:
        row: dict[str, str] = {"metric": name}
        for label in labels:
            value = metrics_by_label[label].get(name)
            row[label] = f"{value:.4f}" if isinstance(value, (int, float)) else "—"
        if latest and isinstance(baseline.get(name), (int, float)):
            cand = metrics_by_label[latest].get(name)
            if isinstance(cand, (int, float)):
                delta = float(cand) - float(baseline[name])
                rel = delta / float(baseline[name]) if baseline[name] else float("nan")
                row[f"Δ {latest} vs baseline"] = f"{delta:+.4f} ({rel:+.1%})"
        rows.append(row)
    return rows


def report_rows(report: Any) -> list[dict[str, str]]:
    """The experimentation report as a few lines, in the order that decides.

    Proxy sanity first: if the offline metric does not track the product
    number, the rest is not evidence. Then the effect in product units, then
    the offline delta, then what an online test would cost.
    """
    if report is None:
        return []
    proxy = report.proxy_sanity
    units = report.product_units
    design = report.ab_design
    rows = [
        {
            "": "proxy sanity",
            "value": "not checked"
            if not proxy.checked
            else f"{'pass' if proxy.passed else 'FAIL'} rho={proxy.rho:.3f}"
            if proxy.rho is not None
            else ("pass" if proxy.passed else "FAIL"),
            "detail": proxy.reason or f"{proxy.days} day(s), {proxy.source}",
        },
        {
            "": units.unit,
            "value": f"{units.delta:+.2f}",
            "detail": f"candidate {units.candidate:.2f} vs incumbent {units.baseline:.2f}",
        },
        {
            "": report.primary_metric,
            "value": f"{report.primary_delta:+.5f}",
            "detail": (
                f"CI {report.primary_ci[0]:+.5f} to {report.primary_ci[1]:+.5f}"
                + (" (excludes zero)" if report.primary_significant else " (includes zero)")
            ),
        },
    ]
    if design.required_renders:
        days = design.days_at_current_traffic
        rows.append(
            {
                "": "online test",
                "value": f"{design.required_renders:,.0f} renders",
                "detail": f"{days:.1f} days at this rail's traffic" if days else "",
            }
        )
    if report.regressions:
        rows.append(
            {
                "": "cohort regressions",
                "value": str(len(report.regressions)),
                "detail": ", ".join(report.regressions),
            }
        )
    return rows


#: The narrative fields of a conclusion, in the order they argue: what was
#: asked, what was found, what it means. Rendered as prose, because they are
#: paragraphs -- as JSON string values they are a wall of `\n` escapes.
_CONCLUSION_PROSE = (
    ("problem", "problem"),
    ("hypothesis", "hypothesis"),
    ("why_selected", "why this model"),
    ("validation_evidence", "validation evidence"),
    ("qa_evidence", "QA evidence"),
    ("cohort_effects", "cohort effects"),
    ("operational_considerations", "operational considerations"),
    ("recommendation_rationale", "rationale"),
)


def conclusion_rows(payload: dict) -> list[dict[str, str]]:
    """HC4's decision as a few lines, ahead of the argument that supports it.

    The conclusion is the largest document in a campaign -- `report_markdown`
    alone is a page of prose -- and at HC4 a human is answering one question:
    promote or not. That answer is these five lines; everything else is why.
    """
    conclusion = payload.get("conclusion") or {}
    if not conclusion:
        return []
    interpretation = conclusion.get("product_interpretation") or {}
    rows = [
        {
            "": "recommendation",
            "value": str(conclusion.get("recommendation") or "—"),
            "detail": "negative result" if conclusion.get("negative_result") else "",
        },
        {
            "": "QA decision",
            "value": str(payload.get("qa_decision") or conclusion.get("qa_decision") or "—"),
            "detail": "",
        },
        {
            "": "best model",
            "value": str(conclusion.get("best_model") or "—"),
            "detail": "",
        },
        {
            "": "proven causal lift",
            "value": str(interpretation.get("proven_causal_lift") or "—"),
            "detail": "",
        },
    ]
    if conclusion.get("open_questions"):
        rows.append(
            {
                "": "open questions",
                "value": str(len(conclusion["open_questions"])),
                "detail": "; ".join(str(q) for q in conclusion["open_questions"]),
            }
        )
    return rows


def checkpoint_prose(payload: dict) -> list[tuple[str, str]]:
    """``(heading, text)`` for the parts of a checkpoint that are paragraphs.

    Only the conclusion has any today. Splitting it out is what lets the page
    show a reviewable argument instead of a JSON blob with the argument
    escaped inside it.
    """
    conclusion = (payload or {}).get("conclusion") or {}
    if not conclusion:
        return []
    sections = [
        (heading, str(conclusion[key]))
        for key, heading in _CONCLUSION_PROSE
        if conclusion.get(key)
    ]
    if conclusion.get("experiments_summary"):
        sections.append(
            (
                "experiments carrying evidence",
                "\n\n".join(str(e) for e in conclusion["experiments_summary"]),
            )
        )
    if conclusion.get("report_markdown"):
        sections.append(("the report", str(conclusion["report_markdown"])))
    return sections


#: What each rewind target re-enters and what it costs, for an operator
#: choosing between them. Legality is *not* here -- see `rewind_targets`.
REWIND_EFFECTS: dict[str, str] = {
    "backlog": (
        "Re-propose the hypotheses from the baseline. Returns the budget slots "
        "of any mid-flight hypothesis; abandoned ones stay abandoned."
    ),
    "feature": (
        "Re-propose the feature/fit shapes against the approved backlog. For an "
        "approved recipe that cannot execute — columns the split does not carry."
    ),
    "qa": (
        "Re-run QA on the same frozen champion after an evaluation-plumbing "
        "fix. The model is untouched; only QA's own results are cleared."
    ),
}


def rewind_targets(campaign_state: dict) -> list[dict[str, Any]]:
    """Every rewind target, whether it is legal now, and why not when it is not.

    Legality is decided by *calling* ``rewind_update`` and catching its
    refusal, so what the surface offers can never drift from what the action
    permits. A rewind form that offers an illegal target is how an operator
    learns to distrust the whole page.
    """
    from amt.graph.human import rewind_update

    rows: list[dict[str, Any]] = []
    for target, effect in REWIND_EFFECTS.items():
        try:
            rewind_update(campaign_state, target, "probe")
        except ValueError as refusal:
            rows.append(
                {"target": target, "allowed": False, "effect": effect, "reason": str(refusal)}
            )
        else:
            rows.append({"target": target, "allowed": True, "effect": effect, "reason": ""})
    return rows


def summary_rows(payload: dict) -> list[dict[str, str]] | None:
    """Checklist rows for the checkpoint in ``payload``, or ``None``.

    The rendering surface decides what to do with them; this decides which
    question the checkpoint is asking.
    """
    checkpoint = str(payload.get("checkpoint") or "")
    if checkpoint == "HC_BACKLOG":
        return backlog_rows(payload)
    if checkpoint == "HC_FEATURE":
        return shape_rows(payload)
    if checkpoint == "HC_CHAMPION":
        return champion_rows(payload)
    if checkpoint == "HC4":
        return conclusion_rows(payload)
    return None


def clip(text: str, limit: int = PREVIEW_CHARS) -> tuple[str, bool]:
    """``(text, was_truncated)`` for a checkpoint preview."""
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def masked_settings(settings: Any) -> list[dict[str, str]]:
    """Every setting with its value, its source, and secrets hidden.

    The source matters more than it looks: an ``AMT_*`` variable exported in
    the shell shadows ``.env`` silently, and a campaign then runs against a
    catalog nobody chose.
    """
    explicit = set(getattr(settings, "model_fields_set", ()) or ())
    rows = []
    for name, value in sorted(settings.model_dump().items()):
        rows.append(
            {
                "setting": name,
                "value": _mask(name, value),
                "source": "env / .env" if name in explicit else "default",
            }
        )
    return rows


def _mask(name: str, value: Any) -> str:
    lowered = name.lower()
    if value in (None, ""):
        return ""
    if any(marker in lowered for marker in ("secret", "key", "token", "password")):
        text = str(value)
        return f"{text[:4]}…{text[-4:]} ({len(text)} chars)" if len(text) > 12 else "set"
    return str(value)
