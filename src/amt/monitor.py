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
    "backlog_rows",
    "champion_rows",
    "checkpoint_document",
    "clip",
    "config_for",
    "list_campaigns",
    "masked_settings",
    "pending_interrupt",
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
        estimated = validation_rows * n
        notes = []
        if duplicates.get(hypothesis_id):
            notes.append(f"same shape as {', '.join(duplicates[hypothesis_id])}")
        if not (shape.get("feature_recipe") or {}).get("feature_columns"):
            notes.append("no feature columns")
        rows.append(
            {
                "id": hypothesis_id,
                "family": str(shape.get("family") or ""),
                "candidates": f"{candidates.get('strategy') or '?'} n={n}",
                "est. rows": f"{estimated:,}" if estimated else "unknown",
                "fit": "\n".join(
                    f"{key}={fit.get(key)}" for key in FIT_FIELDS if fit.get(key) not in ("", None)
                ),
                "notes": "\n".join(notes),
            }
        )
    return rows


def champion_rows(payload: dict) -> list[dict[str, str]]:
    """This hypothesis's champion against the baseline and its predecessors."""
    rows = []
    for entry in payload.get("comparison") or []:
        rows.append(
            {
                "model": str(entry.get("label") or ""),
                **{
                    metric: f"{value:.5f}" if isinstance(value, (int, float)) else str(value)
                    for metric, value in (entry.get("metrics") or {}).items()
                },
            }
        )
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
