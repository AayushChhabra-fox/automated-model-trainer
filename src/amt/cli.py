"""``amt`` — Typer CLI driving the campaign graph across human checkpoints (PLAN §8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from amt.graph.build import build_graph, sqlite_checkpointer
from amt.graph.state import describe, new_campaign_state
from amt.schemas.common import ExperimentBudget
from amt.schemas.qa import QADecision
from amt.settings import get_settings

app = typer.Typer(
    name="amt",
    help="Automated Model Trainer & QA Agent — autonomous ML experimentation on Databricks.",
    no_args_is_help=True,
    add_completion=False,
)
demo_app = typer.Typer(help="Synthetic demo data helpers (Databricks jobs package).")
app.add_typer(demo_app, name="demo")

console = Console()


# --- wiring -----------------------------------------------------------------------
def _build(mock: bool, campaign_id: str, max_iterations: int | None = None) -> Any:
    """Compile a graph with either the mock stack or the real Databricks stack."""
    settings = get_settings()
    checkpointer = sqlite_checkpointer(settings=settings)
    if mock:
        from amt.testing import mock_stack

        executor, llm_factory, _ = mock_stack()
        return build_graph(
            executor=executor,
            llm_factory=llm_factory,
            settings=settings,
            checkpointer=checkpointer,
            use_mlflow=False,
        )

    from amt.databricks.sql import ReadOnlySQL  # real probe for Stage 1
    from amt.executor.databricks import DatabricksExecutor  # provided by the jobs layer

    return build_graph(
        executor=DatabricksExecutor(settings=settings, campaign_id=campaign_id),
        settings=settings,
        sql_probe=ReadOnlySQL(settings=settings),
        checkpointer=checkpointer,
        use_mlflow=True,
    )


def _config(campaign: str) -> dict:
    return {"configurable": {"thread_id": campaign}, "recursion_limit": 100}


def _pending_interrupt(graph: Any, campaign: str) -> dict | None:
    """Return the payload of the interrupt the campaign is currently suspended on."""
    snapshot = graph.get_state(_config(campaign))
    for task in getattr(snapshot, "tasks", ()) or ():
        for interrupt in getattr(task, "interrupts", ()) or ():
            value = getattr(interrupt, "value", None)
            if isinstance(value, dict):
                return value
    return None


def _state(graph: Any, campaign: str) -> dict:
    snapshot = graph.get_state(_config(campaign))
    return dict(snapshot.values or {})


def _run(graph: Any, payload: Any, campaign: str) -> None:
    """Drive the graph, printing each node as it completes.

    ``stream`` and ``invoke`` reach the same state; the difference is that a
    campaign spending twenty minutes in a Databricks job says which stage it is
    in rather than nothing at all. The executor prints the run's page URL.
    """
    for update in graph.stream(payload, _config(campaign), stream_mode="updates"):
        for node, delta in (update or {}).items():
            # `__interrupt__` arrives here too, with a tuple rather than a
            # state delta; `_report_position` renders that from the snapshot.
            if not isinstance(delta, dict):
                continue
            stage = str(delta.get("stage") or "")
            suffix = f"  [dim]{stage}[/dim]" if stage and stage != node else ""
            console.print(f"[dim]▸[/dim] {node}{suffix}")


def _resume(campaign: str, payload: dict, mock: bool) -> None:
    from langgraph.types import Command

    graph = _build(mock, campaign)
    if not _state(graph, campaign):
        console.print(f"[red]No campaign named {campaign!r}.[/red] Run `amt start` first.")
        raise typer.Exit(code=1)
    _run(graph, Command(resume=payload), campaign)
    _report_position(graph, campaign)


def _report_position(graph: Any, campaign: str) -> None:
    """Print whatever the campaign is now waiting on (or its terminal state)."""
    payload = _pending_interrupt(graph, campaign)
    state = _state(graph, campaign)
    if payload:
        body, truncated = _render_checkpoint(payload)
        overview = summarize(payload)
        if overview is not None:
            console.print(overview)
        console.print(
            Panel(
                body,
                title=f"{payload.get('checkpoint')} — {payload.get('title')}",
                border_style="yellow",
            )
        )
        if truncated:
            console.print(
                f"[dim]preview truncated; `amt show {campaign}` prints the whole document.[/dim]"
            )
        console.print(
            "[yellow]awaiting approval[/yellow] — " + str(payload.get("instructions", ""))
        )
        return
    status = state.get("status", "unknown")
    decision = state.get("qa_decision")
    decision = decision.value if isinstance(decision, QADecision) else decision
    console.print(f"[green]campaign {campaign}[/green] status={status} qa_decision={decision}")


#: A checkpoint document can run to hundreds of KB once probe rows are included, which
#: no terminal panel can usefully show. The panel is a preview; `amt show` is the review.
#: 8 000 cut off the fit spec and the candidate budget on a real backlog.
PREVIEW_CHARS = 20_000


def _clip(text: str) -> tuple[str, bool]:
    """``(text, was_truncated)`` for the checkpoint preview."""
    if len(text) <= PREVIEW_CHARS:
        return text, False
    return text[:PREVIEW_CHARS], True


def checkpoint_document(payload: dict) -> Any:
    """The part of an interrupt payload a human actually decides on."""
    for key in (
        "problem_spec",
        "evaluation_contract",
        "hypotheses",
        "shapes",
        "conclusion",
    ):
        if payload.get(key):
            return payload[key]
    return payload


def _render_checkpoint(payload: dict) -> tuple[str, bool]:
    return _clip(json.dumps(checkpoint_document(payload), indent=2, default=str))


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


def summarize(payload: dict) -> Any:
    """A few lines answering what a reviewer asks before reading the JSON.

    The JSON below it is the document of record; this is the part that decides
    whether to read it. ``None`` when a checkpoint has nothing worth
    summarizing separately.
    """
    checkpoint = str(payload.get("checkpoint") or "")
    if checkpoint == "HC_BACKLOG":
        rows = backlog_rows(payload)
        return _table("hypothesis backlog", rows, drop=("title",))
    if checkpoint == "HC_FEATURE":
        return _table("feature/fit shapes", shape_rows(payload))
    return None


def _table(title: str, rows: list[dict[str, str]], drop: tuple[str, ...] = ()) -> Table:
    """Render row dicts as a Rich table, one column per key."""
    table = Table(title=title, header_style="bold", show_lines=True)
    columns = [key for key in (rows[0] if rows else {}) if key not in drop]
    for column in columns:
        table.add_column(column, overflow="fold")
    for row in rows:
        values = [row.get(column, "") for column in columns]
        if row.get("status") not in (None, "", "pending"):
            values = [f"[red]{value}[/red]" for value in values]
        table.add_row(*values)
    return table


# --- commands ----------------------------------------------------------------------
@app.command()
def start(
    problem: str = typer.Argument(..., help="The product problem, in plain language (HC1)."),
    campaign: str = typer.Option(..., "--campaign", "-c", help="Campaign id (== thread id)."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run: mock executor + fake LLM."),
    max_iterations: int = typer.Option(6, help="Trainer iterations allowed per hypothesis."),
    max_hypotheses: int = typer.Option(2, help="Hypotheses allowed in the campaign."),
    max_compute_minutes: float = typer.Option(240.0, help="Compute budget for the campaign."),
) -> None:
    """Start a campaign and run to the first human checkpoint (HC2)."""
    graph = _build(mock, campaign)
    if _state(graph, campaign):
        console.print(f"[red]Campaign {campaign!r} already exists.[/red] Use `amt status`.")
        raise typer.Exit(code=1)
    state = new_campaign_state(
        problem_text=problem,
        campaign_id=campaign,
        budget=ExperimentBudget(
            max_iterations=max_iterations,
            max_hypotheses=max_hypotheses,
            max_compute_minutes=max_compute_minutes,
        ),
    )
    _run(graph, state, campaign)
    _report_position(graph, campaign)


@app.command()
def approve(
    campaign: str = typer.Argument(..., help="Campaign id."),
    keep: str = typer.Option(
        "", "--keep", help="Comma-separated ids to keep; anything else under review is dropped."
    ),
    drop: str = typer.Option("", "--drop", help="Comma-separated ids to abandon."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
) -> None:
    """Approve the current human checkpoint and continue.

    At HC_BACKLOG and HC_FEATURE this is a checklist: `--keep` also sets the
    order the kept items run in, and `--drop` abandons an item without paying
    for the agent to rewrite the whole list.
    """
    _resume(campaign, {"approved": True, "keep": keep, "drop": drop}, mock)


@app.command()
def reject(
    campaign: str = typer.Argument(..., help="Campaign id."),
    message: str = typer.Option(..., "--message", "-m", help="Feedback for the agent."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
) -> None:
    """Reject the current checkpoint with feedback; the stage is re-run."""
    _resume(campaign, {"approved": False, "feedback": message}, mock)


@app.command()
def resume(
    campaign: str = typer.Argument(..., help="Campaign id."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
) -> None:
    """Continue a campaign that stopped without a pending decision (e.g. after a crash)."""
    graph = _build(mock, campaign)
    if not _state(graph, campaign):
        console.print(f"[red]No campaign named {campaign!r}.[/red]")
        raise typer.Exit(code=1)
    if _pending_interrupt(graph, campaign):
        console.print("[yellow]Campaign is awaiting a human decision.[/yellow]")
        _report_position(graph, campaign)
        raise typer.Exit(code=0)
    _run(graph, None, campaign)
    _report_position(graph, campaign)


@app.command()
def status(
    campaign: str = typer.Argument(..., help="Campaign id."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
) -> None:
    """Show the compact campaign state."""
    graph = _build(mock, campaign)
    state = _state(graph, campaign)
    if not state:
        console.print(f"[red]No campaign named {campaign!r}.[/red]")
        raise typer.Exit(code=1)
    table = Table(title=f"campaign {campaign}", show_header=False)
    for key, value in describe(state).items():
        table.add_row(str(key), str(value))
    console.print(table)
    payload = _pending_interrupt(graph, campaign)
    if payload:
        console.print(
            f"[yellow]awaiting {payload.get('checkpoint')}[/yellow]: {payload.get('title')}"
        )


@app.command()
def show(
    campaign: str = typer.Argument(..., help="Campaign id."),
    out: str = typer.Option("", "--out", "-o", help="Write to this file instead of stdout."),
    probes: bool = typer.Option(
        False, "--probes", help="Keep the raw probe rows (large; omitted by default)."
    ),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
) -> None:
    """Print the checkpoint document in full — the panel at a checkpoint is only a preview."""
    graph = _build(mock, campaign)
    state = _state(graph, campaign)
    if not state:
        console.print(f"[red]No campaign named {campaign!r}.[/red]")
        raise typer.Exit(code=1)
    payload = _pending_interrupt(graph, campaign)
    if payload is None:
        console.print("[yellow]No pending checkpoint; showing the latest ProblemSpec.[/yellow]")
        spec = state.get("problem_spec")
        if spec is None:
            raise typer.Exit(code=0)
        document: Any = spec.model_dump(mode="json")
    else:
        document = checkpoint_document(payload)
    if not probes and isinstance(document, dict):
        document.pop("investigation_notes", None)
    text = json.dumps(document, indent=2, default=str)
    if out:
        Path(out).expanduser().write_text(text, encoding="utf-8")
        console.print(f"[green]wrote[/green] {len(text)} chars to {out}")
        return
    print(text)


@app.command()
def report(
    campaign: str = typer.Argument(..., help="Campaign id."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
) -> None:
    """Print the ScientificConclusion and MLflow references."""
    graph = _build(mock, campaign)
    state = _state(graph, campaign)
    conclusion = state.get("conclusion")
    if conclusion is None:
        console.print("[yellow]No conclusion yet.[/yellow]")
        raise typer.Exit(code=0)
    console.print(Panel(conclusion.report_markdown or "", title="Scientific Conclusion"))
    table = Table(show_header=False)
    table.add_row("qa_decision", str(state.get("qa_decision")))
    table.add_row("recommendation", conclusion.recommendation)
    table.add_row("mlflow_experiment_id", str(state.get("mlflow_experiment_id") or ""))
    table.add_row("mlflow_final_run_id", str(state.get("mlflow_final_run_id") or ""))
    table.add_row("contract_hash", str(state.get("contract_hash") or "")[:12])
    console.print(table)


@app.command()
def promote(
    campaign: str = typer.Argument(..., help="Campaign id."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
    model_name: str = typer.Option("", help="UC model name; defaults to {catalog}.{schema}.amt."),
) -> None:
    """HC4 — record the promotion decision and (when QA=PASS) register the model."""
    graph = _build(mock, campaign)
    state = _state(graph, campaign)
    decision = state.get("qa_decision")
    if decision != QADecision.PASS:
        console.print(
            f"[yellow]QA decision is {decision}; registering a candidate requires PASS.[/yellow]"
        )
    _resume(campaign, {"approved": True, "decision": "promote"}, mock)
    if mock or decision != QADecision.PASS:
        return
    frozen = _state(graph, campaign).get("frozen_champion")
    if frozen is None or not frozen.model_artifact_uri:
        console.print("[yellow]No model artifact to register.[/yellow]")
        return
    from amt.databricks.mlflow_client import register_candidate

    settings = get_settings()
    name = model_name or f"{settings.catalog}.{settings.schema_name}.amt_{campaign}"
    version = register_candidate(frozen.model_artifact_uri, name)
    console.print(f"[green]Registered[/green] {name} version {getattr(version, 'version', '?')}")


@demo_app.command("seed")
def demo_seed(
    catalog: str = typer.Option("", help="Target catalog; defaults to settings."),
    schema: str = typer.Option("", help="Target schema; defaults to settings."),
    users: int = typer.Option(5000),
    items: int = typer.Option(2000),
) -> None:
    """Generate synthetic interaction data on Databricks (implemented in `amt.jobs`)."""
    try:
        from amt.jobs.demo import seed_demo_data  # provided by the Databricks jobs layer
    except ImportError:
        console.print(
            "[yellow]Demo data generation lives in the Databricks jobs package "
            "(`amt.jobs.demo`) and is not available in this install.[/yellow]"
        )
        raise typer.Exit(code=1) from None
    settings = get_settings()
    seed_demo_data(
        catalog=catalog or settings.catalog,
        schema=schema or settings.schema_name,
        users=users,
        items=items,
    )


if __name__ == "__main__":  # pragma: no cover
    app()
