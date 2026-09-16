"""``amt`` — Typer CLI driving the campaign graph across human checkpoints (PLAN §8)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from amt import monitor
from amt.graph.build import build_graph, sqlite_checkpointer
from amt.graph.state import describe, new_campaign_state
from amt.schemas.common import ExperimentBudget
from amt.schemas.qa import QADecision
from amt.settings import get_settings

HELP = """Autonomous ML experimentation on Databricks.

A campaign drafts a ProblemSpec from the data, freezes an EvaluationContract,
trains a baseline, then loops a Trainer inside one hypothesis at a time. Five
decisions are yours: HC2 the spec, HC3 the contract, HC_BACKLOG the hypotheses,
HC_FEATURE every tabular shape at once, HC4 the conclusion.

Driving one, in order:

\b
  doctor                    can this machine run a campaign at all
  start "<problem>" -c <id> begin, and stop at HC2
  status <id>               where it is and what it is waiting on
  show <id>                 the pending checkpoint's document in full
  approve <id>              answer it; --keep/--drop edit a checklist
  reject <id> -m "<why>"    send it back with feedback
  report <id>               the scientific conclusion
  promote <id>              HC4, and register the model when QA passed

\b
`run <id>` is all of that in one interactive session. `config` prints the
resolved settings and where each value came from. `list` names every local
campaign. Long form: docs/RUNBOOK.md.
"""

app = typer.Typer(
    name="amt",
    help=HELP,
    no_args_is_help=True,
    add_completion=False,
)
demo_app = typer.Typer(help="Synthetic demo data helpers (Databricks jobs package).")
app.add_typer(demo_app, name="demo")

console = Console()


# --- wiring -----------------------------------------------------------------------
def _build(
    mock: bool,
    campaign_id: str,
    max_iterations: int | None = None,
    gate_champions: bool = True,
) -> Any:
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
            gate_champions=gate_champions,
        )

    from amt.databricks.sql import ReadOnlySQL  # real probe for Stage 1
    from amt.executor.databricks import DatabricksExecutor  # provided by the jobs layer

    return build_graph(
        executor=DatabricksExecutor(settings=settings, campaign_id=campaign_id),
        settings=settings,
        sql_probe=ReadOnlySQL(settings=settings),
        checkpointer=checkpointer,
        use_mlflow=True,
        gate_champions=gate_champions,
    )


#: The read model lives in `amt.monitor` so the web surface reads a campaign
#: the same way this does. These are the names the CLI uses.
_config = monitor.config_for
_pending_interrupt = monitor.pending_interrupt
_state = monitor.state


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


PREVIEW_CHARS = monitor.PREVIEW_CHARS
FIT_FIELDS = monitor.FIT_FIELDS
checkpoint_document = monitor.checkpoint_document
backlog_rows = monitor.backlog_rows
shape_rows = monitor.shape_rows


def _render_checkpoint(payload: dict) -> tuple[str, bool]:
    return monitor.clip(json.dumps(checkpoint_document(payload), indent=2, default=str))


#: Columns worth a terminal column at each checkpoint. `title` is prose and
#: belongs in the JSON below the table, not squeezed into a cell.
_HIDDEN_COLUMNS = {"HC_BACKLOG": ("title", "note")}

_SUMMARY_TITLES = {
    "HC_BACKLOG": "hypothesis backlog",
    "HC_FEATURE": "feature/fit shapes",
    "HC_CHAMPION": "hypothesis champion vs baseline",
}


def summarize(payload: dict) -> Any:
    """A few lines answering what a reviewer asks before reading the JSON.

    The JSON below it is the document of record; this is the part that decides
    whether to read it. ``None`` when a checkpoint has nothing worth
    summarizing separately.
    """
    rows = monitor.summary_rows(payload)
    if not rows:
        return None
    checkpoint = str(payload.get("checkpoint") or "")
    return _table(
        _SUMMARY_TITLES.get(checkpoint, checkpoint),
        rows,
        drop=_HIDDEN_COLUMNS.get(checkpoint, ()),
    )


def summarize_report(report: Any) -> Table:
    """The experimentation report as a terminal table; the HTML is the full read."""
    return _table("experimentation report", monitor.report_rows(report))


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
    gate_champions: bool = typer.Option(
        True,
        "--gate-champions/--no-gate-champions",
        help="Review each hypothesis champion at HC_CHAMPION before the next one runs.",
    ),
) -> None:
    """Begin a campaign and run to HC2, the ProblemSpec approval."""
    graph = _build(mock, campaign, gate_champions=gate_champions)
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
    freeze: bool = typer.Option(
        False, "--freeze", help="At HC_CHAMPION: stop exploring and take this champion to QA."
    ),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
) -> None:
    """Approve the current human checkpoint and continue.

    At HC_BACKLOG, HC_FEATURE and HC_CHAMPION this is a checklist: `--keep`
    also sets the order the kept items run in, and `--drop` abandons an item
    without paying for the agent to rewrite the whole list. `--freeze` answers
    HC_CHAMPION with "this is good enough; do not spend the rest of the budget".
    """
    _resume(
        campaign,
        {"approved": True, "keep": keep, "drop": drop, "freeze": freeze},
        mock,
    )


@app.command()
def reject(
    campaign: str = typer.Argument(..., help="Campaign id."),
    message: str = typer.Option(..., "--message", "-m", help="Feedback for the agent."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
) -> None:
    """Send the pending checkpoint back with feedback; its stage is re-run.

    Costs a round trip through the agent, which is why HC_BACKLOG and
    HC_FEATURE take `--keep` / `--drop` instead for a per-item objection.
    """
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
def run(
    campaign: str = typer.Argument(..., help="Campaign id."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
) -> None:
    """Drive a campaign to its end from one terminal, answering checkpoints inline.

    The same stream, checkpoints and resumes the other verbs use, in a loop:
    `approve` / `reject <why>` / `keep H1,H3` / `drop H2` / `show` / `status` /
    `quit`. Quitting loses nothing — the campaign is in the checkpointer.
    """
    from langgraph.types import Command

    graph = _build(mock, campaign)
    if not _state(graph, campaign):
        console.print(f"[red]No campaign named {campaign!r}.[/red] Run `amt start` first.")
        raise typer.Exit(code=1)

    # `None` continues from the checkpoint: a campaign already sitting at an
    # interrupt just re-enters that node and suspends again, which is cheap
    # because every interrupting node does nothing but build its payload.
    payload: Any = None
    while True:
        _run(graph, payload, campaign)
        payload = None
        pending = _pending_interrupt(graph, campaign)
        _report_position(graph, campaign)
        if pending is None:
            return
        decision = _prompt_decision(graph, campaign, pending)
        if decision is None:
            console.print("[dim]left at the checkpoint; `amt run` picks it up here.[/dim]")
            return
        payload = Command(resume=decision)


def _prompt_decision(graph: Any, campaign: str, pending: dict) -> dict | None:
    """Ask the operator what to do about ``pending``. ``None`` means stop here."""
    while True:
        answer = typer.prompt(
            f"{pending.get('checkpoint')} > approve | reject <why> | keep <ids> | "
            "drop <ids> | freeze | show | status | quit",
            default="show",
            show_default=False,
        ).strip()
        verb, _, rest = answer.partition(" ")
        verb, rest = verb.lower(), rest.strip()
        if verb in {"quit", "q", "exit"}:
            return None
        if verb in {"approve", "a", "y"}:
            return {"approved": True}
        if verb in {"reject", "r", "n"}:
            if not rest:
                console.print("[red]reject needs a reason[/red] — the agent redrafts from it.")
                continue
            return {"approved": False, "feedback": rest}
        if verb == "keep":
            return {"approved": True, "keep": rest}
        if verb == "drop":
            return {"approved": True, "drop": rest}
        if verb == "freeze":
            return {"approved": True, "freeze": True}
        if verb == "show":
            print(json.dumps(checkpoint_document(pending), indent=2, default=str))
            continue
        if verb == "status":
            console.print(_status_table(campaign, _state(graph, campaign)))
            continue
        console.print(f"[red]unknown verb {verb!r}[/red]")


def _status_table(campaign: str, state: dict) -> Table:
    table = Table(title=f"campaign {campaign}", show_header=False)
    for key, value in describe(state).items():
        text = str(value)
        if "(high)" in text or key == "trainer_asks_refused":
            text = f"[yellow]{text}[/yellow]"
        table.add_row(str(key), text)
    return table


@app.command()
def status(
    campaign: str = typer.Argument(..., help="Campaign id."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
) -> None:
    """Where the campaign is, what it is waiting on, and what it has spent."""
    graph = _build(mock, campaign)
    state = _state(graph, campaign)
    if not state:
        console.print(f"[red]No campaign named {campaign!r}.[/red]")
        raise typer.Exit(code=1)
    console.print(_status_table(campaign, state))
    payload = _pending_interrupt(graph, campaign)
    if payload:
        console.print(
            f"[yellow]awaiting {payload.get('checkpoint')}[/yellow]: {payload.get('title')}"
        )


@app.command(name="list")
def list_campaigns() -> None:
    """Every campaign in the local checkpoint database, most recent first."""
    settings = get_settings()
    campaigns = monitor.list_campaigns(settings.resolved_checkpoint_db())
    if not campaigns:
        console.print(f"[yellow]no campaigns in {settings.resolved_checkpoint_db()}[/yellow]")
        return
    for campaign in campaigns:
        console.print(campaign)


@app.command()
def config() -> None:
    """Print the resolved settings, masked, with where each value came from.

    The trap this exists for: an `AMT_*` variable exported in the shell
    silently shadows `.env`, and the campaign then writes to a catalog nobody
    chose. `source` is what tells you.
    """
    table = Table(title="resolved settings", header_style="bold")
    for column in ("setting", "value", "source"):
        table.add_column(column, overflow="fold")
    for row in monitor.masked_settings(get_settings()):
        table.add_row(row["setting"], row["value"], row["source"])
    console.print(table)


@app.command()
def web(
    host: str = typer.Option("127.0.0.1", help="Bind address. Localhost by design."),
    port: int = typer.Option(8765, help="Port."),
    mock: bool = typer.Option(False, "--mock", help="Offline dry run."),
    gate_champions: bool = typer.Option(
        True, "--gate-champions/--no-gate-champions", help="Review each hypothesis champion."
    ),
    debug: bool = typer.Option(False, "--debug", help="Flask debug mode."),
) -> None:
    """Serve the campaign surface in a browser, with the CLI as the fallback.

    Same graph, same checkpointer, same decisions — what a browser adds is
    somewhere for the twenty minutes between answering a checkpoint and
    reaching the next one to go. No auth: this runs as you, on your machine,
    bound to localhost.
    """
    try:
        from amt.web.app import create_app
    except ImportError:
        console.print(
            "[red]the web surface needs the `web` extra[/red] — "
            "`uv sync --extra web`, then try again."
        )
        raise typer.Exit(code=1) from None

    console.print(f"[green]amt[/green] http://{host}:{port}")
    create_app(mock=mock, gate_champions=gate_champions).run(
        host=host, port=port, debug=debug, threaded=True
    )


#: Families whose backend is not on a bare Databricks runtime. An ML runtime
#: carries them; a plain one does not, and `requires_pip` then installs them
#: per run -- which works, but on an existing cluster the install is
#: cluster-scoped and accumulates.
_NEEDS_ML_RUNTIME = ("sasrec", "xgboost_*", "lightgbm_*")


def _existing_cluster_report(settings: Any, cluster_id: str) -> str:
    """What running on ``cluster_id`` actually costs you, read off the cluster.

    Deliberately specific rather than a blanket refusal. Pinning a cluster is a
    choice an operator is entitled to make -- a warm cluster skips the ~4.5
    minutes a job cluster spends starting, which across a twelve-iteration
    campaign is an hour. What it costs is real but narrower than "no tags":
    static cluster tags *do* reach `system.billing.usage.custom_tags`; what
    cannot be there are the per-run dimensions, because they vary per run.
    """
    from amt.databricks.client import get_workspace_client

    cluster = get_workspace_client(settings).clusters.get(cluster_id=cluster_id)
    tags = dict(cluster.custom_tags or {})
    runtime = str(cluster.spark_version or "")
    workers = int(cluster.num_workers or 0)
    single_node = tags.get("ResourceClass") == "SingleNode"

    notes = [
        f"AMT_EXISTING_CLUSTER_ID={cluster_id} ({cluster.cluster_name}):",
        f"  {runtime}, {cluster.node_type_id}, "
        + ("single node" if single_node else f"{workers} worker(s)"),
    ]

    # What the operator asked about: is this attributable at all?
    if tags.get("amt"):
        notes.append(
            f"+ tagged amt={tags['amt']}, so its cost IS attributable to AMT in "
            "system.billing.usage.custom_tags"
        )
    else:
        notes.append(
            "- no amt tag: its cost is indistinguishable from anything else run "
            "on this cluster. Add a cluster tag amt=true"
        )
    # What a static tag structurally cannot do, whatever it is set to.
    notes.append(
        "- per-run tags (amt_campaign_id, amt_role, amt_hypothesis_id) cannot "
        "live on a shared cluster, so cost is not attributable per campaign, "
        "role or hypothesis"
    )
    notes.append("- billed as All-Purpose, roughly twice the Jobs Compute rate")

    if "-ml-" not in runtime:
        notes.append(
            f"- {runtime} is not an ML runtime: the booster and sequence families "
            f"({', '.join(_NEEDS_ML_RUNTIME)}) are installed per run instead, "
            "and on an existing cluster those installs are cluster-scoped and "
            "accumulate (see scripts/clear_cluster_wheels.py)"
        )
    if single_node or workers == 0:
        notes.append(
            "- single node: fine for the boosters, which fit on one machine, but "
            "materialization shuffles the whole source through Spark -- the "
            "62.6M-impression FYL split took 11 minutes on 1 driver + 2 workers"
        )
    return "\n".join(notes)


@app.command()
def doctor(
    tables: str = typer.Option(
        "", "--tables", help="Comma-separated tables the campaign must be able to read."
    ),
    llm: bool = typer.Option(True, "--llm/--no-llm", help="Spend one structured-output call."),
) -> None:
    """Check everything a campaign needs before it spends a cluster minute.

    Auth, the warehouse, the artifact volume, the compute spec, MLflow and one
    real structured-output round-trip. Every failure here used to surface as a
    job that died twenty minutes in.
    """
    settings = get_settings()
    #: ``(name, severity, detail)`` where severity is ok / warn / fail. A
    #: warning is a real finding an operator may accept; only a failure exits
    #: non-zero, because a check that blocks a choice the operator is entitled
    #: to make just teaches them to stop running it.
    checks: list[tuple[str, str, str]] = []

    class Warn(Exception):
        """Raised by a check that found something worth saying, not a blocker."""

    def check(name: str, fn: Any) -> Any:
        try:
            detail = fn()
            checks.append((name, "ok", str(detail or "")))
            return detail
        except Warn as warning:
            checks.append((name, "warn", str(warning)))
            return None
        except Exception as exc:
            checks.append((name, "fail", f"{type(exc).__name__}: {exc}"))
            return None

    def _auth() -> str:
        from amt.databricks.client import get_workspace_client

        user = get_workspace_client(settings).current_user.me()
        return str(getattr(user, "user_name", "") or getattr(user, "id", ""))

    def _warehouse() -> str:
        from amt.databricks.sql import ReadOnlySQL

        probe = ReadOnlySQL(settings=settings)
        if int(probe.scalar("SELECT 1") or 0) != 1:
            raise RuntimeError("warehouse did not return 1")
        return f"warehouse {settings.warehouse_id}"

    def _volume() -> str:
        from amt.databricks.client import get_workspace_client
        from amt.databricks.wheel import upload_bytes

        path = f"{settings.artifact_volume.rstrip('/')}/.amt-doctor"
        upload_bytes(get_workspace_client(settings), path, b"ok")
        return path

    def _compute() -> str:
        from amt.databricks.client import get_workspace_client

        spec = settings.cluster_spec()
        if spec.get("existing_cluster_id"):
            raise Warn(_existing_cluster_report(settings, str(spec["existing_cluster_id"])))
        # A node type from the wrong cloud passes every local check and fails at
        # cluster launch, twenty minutes into the first campaign.
        node_type = str(spec.get("node_type_id") or "")
        offered = get_workspace_client(settings).clusters.list_node_types()
        available = {str(n.node_type_id) for n in (getattr(offered, "node_types", None) or [])}
        if node_type not in available:
            raise RuntimeError(
                f"AMT_NODE_TYPE_ID={node_type!r} does not exist in this workspace "
                f"({len(available)} types offered); it is probably from another cloud"
            )
        described = f"job cluster {node_type} x{spec.get('num_workers')}, {settings.spark_version}"
        if "-ml-" not in settings.spark_version:
            raise Warn(
                f"{described}. AMT_SPARK_VERSION is not an ML runtime, so the "
                f"booster and sequence families ({', '.join(_NEEDS_ML_RUNTIME)}) "
                "are pip-installed on every run -- a few minutes per job. An "
                "ML runtime carries them already."
            )
        return described

    def _mlflow() -> str:
        from amt.databricks.mlflow_client import set_tracking_uri

        set_tracking_uri(settings)
        if settings.mlflow_experiment_id:
            import mlflow

            return str(mlflow.get_experiment(settings.mlflow_experiment_id).name)
        return f"root {settings.mlflow_experiment_root}"

    def _reads() -> str:
        from amt.databricks.sql import ReadOnlySQL

        probe = ReadOnlySQL(settings=settings)
        wanted = [t.strip() for t in tables.split(",") if t.strip()]
        for table in wanted:
            probe.describe_table(table)
        return f"{len(wanted)} table(s) readable" if wanted else "none requested"

    def _llm() -> str:
        from amt.agents.base import StructuredAgent
        from amt.schemas.qa import Phase0Triage

        agent = StructuredAgent(
            name="doctor",
            prompt="Return a Phase0Triage saying a missing preprocessing artifact is fixable.",
            output_model=Phase0Triage,
            model=settings.planner_model,
            settings=settings,
        )
        return f"{settings.planner_model} -> fixable={agent.invoke('ping').fixable}"

    check("auth", _auth)
    check("sql warehouse", _warehouse)
    check("artifact volume writable", _volume)
    check("compute spec", _compute)
    check("mlflow", _mlflow)
    if tables:
        check("source tables readable", _reads)
    if llm:
        check("llm structured output", _llm)

    table = Table(title="amt doctor", header_style="bold")
    for column in ("check", "", "detail"):
        table.add_column(column, overflow="fold")
    marks = {
        "ok": "[green]ok[/green]",
        "warn": "[yellow]warn[/yellow]",
        "fail": "[red]FAIL[/red]",
    }
    for name, severity, detail in checks:
        table.add_row(name, marks[severity], detail)
    console.print(table)

    warnings = [name for name, severity, _ in checks if severity == "warn"]
    if warnings:
        console.print(
            f"[yellow]{len(warnings)} warning(s)[/yellow] — a campaign will run. "
            "Read them and decide."
        )
    if any(severity == "fail" for _name, severity, _detail in checks):
        raise typer.Exit(code=1)


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

    report = state.get("experimentation_report")
    if report is not None:
        console.print(summarize_report(report))
    path = str(state.get("report_path") or "")
    if path:
        console.print(f"[green]report[/green] file://{path}")


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
