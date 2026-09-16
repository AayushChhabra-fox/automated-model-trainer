"""A thin local surface over the campaign graph.

Thin is the design. ``amt.monitor`` is the read model, ``Command(resume=...)``
is the write, and :mod:`amt.web.runner` is the one piece a browser needs that
a terminal does not. There is no API layer, no client-side state and no build
step: server-rendered HTML with a meta refresh while a campaign is running.

Auth: none, deliberately. This runs on the operator's machine with the
operator's ``.env``, exactly as the CLI does, and binds to 127.0.0.1. Multi-user
deployment is out of scope -- see the README.
"""

from __future__ import annotations

from typing import Any

from amt import monitor
from amt.graph.state import describe, new_campaign_state
from amt.schemas.common import ExperimentBudget
from amt.web.runner import RunnerBusy, Runners

__all__ = ["create_app"]


def create_app(
    build: Any = None,
    mock: bool = False,
    gate_champions: bool = True,
) -> Any:
    """Build the Flask app.

    ``build`` is a ``campaign_id -> compiled graph`` factory, injected so the
    tests drive a ``--mock`` stack through the same routes a real campaign
    uses.
    """
    from flask import Flask, redirect, render_template, request, url_for

    from amt.settings import get_settings

    if build is None:
        from amt.cli import _build as cli_build

        def default_build(campaign_id: str) -> Any:
            return cli_build(mock, campaign_id, gate_champions=gate_champions)

        build = default_build

    app = Flask(__name__)
    app.config["AMT_BUILD"] = build
    runners = Runners(build)
    app.config["AMT_RUNNERS"] = runners

    def graph_for(campaign_id: str) -> Any:
        return build(campaign_id)

    @app.get("/")
    def index():
        settings = get_settings()
        campaigns = []
        for campaign_id in monitor.list_campaigns(settings.resolved_checkpoint_db()):
            state = monitor.state(graph_for(campaign_id), campaign_id)
            campaigns.append(
                {
                    "id": campaign_id,
                    "stage": state.get("stage", ""),
                    "status": state.get("status", ""),
                    "awaiting": _awaiting(graph_for(campaign_id), campaign_id),
                    "running": runners.for_campaign(campaign_id).running,
                }
            )
        return render_template("index.html", campaigns=campaigns)

    @app.get("/c/<campaign_id>")
    def campaign(campaign_id: str):
        graph = graph_for(campaign_id)
        state = monitor.state(graph, campaign_id)
        if not state:
            return render_template("missing.html", campaign_id=campaign_id), 404
        pending = monitor.pending_interrupt(graph, campaign_id)
        runner = runners.for_campaign(campaign_id)
        return render_template(
            "campaign.html",
            campaign_id=campaign_id,
            summary=describe(state),
            pending=pending,
            checklist=monitor.summary_rows(pending) if pending else None,
            document=monitor.clip(_json(monitor.checkpoint_document(pending)))
            if pending
            else ("", False),
            hypotheses=[h.model_dump(mode="json") for h in state.get("hypotheses") or []],
            history=[h.model_dump(mode="json") for h in state.get("experiment_history") or []],
            events=list(reversed(runner.events)),
            running=runner.running,
            error=runner.error,
            report=state.get("experimentation_report"),
            report_rows=monitor.report_rows(state.get("experimentation_report")),
        )

    @app.post("/c/<campaign_id>/decide")
    def decide(campaign_id: str):
        """Answer the pending checkpoint. Builds the same payload the CLI does."""
        from langgraph.types import Command

        verb = (request.form.get("verb") or "").strip()
        payload = {
            "approved": verb != "reject",
            "feedback": request.form.get("feedback") or None,
            "keep": request.form.get("keep") or "",
            "drop": request.form.get("drop") or "",
            "freeze": verb == "freeze",
        }
        if verb == "promote":
            payload["decision"] = "promote"
        try:
            runners.resume(campaign_id, Command(resume=payload))
        except RunnerBusy as exc:
            return render_template("busy.html", campaign_id=campaign_id, detail=str(exc)), 409
        return redirect(url_for("campaign", campaign_id=campaign_id))

    @app.post("/c/<campaign_id>/continue")
    def continue_campaign(campaign_id: str):
        """Pick a campaign back up after a crash or a closed laptop."""
        try:
            runners.resume(campaign_id, None)
        except RunnerBusy as exc:
            return render_template("busy.html", campaign_id=campaign_id, detail=str(exc)), 409
        return redirect(url_for("campaign", campaign_id=campaign_id))

    @app.get("/c/<campaign_id>/report")
    def report(campaign_id: str):
        state = monitor.state(graph_for(campaign_id), campaign_id)
        report = state.get("experimentation_report")
        if report is None:
            return render_template("missing.html", campaign_id=campaign_id), 404
        from amt.report import render_report

        return render_report(report, state.get("conclusion"))

    @app.get("/new")
    def new_form():
        return render_template("new.html")

    @app.post("/new")
    def create():
        campaign_id = (request.form.get("campaign_id") or "").strip()
        problem = (request.form.get("problem") or "").strip()
        if not campaign_id or not problem:
            return render_template(
                "new.html", error="a campaign id and a problem are both required"
            ), 400
        graph = graph_for(campaign_id)
        if monitor.state(graph, campaign_id):
            return render_template("new.html", error=f"{campaign_id} already exists"), 400
        state = new_campaign_state(
            problem_text=problem,
            campaign_id=campaign_id,
            budget=ExperimentBudget(
                max_iterations=int(request.form.get("max_iterations") or 6),
                max_hypotheses=int(request.form.get("max_hypotheses") or 3),
                max_compute_minutes=float(request.form.get("max_compute_minutes") or 240),
            ),
        )
        runners.resume(campaign_id, state)
        return redirect(url_for("campaign", campaign_id=campaign_id))

    return app


def _awaiting(graph: Any, campaign_id: str) -> str:
    pending = monitor.pending_interrupt(graph, campaign_id)
    return str(pending.get("checkpoint") or "") if pending else ""


def _json(document: Any) -> str:
    import json

    return json.dumps(document, indent=2, default=str)
