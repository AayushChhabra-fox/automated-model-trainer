"""The web surface drives a campaign to the same terminal state the CLI does.

The app is thin on purpose: the graph is the backend, `amt.monitor` is the read
model and `Command(resume=...)` is the write. These tests exist to prove that
a browser and a terminal cannot disagree about a campaign.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from amt.graph.build import build_graph
from amt.schemas.qa import QADecision
from amt.testing import mock_stack
from amt.web.runner import CampaignRunner, RunnerBusy

flask = pytest.importorskip("flask")

CAMPAIGN = "web-campaign"
PROBLEM = "Does a re-ranker beat the incumbent order of the For You rail?"


@pytest.fixture
def web(tmp_path, monkeypatch):
    """A Flask test client over a mock stack and a real SQLite checkpointer.

    SQLite rather than MemorySaver because the campaign list reads the
    checkpoints table directly, and that is a route worth covering.
    """
    import sqlite3

    from amt.settings import Settings
    from amt.web.app import create_app

    db = tmp_path / "checkpoints.sqlite"
    settings = Settings(_env_file=None, AMT_CHECKPOINT_DB=str(db), AMT_CATALOG="main")
    monkeypatch.setattr("amt.settings.get_settings", lambda: settings)
    monkeypatch.setattr("amt.web.app.get_settings", lambda: settings, raising=False)

    executor, llm_factory, responder = mock_stack(qa_outcome="PASS", converge_after=1)
    connection = sqlite3.connect(str(db), check_same_thread=False)
    connection.execute("PRAGMA busy_timeout=5000")
    saver = SqliteSaver(connection)

    def build(campaign_id: str):
        return build_graph(
            executor=executor,
            llm_factory=llm_factory,
            settings=settings,
            checkpointer=saver,
            use_mlflow=False,
        )

    app = create_app(build=build)
    app.config.update(TESTING=True)
    client = app.test_client()
    client.amt = type(  # type: ignore[attr-defined]
        "Ctx", (), {"build": staticmethod(build), "app": app, "responder": responder}
    )
    return client


def _wait(client, campaign_id: str = CAMPAIGN) -> None:
    """Let the background runner finish, as the browser's refresh would."""
    client.amt.app.config["AMT_RUNNERS"].for_campaign(campaign_id).join(timeout=60)


def _pending(client, campaign_id: str = CAMPAIGN) -> str:
    from amt import monitor

    payload = monitor.pending_interrupt(client.amt.build(campaign_id), campaign_id)
    return str(payload.get("checkpoint") or "") if payload else ""


def _start(client) -> None:
    response = client.post(
        "/new",
        data={"campaign_id": CAMPAIGN, "problem": PROBLEM, "max_hypotheses": "1"},
    )
    assert response.status_code == 302
    _wait(client)


def _decide(client, **form) -> None:
    response = client.post(f"/c/{CAMPAIGN}/decide", data={"verb": "approve", **form})
    assert response.status_code == 302, response.data[:400]
    _wait(client)


# --- the whole campaign, through the browser -----------------------------


def test_the_browser_reaches_the_same_terminal_state_as_the_cli(web):
    from amt import monitor

    _start(web)
    assert _pending(web) == "HC2"

    for expected in ("HC3", "HC_BACKLOG", "HC_CHAMPION", "HC4"):
        _decide(web)
        assert _pending(web) == expected, f"expected {expected}"

    web.post(f"/c/{CAMPAIGN}/decide", data={"verb": "promote"})
    _wait(web)

    state = monitor.state(web.amt.build(CAMPAIGN), CAMPAIGN)
    assert state["status"] == "done"
    assert state["qa_decision"] is QADecision.PASS
    assert state["promotion_decision"] == "promote"


def test_the_campaign_page_shows_the_pending_checkpoint_and_its_verbs(web):
    _start(web)

    page = web.get(f"/c/{CAMPAIGN}").data.decode()

    assert "HC2" in page
    assert 'value="approve"' in page
    assert "reject with feedback" in page
    assert "the document of record" in page


def test_the_checklist_gates_offer_keep_and_drop(web):
    _start(web)
    _decide(web)  # HC2 -> HC3
    _decide(web)  # HC3 -> HC_BACKLOG
    assert _pending(web) == "HC_BACKLOG"

    page = web.get(f"/c/{CAMPAIGN}").data.decode()

    assert 'name="keep"' in page and 'name="drop"' in page
    assert "hyp-1" in page, "the backlog checklist is rendered as a table"


def test_the_champion_review_offers_freeze(web):
    _start(web)
    _decide(web)
    _decide(web)
    _decide(web)
    assert _pending(web) == "HC_CHAMPION"

    page = web.get(f"/c/{CAMPAIGN}").data.decode()

    assert "freeze" in page


def test_rejecting_from_the_browser_redrafts_the_stage(web):
    _start(web)
    assert _pending(web) == "HC2"

    web.post(
        f"/c/{CAMPAIGN}/decide",
        data={"verb": "reject", "feedback": "use a 30-day label window"},
    )
    _wait(web)

    assert _pending(web) == "HC2", "a rejection re-runs the stage"
    assert web.amt.responder.counts["spec"] == 2


def test_dropping_from_the_browser_abandons_only_that_hypothesis(web):
    from amt import monitor

    _start(web)
    _decide(web)
    _decide(web)
    assert _pending(web) == "HC_BACKLOG"

    _decide(web, drop="hyp-1")

    state = monitor.state(web.amt.build(CAMPAIGN), CAMPAIGN)
    assert {h.hypothesis_id: h.status for h in state["hypotheses"]} == {"hyp-1": "abandoned"}


def test_the_progress_log_records_every_node(web):
    _start(web)

    page = web.get(f"/c/{CAMPAIGN}").data.decode()

    assert "progress" in page
    assert "problem_spec" in page
    assert "awaiting HC2" in page


def test_the_report_page_serves_the_html_report(web):
    _start(web)
    for _ in range(4):
        _decide(web)
    assert _pending(web) == "HC4"

    page = web.get(f"/c/{CAMPAIGN}/report").data.decode()

    assert "Is the offline metric measuring the product?" in page
    assert "top-slot conversions per 1,000 renders" in page
    # And the campaign page links to it with the headline numbers inline.
    campaign_page = web.get(f"/c/{CAMPAIGN}").data.decode()
    assert "the full experimentation report" in campaign_page
    assert "proxy sanity" in campaign_page


def test_the_index_lists_the_campaign_and_what_it_waits_on(web):
    _start(web)

    page = web.get("/").data.decode()

    assert CAMPAIGN in page
    assert "HC2" in page


def test_an_empty_index_says_what_to_do_next(web):
    page = web.get("/").data.decode()
    assert "No campaigns yet" in page


# --- the edges ------------------------------------------------------------


def test_an_unknown_campaign_is_a_404_not_a_traceback(web):
    assert web.get("/c/nope").status_code == 404
    assert web.get("/c/nope/report").status_code == 404


def test_a_campaign_id_cannot_be_reused(web):
    _start(web)

    response = web.post("/new", data={"campaign_id": CAMPAIGN, "problem": PROBLEM})

    assert response.status_code == 400
    assert b"already exists" in response.data


def test_a_campaign_needs_both_an_id_and_a_problem(web):
    assert web.post("/new", data={"campaign_id": "x"}).status_code == 400
    assert web.post("/new", data={"problem": PROBLEM}).status_code == 400


def test_a_second_resume_while_one_is_live_is_refused_rather_than_raced():
    """Two threads writing one checkpoint thread is not something LangGraph makes safe."""
    import threading

    runner = CampaignRunner("c")
    release = threading.Event()
    runner.start(lambda log: release.wait(timeout=5))

    with pytest.raises(RunnerBusy):
        runner.start(lambda log: None)

    release.set()
    runner.join(timeout=5)
    # And once it is done, the campaign can be driven again.
    runner.start(lambda log: log("node", "second run"))
    runner.join(timeout=5)
    assert any("second run" in event.message for event in runner.events)


def test_a_failing_run_surfaces_on_the_page_instead_of_vanishing():
    runner = CampaignRunner("c")

    runner.start(lambda log: (_ for _ in ()).throw(RuntimeError("the cluster died")))
    runner.join(timeout=5)

    assert "the cluster died" in runner.error
    assert any(event.kind == "error" for event in runner.events)


def test_the_event_log_is_bounded():
    runner = CampaignRunner("c")
    for i in range(1500):
        runner.log("node", f"line {i}")
    assert len(runner.events) == 1000
