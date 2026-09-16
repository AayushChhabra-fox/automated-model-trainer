"""The read model, and the two numbers `status` used to leave out."""

from __future__ import annotations

import sqlite3

from amt.graph.human import HumanDecision, gate_update
from amt.graph.state import describe, new_campaign_state
from amt.monitor import config_for, list_campaigns, masked_settings, summary_rows
from amt.settings import Settings


def test_a_campaign_is_a_thread_id():
    assert config_for("fyl-ctv")["configurable"]["thread_id"] == "fyl-ctv"


def test_the_recursion_limit_covers_a_batched_gate_plus_a_trainer_loop():
    assert config_for("c")["recursion_limit"] >= 150


# --- listing campaigns ----------------------------------------------------


def test_campaigns_are_listed_most_recently_written_first(tmp_path):
    db = tmp_path / "checkpoints.sqlite"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint_id TEXT)")
    connection.executemany(
        "INSERT INTO checkpoints VALUES (?, ?)",
        [("older", "001"), ("newer", "009"), ("older", "002")],
    )
    connection.commit()
    connection.close()

    assert list_campaigns(db) == ["newer", "older"]


def test_a_missing_or_empty_database_is_not_an_error(tmp_path):
    assert list_campaigns(tmp_path / "nothing.sqlite") == []
    empty = tmp_path / "empty.sqlite"
    sqlite3.connect(empty).close()
    assert list_campaigns(empty) == []


# --- masked settings ------------------------------------------------------


def test_secrets_are_masked_but_their_presence_is_visible():
    settings = Settings(_env_file=None, DATABRICKS_TOKEN="dapiabcdef0123456789", AMT_CATALOG="cat")
    rows = {row["setting"]: row for row in masked_settings(settings)}

    token = rows["databricks_token"]["value"]
    assert "abcdef0123456789" not in token
    assert token.startswith("dapi") and "20 chars" in token


def test_the_source_column_exposes_the_shell_shadowing_env_file():
    """An exported AMT_CATALOG silently overrides `.env`; `source` is the tell."""
    settings = Settings(_env_file=None, AMT_CATALOG="from_shell")
    rows = {row["setting"]: row for row in masked_settings(settings)}

    assert rows["catalog"] == {
        "setting": "catalog",
        "value": "from_shell",
        "source": "env / .env",
    }
    assert rows["phase0_max_attempts"]["source"] == "default"


# --- gate cycles and the clamp notice -------------------------------------


def test_an_approval_does_not_count_as_a_gate_cycle():
    assert gate_update({"gate_cycles": 2}, HumanDecision(approved=True))["gate_cycles"] == 2


def test_every_rejection_counts():
    state = {"gate_cycles": 0}
    for _ in range(3):
        state = {**state, **gate_update(state, HumanDecision(approved=False, feedback="no"))}
    assert state["gate_cycles"] == 3


def test_status_flags_a_campaign_that_has_spent_the_day_arguing():
    """Eight of these cost no compute, so nothing else in `status` moved."""
    assert describe({**new_campaign_state("p", "c"), "gate_cycles": 2})["gate_cycles"] == "2"
    assert "high" in describe({**new_campaign_state("p", "c"), "gate_cycles": 3})["gate_cycles"]


def test_status_says_when_the_trainer_asked_for_more_than_it_got():
    state = {
        **new_campaign_state("p", "c"),
        "structural_asks_clamped": ["hyp-1 it2: feature_recipe (kept the approved recipe)"],
    }

    described = describe(state)

    assert "kept the approved shape" in described["trainer_asks_refused"]


def test_a_campaign_with_no_clamped_asks_says_nothing_about_them():
    assert "trainer_asks_refused" not in describe(new_campaign_state("p", "c"))


def test_the_backlog_line_breaks_down_by_status(evaluation_contract):
    from amt.schemas.hypothesis import Hypothesis

    def hypothesis(hypothesis_id, status):
        return Hypothesis(
            hypothesis_id=hypothesis_id,
            title="t",
            proposed_change="c",
            why_it_should_improve="w",
            motivating_evidence="e",
            experiment_design="d",
            metric_expected_to_improve="ndcg@10",
            falsification_criteria="f",
            status=status,
        )

    described = describe(
        {
            **new_campaign_state("p", "c"),
            "hypotheses": [
                hypothesis("h1", "pending"),
                hypothesis("h2", "abandoned"),
                hypothesis("h3", "infeasible"),
            ],
            "hypotheses_approved": True,
        }
    )

    assert described["hypotheses"] == "3 (1 abandoned, 1 infeasible, 1 pending) approved"


# --- which question a checkpoint is asking --------------------------------


def test_each_checklist_checkpoint_has_rows_and_the_others_do_not():
    assert summary_rows({"checkpoint": "HC_BACKLOG", "hypotheses": [{}]})
    assert summary_rows({"checkpoint": "HC_FEATURE", "shapes": {"h1": {}}})
    assert summary_rows(
        {"checkpoint": "HC_CHAMPION", "comparison": [{"label": "baseline", "metrics": {}}]}
    )
    assert summary_rows({"checkpoint": "HC2"}) is None
    assert summary_rows({"checkpoint": "HC_FEATURE", "shapes": {}}) == []


def test_the_champion_table_puts_the_primary_first_and_shows_the_delta():
    from amt import monitor

    rows = monitor.champion_rows(
        {
            "primary_metric": "ndcg@4",
            "comparison": [
                {
                    "label": "baseline",
                    "metrics": {"hitrate@1": 0.5, "ndcg@4": 0.8, "n_users": 10, "reproduction/ndcg@4": 0.9},
                },
                {"label": "H1", "metrics": {"ndcg@4": 0.76, "hitrate@1": 0.45, "auc": 0.6}},
            ],
        }
    )

    assert [r["metric"] for r in rows] == ["ndcg@4", "hitrate@1", "auc"], "primary first, no bookkeeping"
    assert rows[0]["Δ H1 vs baseline"].startswith("-0.0400 (-5.0%)")
    assert rows[2]["baseline"] == "—", "a diagnostic the incumbent does not have"
