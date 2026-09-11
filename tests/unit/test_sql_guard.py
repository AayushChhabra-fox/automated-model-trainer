"""The Stage-1 data probe must be structurally incapable of mutating anything."""

from __future__ import annotations

import pytest

from amt.databricks.sql import (
    ReadOnlySQL,
    UnsafeSQLError,
    assert_identifier,
    assert_read_only,
)


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select count(*) from main.amt.interactions",
        "WITH x AS (SELECT 1) SELECT * FROM x",
        "DESCRIBE TABLE main.amt.interactions",
        "SHOW TABLES IN main.amt",
        "  SELECT 1;  ",
        "-- a comment\nSELECT 1",
    ],
)
def test_read_only_statements_are_allowed(sql):
    assert assert_read_only(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DROP TABLE main.amt.interactions",
        "DELETE FROM main.amt.interactions",
        "UPDATE main.amt.interactions SET x = 1",
        "INSERT INTO main.amt.interactions VALUES (1)",
        "MERGE INTO t USING s ON t.id = s.id",
        "CREATE TABLE t AS SELECT 1",
        "ALTER TABLE t ADD COLUMN c INT",
        "GRANT SELECT ON t TO `u`",
        "TRUNCATE TABLE t",
        "SELECT 1; DROP TABLE t",
        "/* SELECT */ DROP TABLE t",
        "-- SELECT 1\nDROP TABLE t",
        "",
        "   ",
        "-- only a comment",
    ],
)
def test_mutating_statements_are_rejected(sql):
    with pytest.raises(UnsafeSQLError):
        assert_read_only(sql)


def test_identifier_validation():
    assert assert_identifier("main.amt.interactions") == "main.amt.interactions"
    for bad in ("main.amt.interactions; DROP TABLE x", "`weird`", "a b", "", "1table"):
        with pytest.raises(UnsafeSQLError):
            assert_identifier(bad)


class ExplodingClient:
    """Any attempt to reach Databricks fails the test."""

    @property
    def statement_execution(self):  # pragma: no cover - must never be reached
        raise AssertionError("SQL guard let an unsafe statement through")


def test_guard_runs_before_the_client_is_touched():
    sql = ReadOnlySQL(client=ExplodingClient(), warehouse_id="wh-1")
    with pytest.raises(UnsafeSQLError):
        sql.query("DROP TABLE main.amt.interactions")


def test_helpers_build_read_only_statements():
    captured: list[str] = []

    class RecordingSQL(ReadOnlySQL):
        def query(self, sql: str, limit: int | None = None):
            captured.append(assert_read_only(sql))
            return [{"n": 1, "min_ts": None, "max_ts": None, "col_name": "user_id"}]

    sql = RecordingSQL(client=object(), warehouse_id="wh-1")
    sql.describe_table("main.amt.interactions")
    sql.row_count("main.amt.interactions")
    sql.min_max_ts("main.amt.interactions", "event_ts")
    sql.null_rates("main.amt.interactions", ["user_id"])
    sql.sample("main.amt.interactions", 5)
    assert len(captured) == 5
    assert all(s.split()[0].upper() in {"SELECT", "WITH", "DESCRIBE", "SHOW"} for s in captured)


def test_sample_limit_is_bounded():
    class RecordingSQL(ReadOnlySQL):
        def query(self, sql: str, limit: int | None = None):
            assert "LIMIT" in sql
            return []

    RecordingSQL(client=object(), warehouse_id="wh-1", row_limit=50).sample("main.amt.t", 10_000)
