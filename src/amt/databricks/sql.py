"""Read-only Databricks SQL access for the Problem Specification agent.

Enforcement is structural, not prompt-based: :func:`assert_read_only` rejects anything
that is not a single ``SELECT`` / ``WITH`` / ``DESCRIBE`` / ``SHOW`` statement, so the
data-probing agent physically cannot mutate the workspace.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from amt.settings import Settings, get_settings

ALLOWED_PREFIXES: tuple[str, ...] = ("SELECT", "WITH", "DESCRIBE", "SHOW")

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*){0,3}$")


class UnsafeSQLError(ValueError):
    """Raised when a statement is not provably read-only."""


def strip_comments(sql: str) -> str:
    return _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql)).strip()


def assert_read_only(sql: str) -> str:
    """Return the normalized statement, or raise :class:`UnsafeSQLError`.

    Rejects multi-statement payloads, comment-prefixed statements, and anything whose
    first keyword is not in :data:`ALLOWED_PREFIXES`.
    """
    if not isinstance(sql, str) or not sql.strip():
        raise UnsafeSQLError("Empty SQL statement.")
    cleaned = strip_comments(sql)
    if not cleaned:
        raise UnsafeSQLError("SQL statement contains only comments.")
    body = cleaned.rstrip().rstrip(";").strip()
    if ";" in body:
        raise UnsafeSQLError("Multiple SQL statements are not allowed.")
    first = body.split(None, 1)[0].upper() if body.split() else ""
    if first not in ALLOWED_PREFIXES:
        raise UnsafeSQLError(
            f"Only {'/'.join(ALLOWED_PREFIXES)} statements are allowed; got {first or '<empty>'!r}."
        )
    return body


def assert_identifier(name: str) -> str:
    """Validate a table/column identifier before interpolating it into SQL."""
    if not _IDENT.match((name or "").strip()):
        raise UnsafeSQLError(f"Invalid identifier: {name!r}")
    return name.strip()


class SqlProbe(Protocol):
    """The read-only surface exposed to the Problem Specification agent."""

    def describe_table(self, table: str) -> list[dict[str, Any]]: ...

    def profile(self, table: str, columns: list[str] | None = None) -> dict[str, Any]: ...

    def sample(self, table: str, n: int = 10) -> list[dict[str, Any]]: ...


class ReadOnlySQL:
    """Statement Execution API wrapper that only ever runs read-only SQL."""

    def __init__(
        self,
        client: Any | None = None,
        warehouse_id: str | None = None,
        settings: Settings | None = None,
        row_limit: int = 200,
    ) -> None:
        self._settings = settings or get_settings()
        self._client = client
        self.warehouse_id = warehouse_id or self._settings.warehouse_id
        self.row_limit = row_limit

    @property
    def client(self) -> Any:
        if self._client is None:
            from amt.databricks.client import get_workspace_client

            self._client = get_workspace_client(self._settings)
        return self._client

    # --- core -----------------------------------------------------------------
    def query(self, sql: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Execute a read-only statement and return rows as dicts."""
        statement = assert_read_only(sql)
        if not self.warehouse_id:
            raise RuntimeError("DATABRICKS_WAREHOUSE_ID is not configured.")
        resp = self.client.statement_execution.execute_statement(
            statement=statement,
            warehouse_id=self.warehouse_id,
            wait_timeout="50s",
        )
        return _rows_from_response(resp, limit or self.row_limit)

    def scalar(self, sql: str) -> Any:
        rows = self.query(sql, limit=1)
        if not rows:
            return None
        return next(iter(rows[0].values()))

    # --- helpers ---------------------------------------------------------------
    def describe_table(self, table: str) -> list[dict[str, Any]]:
        return self.query(f"DESCRIBE TABLE {assert_identifier(table)}")

    def row_count(self, table: str, where: str | None = None) -> int:
        sql = f"SELECT count(*) AS n FROM {assert_identifier(table)}"
        if where:
            sql += f" WHERE {where}"
        return int(self.scalar(sql) or 0)

    def min_max_ts(self, table: str, column: str) -> dict[str, Any]:
        t, c = assert_identifier(table), assert_identifier(column)
        rows = self.query(f"SELECT min({c}) AS min_ts, max({c}) AS max_ts FROM {t}")
        return rows[0] if rows else {"min_ts": None, "max_ts": None}

    def null_rates(self, table: str, columns: list[str]) -> dict[str, float]:
        t = assert_identifier(table)
        cols = [assert_identifier(c) for c in columns]
        if not cols:
            return {}
        exprs = ", ".join(
            f"avg(CASE WHEN {c} IS NULL THEN 1.0 ELSE 0.0 END) AS `{c}`" for c in cols
        )
        rows = self.query(f"SELECT {exprs} FROM {t}")
        if not rows:
            return {}
        row = rows[0]
        return {c: float(row[c]) if row.get(c) is not None else 0.0 for c in cols if c in row}

    def sample(self, table: str, n: int = 10) -> list[dict[str, Any]]:
        n = max(1, min(int(n), self.row_limit))
        return self.query(f"SELECT * FROM {assert_identifier(table)} LIMIT {n}", limit=n)

    def profile(self, table: str, columns: list[str] | None = None) -> dict[str, Any]:
        """Schema + volume + null rates + (optional) temporal coverage."""
        schema = self.describe_table(table)
        col_names = columns or [
            str(r.get("col_name"))
            for r in schema
            if r.get("col_name") and not str(r.get("col_name")).startswith("#")
        ]
        profile: dict[str, Any] = {
            "table": table,
            "schema": schema,
            "row_count": self.row_count(table),
            "null_rates": self.null_rates(table, col_names[:32]),
        }
        ts_cols = [
            str(r.get("col_name"))
            for r in schema
            if "timestamp" in str(r.get("data_type", "")).lower()
            or "date" in str(r.get("data_type", "")).lower()
        ]
        if ts_cols:
            profile["temporal_coverage"] = {c: self.min_max_ts(table, c) for c in ts_cols[:3]}
        return profile


def _rows_from_response(resp: Any, limit: int) -> list[dict[str, Any]]:
    """Normalize a StatementResponse into a list of dicts."""
    manifest = getattr(resp, "manifest", None)
    schema = getattr(manifest, "schema", None)
    columns = [c.name for c in getattr(schema, "columns", None) or []]
    result = getattr(resp, "result", None)
    data = getattr(result, "data_array", None) or []
    rows: list[dict[str, Any]] = []
    for raw in data[:limit]:
        if columns:
            rows.append(dict(zip(columns, raw)))
        else:
            rows.append({f"c{i}": v for i, v in enumerate(raw)})
    return rows
