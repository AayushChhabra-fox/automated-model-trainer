"""Seed a synthetic interactions table through the SQL warehouse.

``amt demo seed`` needs a Spark session, which a laptop does not have; this
writes the same shape of data straight into Unity Catalog so a rehearsal
campaign has something to read. Power-law item popularity, so a popularity
baseline is beatable but not trivially.

    uv run python scripts/seed_demo_table.py [table] [rows]
"""

from __future__ import annotations

import sys

from amt.databricks.client import get_workspace_client
from amt.settings import get_settings

DDL = """
CREATE OR REPLACE TABLE {table} AS
WITH s AS (SELECT explode(sequence(1, {rows})) AS i)
SELECT
  concat('u', cast(pmod(i * 7919, 1500) AS string)) AS user_id,
  concat('t', cast(cast(pow(pmod(i * 104729, 400), 0.7) AS int) AS string)) AS item_id,
  timestamp'2026-08-01 00:00:00'
    + make_interval(0, 0, 0, 0, 0, 0, pmod(i * 613, 3024000)) AS ts,
  'start' AS event_type
FROM s
"""

PROFILE = (
    "SELECT count(*) AS rows, count(DISTINCT user_id) AS users, "
    "count(DISTINCT item_id) AS items, min(ts) AS first_ts, max(ts) AS last_ts FROM {table}"
)


def main(argv: list[str]) -> int:
    settings = get_settings()
    table = argv[0] if argv else settings.table("demo_interactions")
    rows = int(argv[1]) if len(argv) > 1 else 40_000
    client = get_workspace_client(settings)

    for statement in (DDL.format(table=table, rows=rows), PROFILE.format(table=table)):
        response = client.statement_execution.execute_statement(
            statement=statement, warehouse_id=settings.warehouse_id, wait_timeout="50s"
        )
        if response.status.error:
            print(f"failed: {response.status.error.message}")
            return 1
        if response.result and response.result.data_array:
            print(f"{table}: {response.result.data_array[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
