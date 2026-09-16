"""Confirm the Phase 8 data facts against the table before spending a cluster minute.

Everything the proving campaign's contract asserts about `pr_video_impressions`
is checked here: the container, the surface, the label, the slot parse, the
render key, the window, and the effect of the frozen sampling. Read-only, runs
on the SQL warehouse, costs nothing.

    uv run python scripts/probe_fyl.py
"""

from __future__ import annotations

import sys

from amt.databricks.client import get_workspace_client
from amt.settings import get_settings

SOURCE = "pnr_prod.silver.pr_video_impressions"
CONTAINER = "671107897793"
CTV_CLIENTS = "'ROKU','FIRETV','TIZEN','LG','VIZIO','ANDROIDTV'"
SCHEMA_EXAMPLE_ACCOUNT = "af44641e-a0ed-4d88-92d0-6fa68e0cbf1a"
TRAIN_END, VAL_END, QA_END = "2026-08-30", "2026-09-06", "2026-09-13"
WINDOW_START = "2026-07-27"

ELIGIBLE = f"""
  parent_id = '{CONTAINER}'
  AND parent_type = 'container'
  AND client_name IN ({CTV_CLIENTS})
  AND item_index IS NOT NULL
  AND account_id <> '{SCHEMA_EXAMPLE_ACCOUNT}'
  AND event_received >= '{WINDOW_START}'
  AND event_received < '{QA_END}'
"""

SLOT = "try_cast(element_at(split(cast(item_index AS string), ','), -1) AS DOUBLE)"

QUERIES: dict[str, str] = {
    "window and volume": f"""
        SELECT count(*) AS impressions,
               count(DISTINCT session_id) AS sessions,
               count(DISTINCT account_id) AS accounts,
               count(DISTINCT entity_id) AS titles,
               min(event_received) AS first_seen,
               max(event_received) AS last_seen,
               round(100 * avg(CASE WHEN conversion_flag THEN 1 ELSE 0 END), 2) AS conv_pct
        FROM {SOURCE} WHERE {ELIGIBLE}
    """,
    "conversion by slot": f"""
        SELECT {SLOT} AS slot, count(*) AS impressions,
               round(100 * avg(CASE WHEN conversion_flag THEN 1 ELSE 0 END), 2) AS conv_pct
        FROM {SOURCE} WHERE {ELIGIBLE} GROUP BY 1 ORDER BY 1 LIMIT 8
    """,
    "conversion by client at slot 0": f"""
        SELECT client_name, count(*) AS impressions,
               round(100 * avg(CASE WHEN conversion_flag THEN 1 ELSE 0 END), 2) AS conv_pct
        FROM {SOURCE} WHERE {ELIGIBLE} AND {SLOT} = 0 GROUP BY 1 ORDER BY 2 DESC
    """,
    "renders per split, after the frozen sampling": f"""
        WITH dedup AS (
          SELECT concat_ws('|', session_id, page_id, parent_id,
                           cast(event_received AS string)) AS group_id,
                 session_id, entity_id, event_received, conversion_flag, {SLOT} AS slot,
                 row_number() OVER (
                   PARTITION BY session_id, page_id, parent_id, event_received, entity_id
                   ORDER BY {SLOT}
                 ) AS rn
          FROM {SOURCE} WHERE {ELIGIBLE}
        ),
        grp AS (
          SELECT group_id, min(session_id) AS session_id, min(event_received) AS ts,
                 count(*) AS candidates, sum(CASE WHEN conversion_flag THEN 1 ELSE 0 END) AS conversions
          FROM dedup WHERE rn = 1 GROUP BY group_id HAVING count(*) >= 2
        )
        SELECT CASE WHEN ts < '{TRAIN_END}' THEN 'train'
                    WHEN ts < '{VAL_END}' THEN 'validation'
                    ELSE 'qa' END AS split,
               count(*) AS all_renders,
               sum(CASE WHEN abs(hash(session_id)) % 10 = 0 THEN 1 ELSE 0 END) AS sampled_10pct,
               sum(CASE WHEN abs(hash(session_id)) % 5 = 0 THEN 1 ELSE 0 END) AS sampled_20pct,
               round(avg(candidates), 2) AS avg_candidates,
               round(100 * avg(CASE WHEN conversions > 0 THEN 1 ELSE 0 END), 1) AS pct_with_conversion
        FROM grp GROUP BY 1 ORDER BY 1
    """,
    "candidate features available on the impression row": f"""
        SELECT count(*) AS rows,
               round(100 * avg(CASE WHEN days_since_air_date IS NOT NULL THEN 1 ELSE 0 END), 1) AS pct_air_date,
               round(100 * avg(CASE WHEN duration_seconds IS NOT NULL THEN 1 ELSE 0 END), 1) AS pct_duration,
               round(100 * avg(CASE WHEN content_type IS NOT NULL THEN 1 ELSE 0 END), 1) AS pct_content_type,
               round(100 * avg(CASE WHEN size(genres_array) > 0 THEN 1 ELSE 0 END), 1) AS pct_genres,
               round(100 * avg(CASE WHEN network_name IS NOT NULL THEN 1 ELSE 0 END), 1) AS pct_network,
               round(100 * avg(CASE WHEN is_sportingevent IS NOT NULL THEN 1 ELSE 0 END), 1) AS pct_sporting
        FROM {SOURCE} WHERE {ELIGIBLE}
    """,
    "the dashboard's own daily series": """
        SELECT min(event_date) AS first_day, max(event_date) AS last_day,
               count(*) AS days, round(avg(container_ctr_pct), 3) AS avg_ctr_pct
        FROM product_api_dev.cursor_agent.hp_dash_home_container_daily
        WHERE container_id = '671107897793' AND cohort = 'all' AND page_id = '727846815041'
          AND event_date >= '2026-09-07' AND event_date < '2026-09-13'
    """,
}


def main(argv: list[str]) -> int:
    settings = get_settings()
    client = get_workspace_client(settings)
    failures = 0
    for name, sql in QUERIES.items():
        print(f"\n=== {name} ===")
        response = client.statement_execution.execute_statement(
            statement=sql, warehouse_id=settings.warehouse_id, wait_timeout="50s"
        )
        if response.status.error:
            print(f"  FAILED: {response.status.error.message}")
            failures += 1
            continue
        columns = [c.name for c in response.manifest.schema.columns]
        rows = (response.result.data_array if response.result else []) or []
        print("  " + " | ".join(columns))
        for row in rows:
            print("  " + " | ".join("" if v is None else str(v) for v in row))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
