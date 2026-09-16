"""Second Phase 8 probe: the split boundaries, the dashboard series, the gaps.

The first probe (``probe_fyl.py``) confirmed the container, the label and the
slot mechanics. This one settles the three things that decide the contract:
which calendar days the boundaries land on, whether the product metric's own
daily series covers the QA week (the proxy check needs seven overlapping
days), and which of the candidate features are actually populated.

    uv run python scripts/probe_fyl_boundaries.py
"""

from __future__ import annotations

import sys

from amt.databricks.client import get_workspace_client
from amt.settings import get_settings

QUERIES = {
    "dayofweek of the candidate boundaries": """
  SELECT d, date_format(d, 'EEEE') AS dow FROM (
    SELECT explode(array(date'2026-07-27', date'2026-08-29', date'2026-08-30',
                         date'2026-09-05', date'2026-09-06', date'2026-09-12')) AS d)
""",
    "dashboard series over a shifted QA week (09-06..09-12)": """
  SELECT count(*) AS days, min(event_date) AS first_day, max(event_date) AS last_day,
         round(min(container_ctr_pct),2) AS min_ctr, round(max(container_ctr_pct),2) AS max_ctr
  FROM product_api_dev.cursor_agent.hp_dash_home_container_daily
  WHERE container_id='671107897793' AND cohort='all' AND page_id='727846815041'
    AND event_date >= '2026-09-06' AND event_date <= '2026-09-12'
""",
    "dashboard daily values, so the correlation has something to move on": """
  SELECT event_date, round(container_ctr_pct,3) AS ctr_pct
  FROM product_api_dev.cursor_agent.hp_dash_home_container_daily
  WHERE container_id='671107897793' AND cohort='all' AND page_id='727846815041'
    AND event_date >= '2026-09-06' AND event_date <= '2026-09-12' ORDER BY 1
""",
    "is days_since_air_date really empty, or just for this container": """
  SELECT count(*) AS rows,
         sum(CASE WHEN days_since_air_date IS NOT NULL THEN 1 ELSE 0 END) AS non_null,
         sum(CASE WHEN original_air_date IS NOT NULL THEN 1 ELSE 0 END) AS air_date_non_null,
         sum(CASE WHEN release_year IS NOT NULL THEN 1 ELSE 0 END) AS release_year_non_null
  FROM pnr_prod.silver.pr_video_impressions
  WHERE event_received >= '2026-09-01' AND event_received < '2026-09-02'
""",
    "renders per split under the shifted boundaries": """
  WITH dedup AS (
    SELECT concat_ws('|', session_id, page_id, parent_id, cast(event_received AS string)) AS gid,
           session_id, entity_id, event_received, conversion_flag,
           row_number() OVER (PARTITION BY session_id, page_id, parent_id, event_received, entity_id
                              ORDER BY try_cast(element_at(split(cast(item_index AS string), ','), -1) AS DOUBLE)) AS rn
    FROM pnr_prod.silver.pr_video_impressions
    WHERE parent_id='671107897793' AND parent_type='container'
      AND client_name IN ('ROKU','FIRETV','TIZEN','LG','VIZIO','ANDROIDTV')
      AND item_index IS NOT NULL AND account_id <> 'af44641e-a0ed-4d88-92d0-6fa68e0cbf1a'
      AND event_received >= '2026-07-27' AND event_received < '2026-09-13'),
  grp AS (SELECT gid, min(session_id) AS sid, min(event_received) AS ts, count(*) AS cands,
                 sum(CASE WHEN conversion_flag THEN 1 ELSE 0 END) AS conv
          FROM dedup WHERE rn=1 GROUP BY gid HAVING count(*) >= 2)
  SELECT CASE WHEN ts < '2026-08-30' THEN 'train'
              WHEN ts < '2026-09-06' THEN 'validation' ELSE 'qa' END AS split,
         count(DISTINCT date(ts)) AS days,
         sum(CASE WHEN abs(hash(sid)) % 10 = 0 THEN 1 ELSE 0 END) AS train_sample_10pct,
         sum(CASE WHEN abs(hash(sid)) % 5 = 0 THEN 1 ELSE 0 END) AS eval_sample_20pct,
         sum(CASE WHEN abs(hash(sid)) % 10 = 0 AND conv > 0 THEN 1 ELSE 0 END) AS train_labelled,
         sum(CASE WHEN abs(hash(sid)) % 5 = 0 AND conv > 0 THEN 1 ELSE 0 END) AS eval_labelled
  FROM grp GROUP BY 1 ORDER BY 1
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
        print("  " + " | ".join(c.name for c in response.manifest.schema.columns))
        for row in (response.result.data_array if response.result else []) or []:
            print("  " + " | ".join("" if v is None else str(v) for v in row))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
