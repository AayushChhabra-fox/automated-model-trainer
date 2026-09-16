"""Download a job's ``result.json`` by its ``amt.job_run_id`` tag.

The artifact is the audit record: when the orchestrator's view of a run
disagrees with what the job actually reported, this is what settles it.

    uv run python scripts/fetch_result.py <job_run_id> [key ...]
"""

from __future__ import annotations

import json
import sys

from amt.executor.databricks import DatabricksExecutor
from amt.settings import get_settings


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    payload = DatabricksExecutor(settings=get_settings()).fetch_result(argv[0])
    keys = argv[1:] or sorted(payload)
    for key in keys:
        print(f"{key}: {json.dumps(payload.get(key), indent=2, default=str)[:2000]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
