"""Uninstall every ``amt`` wheel from a cluster and restart it.

Wheels submitted to an existing cluster are installed *cluster-scoped*, so they
accumulate across runs.  Builds that share a version are indistinguishable to
pip, which keeps serving the first one installed; ``amt._version`` stamps new
builds so that cannot recur, but a cluster that already carries several
same-version installs has to be cleared once by hand.

    uv run python scripts/clear_cluster_wheels.py <cluster-id>
"""

from __future__ import annotations

import sys

from amt.databricks.client import get_workspace_client
from amt.settings import get_settings

PACKAGE = "amt-"


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(__doc__)
        return 2
    cluster_id = argv[0]
    client = get_workspace_client(get_settings())

    statuses = list(client.libraries.cluster_status(cluster_id=cluster_id))
    stale = [
        status.library
        for status in statuses
        if PACKAGE in str(getattr(status.library, "whl", "") or "")
    ]
    for status in statuses:
        print(f"  {status.status} {getattr(status.library, 'whl', status.library)}")

    if stale:
        client.libraries.uninstall(cluster_id=cluster_id, libraries=stale)
        print(f"uninstalled {len(stale)} wheel(s) -- takes effect on restart")

    print(f"restarting {cluster_id} ...", flush=True)
    client.clusters.restart(cluster_id=cluster_id).result()

    remaining = list(client.libraries.cluster_status(cluster_id=cluster_id))
    print(f"cluster running; {len(remaining)} library(ies) attached")
    for status in remaining:
        print(f"  {status.status} {getattr(status.library, 'whl', status.library)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
