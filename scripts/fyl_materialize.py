"""Materialize the proving campaign's splits, and print what came out.

The grouped-rerank Spark branch has never run on real data, and it is the one
thing every later stage depends on. Running it alone first makes a failure
attributable to the runtime rather than to an agent's draft, and shows the
propensities before any model is fitted -- if slot 1 does not come out near
0.15 the position-bias correction is wrong and nothing downstream is worth
computing.

    uv run python scripts/fyl_materialize.py
"""

from __future__ import annotations

import sys
import time

# `scripts/` is not a package on the path when run as a file, so make it one.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

from fyl_contract import CAMPAIGN, contract

from amt.executor.databricks import DatabricksExecutor
from amt.settings import get_settings


def main(argv: list[str]) -> int:
    settings = get_settings()
    spec = settings.cluster_spec()
    if spec.get("existing_cluster_id"):
        print("refusing: AMT_EXISTING_CLUSTER_ID is set, so the run would carry no cost tags")
        return 2

    frozen = contract()
    executor = DatabricksExecutor(settings=settings, campaign_id=CAMPAIGN)
    print(f"campaign   {CAMPAIGN}")
    print(f"contract   {frozen.contract_hash[:16]}")
    print(f"mode       {frozen.evaluation_mode}, {frozen.relevance_weighting}-weighted")
    print(f"group key  {frozen.dataset.group_key}")
    print(f"cluster    {spec}")
    print(f"tags       {executor._cost_tags('materialize')}")

    started = time.time()
    result = executor.materialize_splits(frozen)
    print(f"\nelapsed    {time.time() - started:.0f}s")

    for name, ref in result.splits().items():
        print(f"{name:<11} {ref.table} v{ref.delta_version} rows={ref.row_count:,}")

    propensities = result.propensities
    print(f"\npropensities ({len(propensities)} client x slot)")
    for key, theta in sorted(propensities.items()):
        print(f"  {key:<16} theta={theta:.4f}  weight={1.0 / theta:>6.1f}x")
    if not propensities:
        print("  NONE -- the pbm branch did not estimate anything, which is a bug")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
