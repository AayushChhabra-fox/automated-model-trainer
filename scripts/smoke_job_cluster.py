"""Prove a tagged job cluster can run the wheel, before a campaign depends on it.

Every AMT run so far went to a hand-prepared all-purpose cluster. This submits
the cheapest real role -- ``materialize`` -- on a job cluster with the B9 cost
tags, and prints the run URL, the resolved tags and the elapsed time. It writes
only to the configured scratch schema.

    uv run python scripts/smoke_job_cluster.py [source_table]
"""

from __future__ import annotations

import sys
import time

from amt.executor.databricks import DatabricksExecutor
from amt.schemas.common import MetricDef, SplitRef
from amt.schemas.evaluation_contract import (
    AcceptanceCriteria,
    ColumnMap,
    DatasetDefinition,
    EvaluationContract,
    SplitPolicy,
    TemporalRange,
)
from amt.settings import get_settings

CAMPAIGN = "smoke-job-cluster"
METRIC = "ndcg@10"


def _placeholder(name: str) -> SplitRef:
    return SplitRef(name=name, table="", delta_version=0, row_count=0, membership_hash="")


def contract(source_table: str) -> EvaluationContract:
    return EvaluationContract(
        spec_hash="smoke",
        dataset=DatasetDefinition(
            source_table=source_table,
            source_tables=[source_table],
            columns=ColumnMap(user="user_id", item="item_id", timestamp="ts"),
            filters=["event_type = 'start'"],
            min_interactions_per_user=3,
            snapshot_id="smoke-0",
            dataset_version="0",
            temporal_range=TemporalRange(start="2026-08-01", end="2026-09-05"),
            label_construction="synthetic start events",
        ),
        # Placeholders; materialization returns the real references.
        train_split=_placeholder("train"),
        validation_split=_placeholder("validation"),
        qa_split=_placeholder("qa"),
        split_policy=SplitPolicy(
            strategy="temporal",
            train_end="2026-08-25",
            validation_end="2026-08-30",
            qa_end="2026-09-05",
            rationale="smoke test",
        ),
        metrics=[MetricDef(name=METRIC, definition="NDCG at 10", k=10, is_primary=True)],
        primary_metric=METRIC,
        acceptance_criteria=AcceptanceCriteria(primary_metric=METRIC),
    )


def main(argv: list[str]) -> int:
    settings = get_settings()
    source = argv[0] if argv else settings.table("demo_interactions")
    executor = DatabricksExecutor(settings=settings, campaign_id=CAMPAIGN)

    spec = settings.cluster_spec()
    if spec.get("existing_cluster_id"):
        print(
            f"refusing: AMT_EXISTING_CLUSTER_ID={spec['existing_cluster_id']} is set, and "
            "an all-purpose cluster carries no job tags — this test would prove nothing"
        )
        return 2
    print(f"cluster   {spec}")
    print(f"tags      {executor._cost_tags('materialize')}")
    print(f"libraries {executor._libraries('lightgbm_reranker')} (for a booster run)")

    started = time.time()
    result = executor.materialize_splits(contract(source))
    print(f"elapsed   {time.time() - started:.0f}s")
    for name, ref in result.splits().items():
        print(f"{name:<11} {ref.table} v{ref.delta_version} rows={ref.row_count}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
