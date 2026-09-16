"""Score the incumbent on the proving campaign's validation split.

This is the number every later claim is measured against: the logged order of
the rail, scored through the same harness, under the same contract, with the
same position-bias weighting the candidates will get. Run alone before any
model is fitted, because if the incumbent's own score is implausible the
evaluation is wrong and nothing a booster produces will mean anything.

    uv run python scripts/fyl_baseline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from amt.executor.databricks import DatabricksExecutor  # noqa: E402
from amt.schemas.problem_spec import BaselineDefinition  # noqa: E402
from amt.settings import get_settings  # noqa: E402
from fyl_contract import CAMPAIGN, contract  # noqa: E402


class _SpecStub:
    """Only what `run_baseline` reads: which family the baseline is."""

    baseline = BaselineDefinition(
        kind="logged_order",
        name="the incumbent's own order",
        description="Rank each render's candidates by the slot the rail showed them in.",
    )


def materialized_contract(executor: DatabricksExecutor, job_run_id: str):
    """The contract with the real split references folded in.

    `materialize_splits_node` does this in a campaign. A standalone script has
    to do it too, or every job refuses with "split 'train' has no table": the
    placeholder refs a human approves at HC3 carry no table until
    materialization returns one.
    """
    from amt.executor.databricks import _to_split_materialization

    frozen = contract()
    materialization = _to_split_materialization(executor.fetch_result(job_run_id), frozen)
    return frozen.model_copy(
        update={
            "train_split": materialization.train,
            "validation_split": materialization.validation,
            "qa_split": materialization.qa,
        }
    ), materialization


#: The materialize run whose splits every later stage reads. Pinned rather than
#: re-derived so a rerun of this script cannot quietly score a different
#: population than the one already frozen.
MATERIALIZE_JOB_RUN_ID = "f344b2bcc2d642d58f959d28299a1b0c"


def main(argv: list[str]) -> int:
    settings = get_settings()
    executor = DatabricksExecutor(settings=settings, campaign_id=CAMPAIGN)
    frozen, materialization = materialized_contract(
        executor, argv[0] if argv else MATERIALIZE_JOB_RUN_ID
    )
    print(f"splits    {frozen.validation_split.table} v{frozen.validation_split.delta_version}")
    print(
        f"          {frozen.validation_split.row_count:,} renders, "
        f"{frozen.validation_split.positive_labels:,} with a conversion"
    )
    print(f"theta     {len(materialization.propensities)} client x slot")
    print(f"contract  {frozen.contract_hash[:16]}")
    print(f"metrics   {[m.name for m in frozen.metrics]}")
    print(f"weighting {frozen.relevance_weighting}")
    print(f"baseline  {_SpecStub.baseline.kind}")

    started = time.time()
    result = executor.run_baseline(_SpecStub(), frozen, "validation")
    print(f"elapsed   {time.time() - started:.0f}s\n")

    print(f"run       {result.mlflow_run_id}")
    print(f"family    {result.family}")
    print(f"status    {result.status}{'  ' + (result.error or '') if result.error else ''}")
    print("\nthe incumbent, on 498k validation renders:")
    for name, value in sorted(result.validation_metrics.items()):
        print(f"  {name:<28} {value:.5f}")
    if result.cohort_metrics:
        print("\nby cohort:")
        for cohort, block in sorted(result.cohort_metrics.items()):
            inline = "  ".join(f"{k}={v:.4f}" for k, v in sorted(block.items()))
            print(f"  {cohort:<22} {inline}")
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
