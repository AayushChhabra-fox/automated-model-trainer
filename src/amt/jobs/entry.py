"""``amt-job`` console entry point -- the only thing Databricks invokes.

The wheel is submitted as a ``python_wheel_task`` with
``entry_point="amt-job"`` and a flat ``--flag value`` parameter list; this
module parses that, dispatches on ``--role``, and is responsible for the two
things every job must do regardless of role:

* stamp the PLAN §7 tags on the MLflow run, and
* log the role's result as the ``result.json`` artifact, which is what
  ``DatabricksExecutor`` reads back.

Hard constraint (CLAUDE.md): nothing reachable from here may import
LangGraph, LangChain or any LLM client.  MLflow, Spark and torch are all
imported lazily so the module also imports cleanly in a bare test env.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Mapping

__all__ = [
    "ROLES",
    "ROLE_TAGS",
    "RESULT_ARTIFACT",
    "build_parser",
    "load_json_arg",
    "stable_hash",
    "get_spark",
    "to_pandas",
    "run_tags",
    "dispatch",
    "main",
]

ROLES = ("materialize", "baseline", "train", "qa_phase0", "qa_phase1")

# PLAN §7 uses short role tags on the MLflow run.
ROLE_TAGS = {
    "materialize": "materialize",
    "baseline": "baseline",
    "train": "train",
    "qa_phase0": "qa0",
    "qa_phase1": "qa1",
}

RESULT_ARTIFACT = "result.json"


# --------------------------------------------------------------------------
# argument handling
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="amt-job", add_help=True)
    parser.add_argument("--role", required=True, choices=list(ROLES))
    parser.add_argument(
        "--allowed-splits",
        default="",
        help="comma separated subset of train,validation,qa this job may read",
    )
    parser.add_argument("--contract-json", default=None, help="path or JSON string")
    parser.add_argument("--config-json", default=None, help="path or JSON string")
    parser.add_argument("--frozen-json", default=None, help="path or JSON string")
    parser.add_argument("--baseline-run-id", default=None)
    parser.add_argument("--mlflow-experiment-id", default=None)
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--hypothesis-id", default=None)
    # Correlation id the executor stamps as amt.job_run_id so it can find the
    # MLflow run this job created.
    parser.add_argument("--job-run-id", default=None)
    parser.add_argument(
        "--local-dir",
        default=os.environ.get("AMT_LOCAL_DATA_DIR"),
        help="local data root for no-Spark dry runs",
    )
    return parser


def load_json_arg(value: Any) -> Any:
    """Accept either a path to a JSON file or an inline JSON string."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return None
    if os.path.exists(text):
        with open(text, encoding="utf-8") as fh:
            return json.load(fh)
    return json.loads(text)


def stable_hash(payload: Any) -> str:
    """sha256 of canonical JSON -- used for ``config_hash``."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# runtime helpers
# --------------------------------------------------------------------------


def get_spark() -> Any | None:
    """Active Spark session, or ``None`` when running off-cluster."""
    try:
        from pyspark.sql import SparkSession
    except Exception:
        return None
    try:
        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception:
        return None


def to_pandas(df: Any):
    """Collect a Spark DataFrame to pandas; pass pandas through untouched."""
    if hasattr(df, "toPandas"):
        return df.toPandas()
    return df


def _mlflow():
    try:
        import mlflow

        return mlflow
    except Exception:
        return None


def run_tags(args: argparse.Namespace, extra: Mapping[str, Any] | None = None) -> dict:
    """The PLAN §7 tag set for this job's MLflow run."""
    contract = load_json_arg(args.contract_json) or {}
    config = load_json_arg(args.config_json) or {}
    from amt.jobs.eval.harness import HARNESS_VERSION

    tags = {
        "amt.campaign_id": args.campaign_id or "",
        "amt.stage": args.role,
        "amt.role": ROLE_TAGS.get(args.role, args.role),
        "amt.hypothesis_id": args.hypothesis_id or "",
        "amt.contract_hash": str(contract.get("contract_hash") or ""),
        "amt.spec_hash": str(contract.get("spec_hash") or ""),
        "amt.harness_version": HARNESS_VERSION,
        "amt.config_hash": stable_hash(config) if config else "",
        "amt.code_version": os.environ.get("AMT_CODE_VERSION", ""),
        "amt.job_run_id": args.job_run_id
        or os.environ.get("DATABRICKS_JOB_RUN_ID", ""),
        "amt.allowed_splits": args.allowed_splits or "",
    }
    tags.update({k: str(v) for k, v in (extra or {}).items()})
    return {k: str(v) for k, v in tags.items()}


def log_result(result: Mapping[str, Any], mlflow: Any | None = None) -> None:
    """Write ``result.json`` as an artifact of the active MLflow run."""
    mlflow = mlflow or _mlflow()
    if mlflow is None or mlflow.active_run() is None:
        return
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, RESULT_ARTIFACT)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        mlflow.log_artifact(path)


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def _handlers() -> dict[str, Callable[[argparse.Namespace], dict]]:
    from amt.jobs import materialize_splits, qa, train

    return {
        "materialize": materialize_splits.run,
        "baseline": train.run,
        "train": train.run,
        "qa_phase0": qa.run_phase0,
        "qa_phase1": qa.run_phase1,
    }


def dispatch(args: argparse.Namespace) -> dict:
    """Run the role handler (no MLflow side effects)."""
    handler = _handlers()[args.role]
    return handler(args)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    mlflow = _mlflow()

    started = time.time()
    if mlflow is None:
        result = dispatch(args)
        print(json.dumps(result, default=str))
        return 0 if result.get("status") not in {"failed"} else 1

    if args.mlflow_experiment_id:
        mlflow.set_experiment(experiment_id=str(args.mlflow_experiment_id))

    with mlflow.start_run(run_name=f"{args.role}-{args.job_run_id or ''}".strip("-")):
        mlflow.set_tags(run_tags(args))
        try:
            result = dispatch(args)
        except Exception as exc:  # noqa: BLE001 - the job must report, not crash silently
            result = _failure_result(args, exc, time.time() - started)
            log_result(result, mlflow)
            mlflow.set_tag("amt.status", "failed")
            traceback.print_exc()
            return 1
        result.setdefault("status", "success")
        result.setdefault("runtime_seconds", time.time() - started)
        result.setdefault("mlflow_run_id", mlflow.active_run().info.run_id)
        log_result(result, mlflow)
        mlflow.set_tag("amt.status", str(result.get("status")))
        print(json.dumps({k: v for k, v in result.items() if k != "per_user"},
                         default=str))
    return 0


def _failure_result(
    args: argparse.Namespace, exc: BaseException, runtime: float
) -> dict:
    from amt.jobs.eval.harness import HARNESS_VERSION

    contract = load_json_arg(args.contract_json) or {}
    return {
        "run_id": args.job_run_id or "",
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
        "role": args.role,
        "runtime_seconds": runtime,
        "contract_hash": contract.get("contract_hash"),
        "harness_version": HARNESS_VERSION,
    }


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
