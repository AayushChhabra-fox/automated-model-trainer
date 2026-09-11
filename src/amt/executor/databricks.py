"""``DatabricksExecutor`` -- the real compute backend (PLAN §5).

Every method follows the same five steps:

1. ensure a wheel built from the current source tree is on the UC Volume;
2. upload the contract / config / frozen-champion JSON payloads next to it
   (Volume paths, not inline CLI args -- contracts are far too big for a
   command line, and a Volume path is also an audit record);
3. submit a one-time ``python_wheel_task`` run with ``--role`` and, crucially,
   ``--allowed-splits`` -- the executor is where split scoping is *decided*
   and ``SplitGuard`` on the compute side is where it is *enforced*;
4. wait for a terminal state;
5. find the MLflow run the job created (by its ``amt.job_run_id`` tag),
   download ``result.json``, and convert it into the Pydantic schema the
   orchestrator speaks.

The ``amt.schemas`` imports are lazy so this module stays importable in a
job-side context.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from amt.databricks import jobs as jobs_mod
from amt.databricks import wheel as wheel_mod

__all__ = ["ROLE_SPLITS", "DatabricksExecutor"]

# The split grants for each role. This table *is* the scientific-integrity
# boundary: train roles can never be handed `qa`, and phase 1 gets `qa` alone
# so it cannot re-read the data the champion was tuned on.
ROLE_SPLITS: dict[str, tuple[str, ...]] = {
    "materialize": (),
    "baseline": ("train", "validation"),
    "train": ("train", "validation"),
    "qa_phase0": ("validation",),
    "qa_phase1": ("qa",),
}

RESULT_ARTIFACT = "result.json"


class DatabricksExecutor:
    """Submits AMT job runs to Databricks and returns Pydantic results."""

    def __init__(
        self,
        settings: Any | None = None,
        client: Any | None = None,
        repo_root: str | None = None,
        poll_seconds: float = jobs_mod.DEFAULT_POLL_SECONDS,
        timeout_seconds: float = jobs_mod.DEFAULT_TIMEOUT_SECONDS,
        campaign_id: str = "",
        mlflow_experiment_id: str = "",
    ) -> None:
        if settings is None:
            from amt.settings import get_settings

            settings = get_settings()
        self.settings = settings
        self._client = client
        self.repo_root = repo_root
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.campaign_id = campaign_id
        self.mlflow_experiment_id = mlflow_experiment_id
        self._wheel_uri: str | None = None

    # -- plumbing ---------------------------------------------------------

    @property
    def client(self) -> Any:
        if self._client is None:
            from amt.databricks.client import get_workspace_client

            self._client = get_workspace_client(self.settings)
        return self._client

    @property
    def volume_path(self) -> str:
        return self.settings.artifact_volume

    def ensure_wheel(self, force: bool = False) -> str:
        if self._wheel_uri is None or force:
            self._wheel_uri = wheel_mod.ensure_wheel(
                self.client, self.volume_path, root=self.repo_root, force=force
            )
        return self._wheel_uri

    def _upload_payload(self, job_run_id: str, name: str, payload: Any) -> str:
        blob = json.dumps(payload, default=str, sort_keys=True).encode("utf-8")
        remote = (
            f"{self.volume_path.rstrip('/')}/campaigns/"
            f"{self.campaign_id or 'adhoc'}/{job_run_id}/{name}.json"
        )
        return wheel_mod.upload_bytes(self.client, remote, blob)

    @staticmethod
    def _dump(model: Any) -> Any:
        """Pydantic model -> plain JSON-able dict (pass dicts through)."""
        if model is None:
            return None
        if isinstance(model, Mapping):
            return dict(model)
        if hasattr(model, "model_dump"):
            return json.loads(model.model_dump_json())
        return model

    def _frozen_payload(self, frozen: Any) -> dict:
        """Serialize the FrozenChampion in the shape ``amt.jobs.qa`` reads.

        The schema calls the artifact ``model_artifact_uri`` and its logged
        validation behaviour ``validation_metrics``; the job side (which must
        stay importable without ``amt.schemas``) reads ``model_uri`` and
        ``val_metrics``.  The aliases are added here, at the seam, so neither
        side has to know about the other's naming.
        """
        payload = self._dump(frozen) or {}
        if not isinstance(payload, dict):
            return payload
        payload.setdefault(
            "model_uri",
            str(
                payload.get("model_artifact_uri")
                or payload.get("checkpoint_uri")
                or ""
            ),
        )
        val_metrics = dict(payload.get("validation_metrics") or {})
        payload.setdefault("val_metrics", val_metrics)
        if "catalog_size" not in payload and "catalog_size" in val_metrics:
            payload["catalog_size"] = val_metrics["catalog_size"]
        return payload

    def _contract_payload(self, contract: Any) -> dict:
        """Serialize the contract *with* its hash, which jobs assert against."""
        payload = self._dump(contract) or {}
        if isinstance(payload, dict):
            payload.setdefault(
                "contract_hash", str(getattr(contract, "contract_hash", "") or "")
            )
            payload.setdefault(
                "harness_version", str(getattr(contract, "harness_version", "") or "")
            )
            payload.setdefault("spec_hash", str(getattr(contract, "spec_hash", "") or ""))
            payload.setdefault(
                "primary_metric", str(getattr(contract, "primary_metric", "") or "")
            )
        return payload

    # -- the one submission path ------------------------------------------

    def _submit(
        self,
        role: str,
        *,
        contract: Any,
        config: Any = None,
        frozen: Any = None,
        baseline_run_id: str | None = None,
        hypothesis_id: str = "",
        allowed_splits: Sequence[str] | None = None,
        extra_parameters: Sequence[str] = (),
    ) -> dict:
        from amt.executor.base import ExecutorError

        job_run_id = uuid.uuid4().hex
        splits = tuple(
            allowed_splits if allowed_splits is not None else ROLE_SPLITS[role]
        )

        wheel_uri = self.ensure_wheel()
        parameters: list[str] = [
            "--role",
            role,
            "--allowed-splits",
            ",".join(splits),
            "--job-run-id",
            job_run_id,
        ]
        parameters += [
            "--contract-json",
            self._upload_payload(job_run_id, "contract", self._contract_payload(contract)),
        ]
        if config is not None:
            parameters += [
                "--config-json",
                self._upload_payload(job_run_id, "config", self._dump(config)),
            ]
        if frozen is not None:
            parameters += [
                "--frozen-json",
                self._upload_payload(job_run_id, "frozen", self._frozen_payload(frozen)),
            ]
        if baseline_run_id:
            parameters += ["--baseline-run-id", str(baseline_run_id)]
        if self.mlflow_experiment_id:
            parameters += ["--mlflow-experiment-id", str(self.mlflow_experiment_id)]
        if self.campaign_id:
            parameters += ["--campaign-id", str(self.campaign_id)]
        if hypothesis_id:
            parameters += ["--hypothesis-id", str(hypothesis_id)]
        parameters += list(extra_parameters)

        started = time.time()
        try:
            run_id = jobs_mod.submit_run(
                self.client,
                run_name=f"amt-{role}-{job_run_id[:8]}",
                wheel_uri=wheel_uri,
                parameters=parameters,
                cluster=self.settings.cluster_spec(),
            )
            run = jobs_mod.wait_for_terminal(
                self.client,
                run_id,
                poll_seconds=self.poll_seconds,
                timeout_seconds=self.timeout_seconds,
            )
        except jobs_mod.JobRunError as exc:
            raise ExecutorError(str(exc)) from exc

        payload = self.fetch_result(job_run_id)
        payload.setdefault("runtime_seconds", time.time() - started)
        if self.mlflow_experiment_id:
            payload.setdefault("mlflow_experiment_id", str(self.mlflow_experiment_id))
        if hypothesis_id:
            payload.setdefault("hypothesis_id", str(hypothesis_id))
        payload["job_run_id"] = job_run_id
        payload["databricks_run_id"] = run_id
        payload["run_page_url"] = jobs_mod.run_page_url(run)
        return payload

    # -- reading the result back ------------------------------------------

    def find_mlflow_run(self, job_run_id: str) -> Any:
        """Locate the run the job created via its ``amt.job_run_id`` tag."""
        from mlflow.tracking import MlflowClient

        from amt.executor.base import ExecutorError

        client = MlflowClient()
        experiment_ids = (
            [str(self.mlflow_experiment_id)] if self.mlflow_experiment_id else None
        )
        if not experiment_ids:
            raise ExecutorError(
                "mlflow_experiment_id is required to locate the job's MLflow run"
            )
        runs = client.search_runs(
            experiment_ids=experiment_ids,
            # The tag key itself contains dots, so it must be quoted or the
            # MLflow search parser reads ``amt`` as the key and chokes.
            filter_string=f'tags."amt.job_run_id" = \'{job_run_id}\'',
            max_results=1,
        )
        if not runs:
            raise ExecutorError(
                f"no MLflow run tagged amt.job_run_id={job_run_id!r} in "
                f"experiment {experiment_ids[0]}"
            )
        return runs[0]

    def fetch_result(self, job_run_id: str) -> dict:
        """Download and parse the job's ``result.json`` artifact."""
        import mlflow

        from amt.executor.base import ExecutorError

        run = self.find_mlflow_run(job_run_id)
        run_id = run.info.run_id
        with tempfile.TemporaryDirectory() as tmp:
            local = mlflow.artifacts.download_artifacts(
                run_id=run_id, artifact_path=RESULT_ARTIFACT, dst_path=tmp
            )
            if os.path.isdir(local):
                local = os.path.join(local, RESULT_ARTIFACT)
            with open(local, encoding="utf-8") as fh:
                payload = json.load(fh)
        if not isinstance(payload, dict):
            raise ExecutorError(f"{RESULT_ARTIFACT} for run {run_id} is not an object")
        payload.setdefault("mlflow_run_id", run_id)
        return payload

    # -- Executor protocol -------------------------------------------------

    def materialize_splits(self, contract: Any) -> Any:
        payload = self._submit("materialize", contract=contract)
        return _to_split_materialization(payload, contract)

    def run_baseline(
        self,
        spec: Any,
        contract: Any,
        split: Literal["validation", "qa"],
    ) -> Any:
        if split != "validation":
            from amt.executor.base import ExecutorError

            # A baseline job is `amt.jobs.train` with --role baseline, and that
            # module refuses anything outside train,validation. QA's baseline
            # numbers come from phase 1, which loads this run's logged model on
            # the QA split -- refitting a baseline with `qa` granted would put
            # the QA population inside a training role.
            raise ExecutorError(
                "run_baseline(split='qa') is not supported: QA-split baseline "
                "metrics are produced by qa_phase1, which loads the baseline "
                "model logged by the validation baseline run."
            )
        payload = self._submit(
            "baseline",
            contract=contract,
            config=_baseline_config(spec),
            allowed_splits=ROLE_SPLITS["baseline"],
        )
        return _to_training_run_result(payload, role="baseline")

    def run_training(self, config: Any, contract: Any, plan: Any) -> Any:
        payload = self._submit(
            "train",
            contract=contract,
            config=config,
            hypothesis_id=str(
                getattr(plan, "hypothesis_id", "")
                or getattr(config, "hypothesis_id", "")
                or ""
            ),
        )
        return _to_training_run_result(payload, role="train")

    def run_qa_phase0(self, frozen: Any, contract: Any) -> Any:
        payload = self._submit("qa_phase0", contract=contract, frozen=frozen)
        return _to_phase0_result(payload)

    def run_qa_phase1(self, frozen: Any, baseline_ref: Any, contract: Any) -> Any:
        payload = self._submit(
            "qa_phase1",
            contract=contract,
            frozen=frozen,
            baseline_run_id=_baseline_run_id(baseline_ref),
        )
        return _to_phase1_result(payload)


# --------------------------------------------------------------------------
# result.json -> Pydantic
# --------------------------------------------------------------------------


def _baseline_run_id(baseline_ref: Any) -> str:
    if baseline_ref is None:
        return ""
    if isinstance(baseline_ref, str):
        return baseline_ref
    for attr in ("mlflow_run_id", "run_id"):
        value = getattr(baseline_ref, attr, None) or (
            baseline_ref.get(attr) if isinstance(baseline_ref, Mapping) else None
        )
        if value:
            return str(value)
    return ""


def _baseline_config(spec: Any) -> dict:
    """Config for the baseline run, taken from the ProblemSpec's baseline.

    Shaped as an ``ExperimentConfig`` so the run result round-trips back into
    the schema: the jobs side reads ``model.family``, the orchestrator side
    validates the whole dict.
    """
    baseline = getattr(spec, "baseline", None)
    # BaselineDefinition names the model family in ``kind``.
    kind = str(getattr(baseline, "kind", "") or "popularity")
    family = kind if kind in {"popularity", "sasrec"} else "popularity"
    return {
        "family": family,
        "model": {"family": family},
        "training": {},
        "loss": {},
        "features": {},
        "rationale": str(getattr(baseline, "description", "") or "")
        or f"Baseline reference: {getattr(baseline, 'name', family)}",
    }


def _keep(model_cls: Any, payload: Mapping[str, Any]) -> dict:
    """Drop keys the (concurrently evolving) schema does not declare."""
    fields = getattr(model_cls, "model_fields", {})
    return {k: v for k, v in payload.items() if k in fields}


def _to_training_run_result(payload: Mapping[str, Any], role: str = "train") -> Any:
    from amt.schemas.run_result import TrainingRunResult

    metrics = dict(payload.get("val_metrics") or {})
    config = payload.get("config") or {}
    mapped = {
        "run_id": payload.get("run_id") or payload.get("job_run_id") or "",
        "mlflow_run_id": payload.get("mlflow_run_id") or "",
        "mlflow_experiment_id": str(payload.get("mlflow_experiment_id") or ""),
        "job_run_id": payload.get("job_run_id"),
        "role": role,
        "hypothesis_id": str(
            payload.get("hypothesis_id") or config.get("hypothesis_id") or ""
        ),
        "config": payload.get("config"),
        "config_hash": payload.get("config_hash") or "",
        "family": str(
            (config.get("model") or {}).get("family") or config.get("family") or ""
        ),
        "train_metrics": dict(payload.get("train_metrics") or {}),
        "validation_metrics": metrics,
        "cohort_metrics": dict(payload.get("cohort_metrics") or {}),
        "runtime_seconds": float(payload.get("runtime_seconds") or 0.0),
        "model_artifact_uri": payload.get("model_uri") or None,
        "status": payload.get("status") or "success",
        "error": payload.get("error"),
        "contract_hash": payload.get("contract_hash") or "",
        "harness_version": payload.get("harness_version") or "",
    }
    return TrainingRunResult.model_validate(_keep(TrainingRunResult, mapped))


def _to_split_materialization(payload: Mapping[str, Any], contract: Any) -> Any:
    from amt.schemas.common import SplitRef
    from amt.schemas.evaluation_contract import SplitMaterialization

    splits = payload.get("splits") or {}
    refs = {
        name: SplitRef.model_validate({"name": name, **dict(ref)})
        for name, ref in splits.items()
    }
    mapped = {
        "contract_id": str(getattr(contract, "contract_id", "") or ""),
        "contract_hash": str(
            payload.get("contract_hash")
            or getattr(contract, "contract_hash", "")
            or ""
        ),
        "train": refs.get("train"),
        "validation": refs.get("validation"),
        "qa": refs.get("qa"),
        "job_run_id": payload.get("job_run_id"),
    }
    return SplitMaterialization.model_validate(_keep(SplitMaterialization, mapped))


def _to_phase0_result(payload: Mapping[str, Any]) -> Any:
    from amt.schemas.qa import IntegrityCheck, Phase0Result, QADecision

    checks = [
        IntegrityCheck.model_validate(
            {
                "name": c.get("name", ""),
                "passed": bool(c.get("passed")),
                "detail": str(c.get("detail", "")),
            }
        )
        for c in (payload.get("checks") or [])
    ]
    by_name = {c.name: c.passed for c in checks}
    passed = bool(payload.get("passed"))
    mapped = {
        "passed": passed,
        "checks": checks,
        "artifact_loaded": by_name.get("artifact_loads", False),
        "schema_compatible": by_name.get("signature_schema", False),
        "contract_hash_matches": by_name.get("contract_hash_matches", False),
        "harness_version_matches": by_name.get("harness_version_matches", False),
        "inference_executed": by_name.get("inference_executes", False),
        "reproduction_within_tolerance": by_name.get(
            "reproduces_logged_validation", False
        ),
        "reproduction_deltas": dict(payload.get("reproduction_deltas") or {}),
        "defects": [
            f"{c.name}: {c.detail}" for c in checks if not c.passed
        ],
        "decision": None if passed else QADecision.INVALID_EXPERIMENT,
        "mlflow_run_id": payload.get("mlflow_run_id") or "",
        "contract_hash": payload.get("contract_hash") or "",
        "harness_version": payload.get("harness_version") or "",
        "runtime_seconds": float(payload.get("runtime_seconds") or 0.0),
    }
    return Phase0Result.model_validate(_keep(Phase0Result, mapped))


def _to_phase1_result(payload: Mapping[str, Any]) -> Any:
    from amt.schemas.qa import PairedDelta, Phase1Result

    candidate = dict(payload.get("candidate_metrics") or {})
    baseline = dict(payload.get("baseline_metrics") or {})
    deltas = []
    for metric, stats in (payload.get("paired_deltas") or {}).items():
        base_value = float(baseline.get(metric, 0.0))
        cand_value = float(candidate.get(metric, 0.0))
        deltas.append(
            PairedDelta.model_validate(
                {
                    "metric": metric,
                    "baseline_value": base_value,
                    "candidate_value": cand_value,
                    "mean_delta": float(stats.get("mean", 0.0)),
                    "relative_delta": (
                        float(stats.get("mean", 0.0)) / base_value
                        if base_value
                        else 0.0
                    ),
                    "ci_low": float(stats.get("ci_low", 0.0)),
                    "ci_high": float(stats.get("ci_high", 0.0)),
                    "fraction_improved": float(stats.get("frac_improved", 0.0)),
                    "n": int(stats.get("n_users", 0)),
                }
            )
        )
    mapped = {
        "mlflow_run_id": payload.get("mlflow_run_id") or "",
        "qa_metrics": candidate,
        "baseline_metrics": baseline,
        "primary_metric": payload.get("primary_metric") or "",
        "paired_deltas": deltas,
        "cohort_metrics": dict(payload.get("cohort_metrics") or {}),
        "coverage": {
            k: float(v) for k, v in candidate.items() if k.startswith("coverage@")
        },
        "operational": {
            k: float(v) for k, v in (payload.get("operational") or {}).items()
        },
        "qa_population_size": int(payload.get("n_qa_users") or 0),
        "contract_hash": payload.get("contract_hash") or "",
        "harness_version": payload.get("harness_version") or "",
        "runtime_seconds": float(payload.get("runtime_seconds") or 0.0),
    }
    return Phase1Result.model_validate(_keep(Phase1Result, mapped))
