"""Stage 5/6 — TrainingRunResult and the compact RunSummary kept in campaign state."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from amt.schemas.common import AmtModel, new_id
from amt.schemas.experiment_config import ExperimentConfig

RunStatus = Literal["success", "failed"]
Verdict = Literal["improved", "no_change", "regressed", "failed"]
RunRole = Literal["baseline", "train", "qa0", "qa1", "final", "materialize"]


def metric_value(metrics: dict[str, float], metric: str) -> float | None:
    """Look up ``metric`` in ``metrics``, tolerating the split prefix.

    The harness emits bare metric names; ``train.py`` logs them under ``val/``
    and MLflow keeps that prefix, so the same number reaches campaign state
    spelled two ways. Anything comparing a contract's ``primary_metric``
    against a stored metrics dict has to go through here: a raw ``.get`` looks
    like it works, silently returns ``None`` for the prefixed half, and turns
    "the candidate won" into "no comparison was produced".
    """
    for key in (metric, f"val/{metric}", f"validation/{metric}"):
        if key in metrics:
            return metrics[key]
    return None


class TrainingRunResult(AmtModel):
    """Compact result of one Databricks training (or baseline) run.

    Detailed curves, artifacts and params live in MLflow; this is the orchestration view.
    """

    run_id: str = Field(default_factory=lambda: new_id("run"), description="AMT run id.")
    mlflow_run_id: str
    mlflow_experiment_id: str = ""
    job_run_id: str | None = None
    role: RunRole = "train"
    hypothesis_id: str = ""
    config: ExperimentConfig | None = None
    config_hash: str = ""
    family: str = ""
    train_metrics: dict[str, float] = Field(default_factory=dict)
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    cohort_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    training_curve: list[dict] = Field(default_factory=list)
    runtime_seconds: float = 0.0
    model_artifact_uri: str | None = None
    status: RunStatus = "success"
    error: str | None = None
    contract_hash: str = ""
    harness_version: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    def primary(self, metric: str) -> float | None:
        """Validation value of ``metric`` accepting bare or ``val/``-prefixed keys."""
        return metric_value(self.validation_metrics, metric)


class RunSummary(AmtModel):
    """One line of experiment history — intentionally tiny."""

    run_id: str
    mlflow_run_id: str
    hypothesis_id: str = ""
    config_hash: str = ""
    # The approved shape this run was fitted under. Two runs sharing it differ
    # only in hyperparameters, which is what makes a bit-identical retune
    # recognisable as a no-op instead of another iteration's worth of budget.
    feature_fit_hash: str = ""
    iteration: int = 0
    primary_metric: str = ""
    primary_metric_value: float | None = None
    verdict: Verdict = "no_change"
    runtime_seconds: float = 0.0
    notes: str = ""
