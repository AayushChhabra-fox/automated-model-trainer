"""Stage 7/8 — HypothesisChampion selection and the immutable FrozenChampion."""

from __future__ import annotations

from datetime import datetime

from pydantic import ConfigDict, Field

from amt.schemas.common import AmtModel, new_id, stable_hash, utcnow
from amt.schemas.experiment_config import ExperimentConfig


class HypothesisChampion(AmtModel):
    """Exactly one best configuration for a single hypothesis (Stage 7a)."""

    hypothesis_id: str
    run_id: str
    mlflow_run_id: str
    config: ExperimentConfig
    config_hash: str = ""
    primary_metric: str
    primary_metric_value: float
    secondary_metrics: dict[str, float] = Field(default_factory=dict)
    cohort_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    cohort_regressions: list[str] = Field(default_factory=list)
    guardrails_passed: bool = True
    complexity_notes: str = ""
    operational_notes: str = ""
    selection_rationale: str = ""
    model_artifact_uri: str | None = None


class FinalChampionSelection(AmtModel):
    """Planner output when comparing hypothesis champions (Stage 7b)."""

    hypothesis_id: str
    rationale: str
    rejected: dict[str, str] = Field(
        default_factory=dict, description="hypothesis_id -> why it was not selected."
    )


class FrozenChampion(AmtModel):
    """The single artifact QA evaluates. Immutable by construction (Stage 8)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    champion_id: str = Field(default_factory=lambda: new_id("champ"))
    hypothesis_id: str
    run_id: str
    mlflow_run_id: str
    config: ExperimentConfig
    config_hash: str
    family: str = ""
    architecture: dict = Field(default_factory=dict)
    features: dict = Field(default_factory=dict)
    hyperparameters: dict = Field(default_factory=dict)
    objective: str = ""
    preprocessing_ref: str = ""
    checkpoint_uri: str | None = None
    model_artifact_uri: str | None = None
    inference_implementation: str = Field(
        default="mlflow.pyfunc", description="How QA loads and runs the artifact."
    )
    validation_metrics: dict[str, float] = Field(default_factory=dict)
    cohort_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    contract_hash: str
    spec_hash: str = ""
    harness_version: str = ""
    frozen_at: datetime = Field(default_factory=utcnow)

    @property
    def freeze_hash(self) -> str:
        return stable_hash(self, exclude={"frozen_at", "champion_id"})
