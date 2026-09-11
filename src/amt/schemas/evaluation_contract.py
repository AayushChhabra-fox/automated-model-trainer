"""Stage 2 — EvaluationContract (SPEC.md §Stage 2).

The contract is the experimental ground truth. Once approved at HC3 it is frozen and its
``contract_hash`` is asserted by every node and every Databricks job.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import Field, model_validator

from amt.schemas.common import (
    AmtModel,
    CohortDef,
    MetricDef,
    SplitRef,
    new_id,
    stable_hash,
    utcnow,
)

HARNESS_VERSION = "1.0.0"


class TemporalRange(AmtModel):
    start: str
    end: str
    timezone: str = "UTC"


class DatasetDefinition(AmtModel):
    source_tables: list[str] = Field(default_factory=list)
    snapshot_id: str = Field(default="", description="Immutable snapshot identifier.")
    dataset_version: str = Field(default="", description="Delta version / materialization id.")
    population_eligibility: str = ""
    temporal_range: TemporalRange | None = None
    exclusions: list[str] = Field(default_factory=list)
    label_construction: str = ""


class SplitPolicy(AmtModel):
    """How each split may be used. Encoded so routing/jobs can enforce it."""

    strategy: str = Field(default="temporal", description="temporal | random_by_user | ...")
    boundaries: dict[str, str] = Field(default_factory=dict)
    validation_reuse_allowed: bool = True
    qa_use_allowed_before_freeze: bool = Field(default=False, frozen=True)


class AcceptanceCriteria(AmtModel):
    """Predefined, pre-registered criteria used by QA — never adjusted after seeing QA."""

    primary_metric: str
    min_absolute_improvement: float = 0.0
    min_relative_improvement: float = 0.0
    require_ci_excludes_zero: bool = True
    max_cohort_regression: float = Field(
        default=0.02, description="Max tolerated relative regression on any reported cohort."
    )
    max_latency_ms_p95: float | None = None
    notes: str = ""


class EvaluationContract(AmtModel):
    contract_id: str = Field(default_factory=lambda: new_id("contract"))
    spec_hash: str = Field(description="Hash of the ProblemSpec this contract was derived from.")
    dataset: DatasetDefinition
    train_split: SplitRef
    validation_split: SplitRef
    qa_split: SplitRef
    split_policy: SplitPolicy = Field(default_factory=SplitPolicy)
    user_population: str = ""
    item_population: str = ""
    label_definition: str = ""
    candidate_generation_rules: str = ""
    cohorts: list[CohortDef] = Field(default_factory=list)
    metrics: list[MetricDef] = Field(default_factory=list)
    primary_metric: str
    harness_version: str = HARNESS_VERSION
    acceptance_criteria: AcceptanceCriteria
    invariants: list[str] = Field(
        default_factory=lambda: [
            "Split membership and boundaries are fixed for the campaign.",
            "Labels, population eligibility and candidate construction are fixed.",
            "Metric and cohort definitions are fixed.",
            "The QA split is not read before champion freeze.",
        ]
    )
    created_at: datetime = Field(default_factory=utcnow)
    revision: int = 0

    @model_validator(mode="after")
    def _check_primary_metric(self) -> EvaluationContract:
        names = {m.name for m in self.metrics}
        if names and self.primary_metric not in names:
            raise ValueError(
                f"primary_metric {self.primary_metric!r} is not among metrics {sorted(names)}"
            )
        if self.acceptance_criteria.primary_metric != self.primary_metric:
            raise ValueError(
                "acceptance_criteria.primary_metric must equal contract.primary_metric"
            )
        return self

    @property
    def contract_hash(self) -> str:
        """Stable identity of the experimental ground truth."""
        return stable_hash(self, exclude={"created_at", "contract_id"})

    def split(self, name: str) -> SplitRef:
        return {
            "train": self.train_split,
            "validation": self.validation_split,
            "qa": self.qa_split,
        }[name]

    def metric(self, name: str) -> MetricDef | None:
        return next((m for m in self.metrics if m.name == name), None)


class SplitMaterialization(AmtModel):
    """Result of materializing the frozen splits as immutable Delta tables."""

    contract_id: str
    contract_hash: str
    train: SplitRef
    validation: SplitRef
    qa: SplitRef
    materialized_at: datetime = Field(default_factory=utcnow)
    job_run_id: str | None = None

    def splits(self) -> dict[str, SplitRef]:
        return {"train": self.train, "validation": self.validation, "qa": self.qa}
