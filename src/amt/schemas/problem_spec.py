"""Stage 1 — ProblemSpec (SPEC.md §Stage 1)."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from amt.schemas.common import (
    AmtModel,
    MetricDef,
    is_harness_metric,
    new_id,
    stable_hash,
    utcnow,
)

Measurability = Literal["offline", "offline_proxy", "simulated", "online_only"]
BaselineKind = Literal["production_model", "existing_model", "heuristic", "popularity", "other"]


class ProblemDescription(AmtModel):
    product_behavior_to_improve: str
    decision: str = Field(description="The decision the model needs to make.")
    prediction_target: str = Field(description="Prediction or ranking target.")
    serving_context: str
    product_constraints: list[str] = Field(default_factory=list)


class DataSource(AmtModel):
    name: str
    table: str | None = None
    path: str | None = None
    relevant_columns: list[str] = Field(default_factory=list)
    grain: str = Field(default="", description="One row per ...")
    join_keys: list[str] = Field(default_factory=list)
    timestamp_columns: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    label_construction: str = ""
    notes: str = ""


class ProductMetric(AmtModel):
    name: str
    definition: str
    measurability: Measurability = Field(
        description="offline | offline_proxy | simulated | online_only"
    )
    notes: str = ""


class ObjectiveFunction(AmtModel):
    name: str = Field(description="e.g. 'BCE(next_item_prediction)' or 'pairwise_ranking_loss'.")
    description: str = ""
    relation_to_offline_metric: str = ""
    expected_product_outcome: str = ""


class BaselineDefinition(AmtModel):
    kind: BaselineKind = "popularity"
    name: str
    description: str = ""
    reference: str | None = Field(
        default=None, description="MLflow model URI / table / code reference for the baseline."
    )
    evaluated_with_same_contract: bool = Field(
        default=True,
        description="Must be True: the baseline is evaluated under the same EvaluationContract.",
    )


class ProbeObservation(AmtModel):
    """One read-only data investigation performed while drafting the spec."""

    action: str
    target: str
    summary: str
    rows: list[dict] = Field(default_factory=list)


class ProblemSpec(AmtModel):
    spec_id: str = Field(default_factory=lambda: new_id("spec"))
    problem_text: str = Field(default="", description="Verbatim human problem statement (HC1).")
    problem_description: ProblemDescription
    data_sources: list[DataSource] = Field(default_factory=list)
    product_metrics: list[ProductMetric] = Field(default_factory=list)
    offline_metrics: list[MetricDef] = Field(default_factory=list)
    primary_offline_metric: str
    objective: ObjectiveFunction
    baseline: BaselineDefinition
    investigation_notes: list[ProbeObservation] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utcnow)
    revision: int = 0

    @model_validator(mode="after")
    def _check_primary_offline_metric(self) -> ProblemSpec:
        # Product metrics stay in product language; the offline metric the
        # campaign optimizes has to be one the harness computes, or the
        # contract derived from this spec could never be evaluated.
        if not is_harness_metric(self.primary_offline_metric):
            raise ValueError(
                f"the harness cannot compute {self.primary_offline_metric!r}; "
                "primary_offline_metric must be '{recall|ndcg|hitrate|mrr|coverage}@k'"
            )
        return self

    @property
    def spec_hash(self) -> str:
        return stable_hash(self, exclude={"created_at", "spec_id", "investigation_notes"})
