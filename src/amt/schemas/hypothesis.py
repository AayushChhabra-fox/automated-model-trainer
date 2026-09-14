"""Stage 3 — Hypothesis and ExperimentPlan (SPEC.md §Stage 3)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from amt.schemas.common import AmtModel, new_id
from amt.schemas.feature_recipe import FeatureRecipe, FitSpec

HypothesisStatus = Literal["pending", "active", "converged", "abandoned", "falsified"]


class Hypothesis(AmtModel):
    """A falsifiable modeling hypothesis. Must answer all six SPEC questions."""

    hypothesis_id: str = Field(default_factory=lambda: new_id("hyp"))
    title: str
    # (1) what change is proposed
    proposed_change: str
    # (2) why it should improve the objective
    why_it_should_improve: str
    # (3) what evidence or intuition motivates it
    motivating_evidence: str
    # (4) what experiment tests it
    experiment_design: str
    # (5) which metric should improve
    metric_expected_to_improve: str
    # (6) what result would weaken or falsify it
    falsification_criteria: str

    expected_effect_size: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    estimated_cost_minutes: float = Field(default=30.0, ge=0.0)
    priority: int = Field(default=1, ge=1, description="1 = highest expected value.")
    status: HypothesisStatus = "pending"


class HypothesisBacklog(AmtModel):
    """Planner output: a prioritized set of hypotheses."""

    hypotheses: list[Hypothesis] = Field(default_factory=list)
    prioritization_rationale: str = ""


class ExperimentPlan(AmtModel):
    """Bounds the Trainer. Contains only what the Trainer needs to stay in-scope."""

    plan_id: str = Field(default_factory=lambda: new_id("plan"))
    hypothesis_id: str
    hypothesis_summary: str
    rationale: str
    baseline_reference: str = Field(
        default="", description="MLflow run id / name of the baseline it is compared against."
    )
    expected_effect: str = ""
    primary_metric: str
    secondary_metrics: list[str] = Field(default_factory=list)
    permitted_model_changes: list[str] = Field(default_factory=list)
    permitted_feature_changes: list[str] = Field(default_factory=list)
    search_space: dict = Field(default_factory=dict)
    family: str = Field(
        default="",
        description="Model family this plan targets. Tabular families need a FeatureRecipe.",
    )
    # Filled at the feature/fit human checkpoint, not by the Planner's plan call.
    feature_recipe: FeatureRecipe | None = None
    fit_spec: FitSpec | None = None
    feature_fit_hash: str = ""
    success_criteria: str = ""
    falsification_criteria: str = ""
    max_iterations: int = Field(default=5, ge=1)
    compute_budget_minutes: float = Field(default=60.0, ge=0.0)
    contract_hash: str = ""
