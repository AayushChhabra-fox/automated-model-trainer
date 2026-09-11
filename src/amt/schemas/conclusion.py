"""Stage 11/12 — ScientificConclusion presented at HC4."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import Field

from amt.schemas.common import AmtModel, utcnow
from amt.schemas.qa import QADecision

Recommendation = Literal[
    "proceed_to_ab_test",
    "shadow_or_canary",
    "supported_needs_more_validation",
    "hypothesis_rejected",
    "modeling_not_the_bottleneck",
    "evaluation_contract_requires_revision",
]


class ProductInterpretation(AmtModel):
    """Keeps expected impact strictly separated from proven causal lift."""

    expected_offline_impact: str
    proven_causal_lift: str = Field(
        default="Not established. Offline improvement does not prove online causal lift.",
    )
    assumptions_and_risks: list[str] = Field(default_factory=list)


class ScientificConclusion(AmtModel):
    campaign_id: str
    problem: str
    hypothesis: str
    experiments_summary: list[str] = Field(
        default_factory=list, description="Only the experiments that carry evidence."
    )
    best_model: str = ""
    why_selected: str = ""
    validation_evidence: str = ""
    qa_evidence: str = ""
    cohort_effects: str = ""
    operational_considerations: str = ""
    product_interpretation: ProductInterpretation
    qa_decision: QADecision
    recommendation: Recommendation
    recommendation_rationale: str = ""
    negative_result: bool = False
    open_questions: list[str] = Field(default_factory=list)
    report_markdown: str = ""
    mlflow_experiment_id: str = ""
    mlflow_final_run_id: str = ""
    contract_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)
