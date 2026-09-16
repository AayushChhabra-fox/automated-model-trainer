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


class ProxySanity(AmtModel):
    """Whether the offline metric tracks the product number it claims to move.

    Measured on the *incumbent*: the model already in production, whose product
    number is already being watched. A metric that disagrees with the dashboard
    day by day for the live ranker cannot certify a replacement, however good
    the replacement's score looks.
    """

    passed: bool = True
    checked: bool = False
    rho: float | None = None
    rho_external: float | None = Field(
        default=None, description="Against the product metric's own daily series."
    )
    rho_own_label: float | None = Field(
        default=None, description="Against the split's own label; weaker evidence."
    )
    days: int = 0
    threshold: float = 0.5
    source: str = ""
    reason: str = ""
    offline_by_day: dict[str, float] = Field(default_factory=dict)
    product_by_day: dict[str, float] = Field(default_factory=dict)


class ProductUnits(AmtModel):
    """The effect in the unit a product owner already watches.

    Top-slot conversions per 1 000 renders, for the candidate and the
    incumbent. Under position-bias weighting the incumbent's figure collapses
    to its own observed rate, so it should match what the dashboard shows --
    and the candidate's is the counterfactual beside it.
    """

    unit: str = "top-slot conversions per 1,000 renders"
    candidate: float = 0.0
    candidate_ci: tuple[float, float] = (0.0, 0.0)
    baseline: float = 0.0
    baseline_ci: tuple[float, float] = (0.0, 0.0)
    n_renders: int = 0

    @property
    def delta(self) -> float:
        return self.candidate - self.baseline


class AbDesign(AmtModel):
    """What it would take to prove this offline effect online."""

    baseline_rate: float = 0.0
    candidate_rate: float = 0.0
    absolute_effect: float = 0.0
    relative_effect: float = 0.0
    mde: float = Field(default=0.0, description="Smallest effect one day of traffic resolves.")
    required_renders: float = 0.0
    days_at_current_traffic: float = 0.0
    power: float = 0.8
    significance: float = 0.05


class CohortRow(AmtModel):
    """One reported slice, candidate against incumbent, paired."""

    name: str
    metric: str = ""
    candidate: float = 0.0
    baseline: float = 0.0
    delta: float = 0.0
    relative_delta: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    n: int = 0
    regressed: bool = False


class ExperimentationReport(AmtModel):
    """What a product owner reads to decide, in product units.

    Extends the :class:`ScientificConclusion` rather than replacing it: the
    conclusion is the narrative and the scientific record, this is the
    evidence table behind it. The Planner writes the narrative and never the
    numbers -- every field here is computed by the QA job.
    """

    campaign_id: str = ""
    primary_metric: str = ""
    primary_delta: float = 0.0
    primary_ci: tuple[float, float] = (0.0, 0.0)
    primary_significant: bool = False
    candidate_metrics: dict[str, float] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    secondary_deltas: dict[str, float] = Field(default_factory=dict)
    product_units: ProductUnits = Field(default_factory=ProductUnits)
    cohort_table: list[CohortRow] = Field(default_factory=list)
    segment_table: list[CohortRow] = Field(default_factory=list)
    regressions: list[str] = Field(
        default_factory=list, description="Cohorts worse than the contract's guardrail."
    )
    proxy_sanity: ProxySanity = Field(default_factory=ProxySanity)
    propensities: dict[str, float] = Field(
        default_factory=dict,
        description="Examination propensity per client x slot, so the weighting is auditable.",
    )
    ab_design: AbDesign = Field(default_factory=AbDesign)
    operational: dict[str, float] = Field(default_factory=dict)
    assumptions: list[str] = Field(default_factory=list)
    evaluation_mode: str = "next_item"
    relevance_weighting: str = "none"
    qa_decision: QADecision | None = None
    contract_hash: str = ""
    created_at: datetime = Field(default_factory=utcnow)


#: The caveats every grouped-rerank report carries, whatever the numbers say.
#: Stated as data rather than prose so they cannot be edited out by an agent
#: writing a narrative it likes.
GROUPED_RERANK_ASSUMPTIONS: tuple[str, ...] = (
    "Replay on logged candidates only: the candidate set is the incumbent's, so this "
    "measures re-ordering, not better candidate generation.",
    "Position bias is corrected by a position-based model estimated from the incumbent's "
    "own per-slot rates, not from randomisation. It corrects for where an item was shown, "
    "not for which items the incumbent chose to show.",
    "The label is playback within 90 seconds; the dashboard counts clicks. Conversion is a "
    "stricter subset, which is what the proxy-sanity correlation tests.",
    "Offline improvement is not causal lift. Only an online experiment establishes that.",
)


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
