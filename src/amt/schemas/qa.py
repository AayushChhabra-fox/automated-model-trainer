"""Stage 9/10 — QA Phase 0 integrity, Phase 1 holdout evaluation, QA decision."""

from __future__ import annotations

from enum import Enum

from pydantic import Field

from amt.schemas.common import AmtModel


class QADecision(str, Enum):
    """Terminal QA outcomes.

    ``INVALID_EXPERIMENT`` may only be produced by Phase 0.
    """

    PASS = "PASS"
    FAIL_HYPOTHESIS = "FAIL_HYPOTHESIS"
    SPEC_REVIEW = "SPEC_REVIEW"
    INVALID_EXPERIMENT = "INVALID_EXPERIMENT"


class IntegrityCheck(AmtModel):
    name: str
    passed: bool
    detail: str = ""
    severity: str = Field(default="error", description="error | warning | info")


class Phase0Result(AmtModel):
    """Integrity & reproducibility. Runs before any QA-split performance is observed."""

    attempt: int = Field(default=1, ge=1)
    passed: bool = False
    checks: list[IntegrityCheck] = Field(default_factory=list)
    artifact_loaded: bool = False
    expected_model_version: str = ""
    schema_compatible: bool = False
    contract_hash_matches: bool = False
    harness_version_matches: bool = False
    inference_executed: bool = False
    reproduction_metrics: dict[str, float] = Field(
        default_factory=dict, description="Recomputed subset of logged validation behavior."
    )
    reproduction_deltas: dict[str, float] = Field(default_factory=dict)
    reproduction_tolerance: float = 1e-3
    reproduction_within_tolerance: bool = False
    defects: list[str] = Field(default_factory=list)
    fixable: bool = Field(
        default=False,
        description="True when the defect can be repaired without touching architecture, "
        "features, hyperparameters or model selection.",
    )
    fix_applied: str = ""
    decision: QADecision | None = Field(
        default=None, description="Only ever None or INVALID_EXPERIMENT."
    )
    mlflow_run_id: str = ""
    contract_hash: str = ""
    harness_version: str = ""
    runtime_seconds: float = 0.0

    @property
    def failed_checks(self) -> list[IntegrityCheck]:
        return [c for c in self.checks if not c.passed]


class PairedDelta(AmtModel):
    """Paired candidate-vs-baseline comparison with a bootstrap CI."""

    metric: str
    baseline_value: float
    candidate_value: float
    mean_delta: float
    relative_delta: float = 0.0
    ci_low: float = 0.0
    ci_high: float = 0.0
    ci_level: float = 0.95
    fraction_improved: float = 0.0
    n: int = 0
    n_bootstrap: int = 1000

    @property
    def significant(self) -> bool:
        """True when the CI excludes zero."""
        return (self.ci_low > 0.0) or (self.ci_high < 0.0)


class Phase1Result(AmtModel):
    """Holdout evaluation of the frozen champion and the baseline on the QA split."""

    mlflow_run_id: str = ""
    qa_metrics: dict[str, float] = Field(default_factory=dict)
    baseline_metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric: str = ""
    paired_deltas: list[PairedDelta] = Field(default_factory=list)
    cohort_metrics: dict[str, dict[str, float]] = Field(default_factory=dict)
    cohort_deltas: dict[str, PairedDelta] = Field(default_factory=dict)
    cold_start: PairedDelta | None = None
    coverage: dict[str, float] = Field(default_factory=dict)
    diversity: dict[str, float] = Field(default_factory=dict)
    calibration: dict[str, float] = Field(default_factory=dict)
    operational: dict[str, float] = Field(
        default_factory=dict,
        description="latency_ms_p50/p95, throughput_qps, model_size_mb, cost_usd_per_1k, ...",
    )
    # Read before any candidate number: if the incumbent's offline metric does
    # not track the product number day by day, nothing below can certify a
    # replacement, and the QA decision is SPEC_REVIEW rather than a verdict on
    # the model.
    proxy_valid: bool = True
    proxy_sanity: dict = Field(default_factory=dict)
    product_units: dict = Field(default_factory=dict)
    ab_design: dict[str, float] = Field(default_factory=dict)
    stability_notes: str = ""
    failure_cases: list[str] = Field(default_factory=list)
    qa_population_size: int = 0
    contract_hash: str = ""
    harness_version: str = ""
    runtime_seconds: float = 0.0

    def delta(self, metric: str) -> PairedDelta | None:
        return next((d for d in self.paired_deltas if d.metric == metric), None)

    @property
    def primary_delta(self) -> PairedDelta | None:
        return self.delta(self.primary_metric)


class Phase0Triage(AmtModel):
    """QA agent's reading of a failed Phase 0."""

    fixable: bool
    defects: list[str] = Field(default_factory=list)
    fix_plan: str = Field(
        default="",
        description="Repair must not modify architecture, features, hyperparameters or selection.",
    )
    rationale: str = ""


class QAAssessment(AmtModel):
    """Stage 10 decision. QA does not optimize; a poor result is evidence."""

    decision: QADecision
    rationale: str
    acceptance_criteria_met: dict[str, bool] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)
    spec_review_reasons: list[str] = Field(default_factory=list)
    generalization_gap_notes: str = ""
