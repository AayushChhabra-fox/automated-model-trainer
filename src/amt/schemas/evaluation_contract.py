"""Stage 2 — EvaluationContract (SPEC.md §Stage 2).

The contract is the experimental ground truth. Once approved at HC3 it is frozen and its
``contract_hash`` is asserted by every node and every Databricks job.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from amt.schemas.common import (
    AmtModel,
    CohortDef,
    MetricDef,
    SplitRef,
    is_harness_metric,
    is_secondary_only_metric,
    new_id,
    stable_hash,
    utcnow,
)
from amt.schemas.problem_spec import EvaluationMode

# Mirrors `amt.jobs.eval.harness.HARNESS_VERSION`; a unit test asserts they
# agree. The contracts must not import the job runtime.
HARNESS_VERSION = "1.1.0"

SplitStrategy = Literal["temporal", "leave_last_n"]

#: How a conversion's relevance is weighted.
#:
#: ``none`` -- every conversion counts 1. Correct for `next_item`, and rigged
#: for a container: on the proving rail slot 0 converts at 29-51 % and slots
#: 1-3 at 5/2/1 %, so a ranker that copies the logged order scores almost
#: perfectly and any change reads as a regression.
#:
#: ``pbm`` -- a position-based model. Materialization estimates the examination
#: propensity of each slot from the *train* split, and a conversion's gain
#: becomes ``1 / max(theta_slot, propensity_floor)``. This corrects for *where*
#: an item was shown, not for *which* items the incumbent chose to show; the
#: report says so.
RelevanceWeighting = Literal["none", "pbm"]

# A boundary is quoted into SQL as a literal and compared against the source's
# timestamp column, so it has to *be* the instant: a date, a date and time with
# an optional zone (Spark casts `2026-09-07 00:00:00 UTC` happily), or epoch
# units when the column is numeric. `TIMESTAMP '2026-09-07'` would end up
# quoted inside its own quotes and compared as text.
_INSTANT = re.compile(
    r"^\d{4}-\d{2}-\d{2}"
    r"(?:[ T]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
    r"(?:\s?(?:Z|[+-]\d{2}:?\d{2}|[A-Za-z][A-Za-z_+/-]*))?$"
    r"|^\d+$"
)


class TemporalRange(AmtModel):
    start: str
    end: str
    timezone: str = "UTC"


class ColumnMap(AmtModel):
    """Physical column names in ``DatasetDefinition.source_table``.

    The harness works on a ``(user, item, timestamp)`` interaction grain; this
    is how that grain is found in the source, so the names must be columns that
    exist rather than a description of them.
    """

    user: str = "user_id"
    item: str = "item_id"
    timestamp: str = "ts"


class DatasetDefinition(AmtModel):
    """What the campaign trains and evaluates on.

    ``source_table``, ``columns``, ``filters`` and ``min_interactions_per_user``
    are executed verbatim by :mod:`amt.jobs.materialize_splits`; the remaining
    fields are the prose a human approves at HC3.
    """

    source_table: str = Field(
        description=(
            "The single fully-qualified table or view the splits are built from. "
            "Must exist and be readable by the job; prepare a view upstream if the "
            "interaction grain has to be derived from several tables."
        )
    )
    columns: ColumnMap = Field(default_factory=ColumnMap)
    filters: list[str] = Field(
        default_factory=list,
        description="SQL boolean predicates over source_table, ANDed onto every read.",
    )
    min_interactions_per_user: int = Field(
        default=2, ge=1, description="Users below this interaction count are dropped."
    )
    source_tables: list[str] = Field(
        default_factory=list, description="Every upstream table the dataset derives from."
    )
    # `grouped_rerank` only: what makes one render one evaluation unit, and
    # where the incumbent's order is recorded.
    group_key: list[str] = Field(
        default_factory=list,
        description="Columns that together identify one render. Executed verbatim.",
    )
    position_column: str = Field(
        default="", description="Column holding the incumbent's slot for the row."
    )
    snapshot_id: str = Field(description="Immutable snapshot identifier.")
    dataset_version: str = Field(description="Delta version / materialization id.")
    population_eligibility: str = ""
    temporal_range: TemporalRange | None = None
    exclusions: list[str] = Field(default_factory=list)
    label_construction: str = ""

    @model_validator(mode="after")
    def _require_immutable_source_reference(self) -> DatasetDefinition:
        """A campaign whose source can move is not reproducible.

        ``vod-nextitem-001`` was approved with both of these empty, which left
        the comparison resting on the Delta version of the *derived* splits: it
        cannot say which state of ``source_table`` produced them. The contract's
        own ``label_construction`` text already demanded a reference here; this
        is that demand made enforceable at draft time rather than at read time.
        """
        missing = [
            name
            for name in ("snapshot_id", "dataset_version")
            if not str(getattr(self, name) or "").strip()
        ]
        if missing:
            raise ValueError(
                f"{', '.join(missing)} must name an immutable reference to "
                f"{self.source_table!r} (a Delta version, a snapshot id or a "
                "timestamp the table can be read `AS OF`); an empty reference "
                "makes the campaign unreproducible"
            )
        return self


class SplitPolicy(AmtModel):
    """How the splits are cut, and what each may be used for.

    ``strategy`` and its boundaries/counts are executed by
    :mod:`amt.jobs.materialize_splits`; the ``*_allowed`` flags are the policy
    the graph and ``SplitGuard`` enforce.
    """

    strategy: SplitStrategy = "temporal"
    # temporal: half-open windows (train <= train_end < validation <= ...).
    train_end: str = Field(default="", description="Upper bound of train, e.g. '2026-09-07'.")
    validation_end: str = Field(default="", description="Upper bound of validation.")
    qa_end: str = Field(default="", description="Upper bound of QA; defaults to max(ts).")
    # leave_last_n: per-user event counts instead of dates.
    n_validation: int = Field(default=1, ge=1)
    n_qa: int = Field(default=1, ge=1)
    rationale: str = Field(default="", description="Why this cut fits the problem.")
    validation_reuse_allowed: bool = True
    qa_use_allowed_before_freeze: bool = Field(default=False, frozen=True)

    @model_validator(mode="after")
    def _check_boundaries(self) -> SplitPolicy:
        if self.strategy == "temporal" and not (self.train_end and self.validation_end):
            raise ValueError(
                "a temporal split needs train_end and validation_end; "
                "materialization has no other way to cut the source"
            )
        for name in ("train_end", "validation_end", "qa_end"):
            value = getattr(self, name)
            if value and not _INSTANT.match(value):
                raise ValueError(
                    f"{name}={value!r} is not a timestamp materialization can compare "
                    "against: give the instant itself, e.g. '2026-09-07', "
                    "'2026-09-07 00:00:00' or '2026-09-07 00:00:00 UTC', with no SQL "
                    "around it"
                )
        return self


class ProductMetricSource(AmtModel):
    """An external daily series of the product number the campaign claims to move.

    Named by the contract so QA can test "our offline metric tracks theirs"
    against a number the campaign did not compute. For the proving rail this is
    the dashboard's own Container CTR, built from a *different* source table --
    which is what makes the check a real cross-source test rather than a
    tautology against our own label.
    """

    table: str = Field(description="Fully-qualified table holding the daily series.")
    date_column: str
    value_column: str
    filter_sql: str = Field(
        default="", description="Boolean predicate selecting this container / cohort."
    )
    description: str = ""


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
    # If the offline metric cannot be shown to track the product number for the
    # ranker already in production, it cannot certify a new one.
    min_proxy_correlation: float = Field(
        default=0.5,
        ge=-1.0,
        le=1.0,
        description="Minimum Spearman rho between the offline metric and the product metric.",
    )
    min_proxy_days: int = Field(
        default=7, ge=0, description="Overlapping daily observations the correlation needs."
    )
    notes: str = ""


class EvaluationContract(AmtModel):
    contract_id: str = Field(default_factory=lambda: new_id("contract"))
    spec_hash: str = Field(description="Hash of the ProblemSpec this contract was derived from.")
    dataset: DatasetDefinition
    train_split: SplitRef
    validation_split: SplitRef
    qa_split: SplitRef
    split_policy: SplitPolicy
    user_population: str = ""
    item_population: str = ""
    label_definition: str = ""
    candidate_generation_rules: str = ""
    cohorts: list[CohortDef] = Field(default_factory=list)
    metrics: list[MetricDef] = Field(default_factory=list)
    primary_metric: str
    evaluation_mode: EvaluationMode = "next_item"
    relevance_weighting: RelevanceWeighting = "none"
    product_metric_source: ProductMetricSource | None = None
    propensity_floor: float = Field(
        default=0.02,
        gt=0.0,
        le=1.0,
        description="Lower bound on an examination propensity; caps the weight at 1/floor.",
    )
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
        unknown = sorted(n for n in names | {self.primary_metric} if not is_harness_metric(n))
        if unknown:
            raise ValueError(
                f"the harness cannot compute {unknown}; metrics must be named "
                f"'{{recall|ndcg|hitrate|mrr|coverage}}@k', e.g. 'ndcg@10'"
            )
        if is_secondary_only_metric(self.primary_metric):
            raise ValueError(
                f"{self.primary_metric!r} is a pointwise metric and may only be a "
                "secondary: the primary metric has to be the one that reorders the "
                "container, and a better AUC over impressions does not"
            )
        pointwise = sorted(n for n in names if is_secondary_only_metric(n))
        if pointwise and self.evaluation_mode != "grouped_rerank":
            raise ValueError(
                f"{pointwise} need a label on every candidate, which only "
                "evaluation_mode 'grouped_rerank' provides"
            )
        return self

    @model_validator(mode="after")
    def _grouped_rerank_needs_a_group(self) -> EvaluationContract:
        """A grouped evaluation with no group key has nothing to group by.

        These two fields are executed verbatim by materialization, which runs
        on a bare cluster: a missing one is a job that dies after the wheel
        upload rather than a contract a human can fix at HC3.
        """
        if self.evaluation_mode != "grouped_rerank":
            if self.relevance_weighting != "none":
                raise ValueError(
                    "relevance_weighting is a property of a grouped evaluation; "
                    "there are no slots to de-bias when ranking a whole catalog"
                )
            return self
        missing = [
            name
            for name, value in (
                ("dataset.group_key", self.dataset.group_key),
                ("dataset.position_column", self.dataset.position_column),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                f"evaluation_mode 'grouped_rerank' requires {missing}: one render is "
                "one evaluation unit, and the incumbent's order is the baseline"
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
    # Examination propensities under `relevance_weighting: pbm`, keyed
    # ``"{client}|{slot}"``. Estimated from the train split only, so they are
    # part of the frozen ground truth rather than something QA can re-derive
    # from the split it is judging -- and printed in the report, because a
    # de-biasing nobody can inspect is a de-biasing nobody should trust.
    propensities: dict[str, float] = Field(default_factory=dict)
    materialized_at: datetime = Field(default_factory=utcnow)
    job_run_id: str | None = None

    def splits(self) -> dict[str, SplitRef]:
        return {"train": self.train, "validation": self.validation, "qa": self.qa}
