"""Shared primitives: ids, stable hashing, metric/split/cohort definitions, budget."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SplitName = Literal["train", "validation", "qa"]
MetricDirection = Literal["maximize", "minimize"]

# The ranking metrics the evaluation harness computes, as ``family@k``.
# Duplicated here rather than imported from ``amt.jobs.eval.harness`` so the
# contracts stay free of the job runtime; ``tests/unit/test_harness.py``
# asserts the two lists agree.
HARNESS_METRIC_FAMILIES = ("recall", "ndcg", "hitrate", "mrr", "coverage")

# Pointwise classification metrics, computed only in ``grouped_rerank`` mode
# where every candidate carries a label. They are admissible as *secondaries*
# and never as the primary: a better AUC over impressions does not move a
# container, and pre-registering one as the thing to beat is how the previous
# campaign ended up optimising a number nobody asked for.
SECONDARY_METRIC_FAMILIES = ("auc", "pr_auc", "logloss", "ece")

_HARNESS_METRIC = re.compile(rf"^({'|'.join(HARNESS_METRIC_FAMILIES)})@([1-9][0-9]*)$")
_SECONDARY_METRIC = re.compile(rf"^({'|'.join(SECONDARY_METRIC_FAMILIES)})$")


def is_harness_metric(name: str, *, primary: bool = False) -> bool:
    """Whether the harness computes ``name`` (``ndcg@10`` yes, ``PR-AUC`` no).

    A contract may only pre-register metrics the harness produces: a primary
    metric nothing computes cannot select a champion, and QA would compare
    candidates on a number that never exists. With ``primary=True`` the
    pointwise families are rejected too, because the primary metric has to be
    the one that moves the product.
    """
    text = str(name).strip()
    if _HARNESS_METRIC.match(text):
        return True
    return not primary and bool(_SECONDARY_METRIC.match(text.lower()))


def is_secondary_only_metric(name: str) -> bool:
    """True for a metric the harness computes but refuses as the primary."""
    return bool(_SECONDARY_METRIC.match(str(name).strip().lower()))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Short, readable, unique identifier: ``<prefix>-<8 hex>``."""
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _canonical(obj: Any) -> Any:
    if isinstance(obj, BaseModel):
        return _canonical(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, float) and obj.is_integer():
        return float(obj)
    return obj


def stable_hash(model: Any, *, exclude: set[str] | None = None) -> str:
    """Deterministic sha256 of a Pydantic model (or any JSON-able object).

    Key order, whitespace and Python object identity do not affect the result, so the
    same logical contract always hashes to the same value across processes and runs.
    """
    payload = model
    if isinstance(model, BaseModel) and exclude:
        payload = model.model_dump(mode="json", exclude=exclude)
    canonical = _canonical(payload)
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AmtModel(BaseModel):
    """Base for every cross-agent contract."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class MetricDef(AmtModel):
    """A metric with a precise, implementation-pinned definition."""

    name: str = Field(description="Metric name, e.g. 'ndcg@10'.")
    definition: str = Field(description="Precise mathematical / operational definition.")
    direction: MetricDirection = "maximize"
    k: int | None = Field(default=None, description="Cutoff for @K metrics.")
    implementation: str = Field(
        default="", description="Harness function reference, e.g. 'harness:ndcg_at_k'."
    )
    unit: str | None = None
    is_primary: bool = False
    tolerance: float | None = Field(
        default=None, description="Absolute tolerance used for reproduction checks."
    )


class CohortDef(AmtModel):
    """A slice of the evaluation population, fixed by the EvaluationContract."""

    name: str
    description: str = ""
    expression: str = Field(
        default="", description="SQL boolean expression over the evaluation population."
    )
    min_support: int = Field(default=0, description="Minimum rows for the cohort to be reported.")


class SplitRef(AmtModel):
    """Immutable reference to a materialized split."""

    name: SplitName
    table: str = Field(description="Fully-qualified Delta table holding the split membership.")
    delta_version: int = Field(description="Delta version read with `VERSION AS OF`.")
    row_count: int = Field(ge=0)
    membership_hash: str = Field(description="Hash of the sorted membership keys.")
    description: str = ""
    temporal_range: tuple[str, str] | None = None


class ExperimentBudget(AmtModel):
    """Bounds the autonomous region between HC3 and HC4."""

    max_iterations: int = Field(default=8, ge=1, description="Trainer iterations per hypothesis.")
    iterations_used: int = Field(default=0, ge=0)
    max_hypotheses: int = Field(default=3, ge=1)
    hypotheses_used: int = Field(default=0, ge=0)
    max_compute_minutes: float = Field(default=240.0, ge=0)
    compute_minutes_used: float = Field(default=0.0, ge=0)

    @property
    def iterations_remaining(self) -> int:
        return max(0, self.max_iterations - self.iterations_used)

    @property
    def hypotheses_remaining(self) -> int:
        return max(0, self.max_hypotheses - self.hypotheses_used)

    @property
    def compute_exhausted(self) -> bool:
        return self.compute_minutes_used >= self.max_compute_minutes

    @property
    def exhausted(self) -> bool:
        return self.iterations_remaining <= 0 or self.compute_exhausted

    def record_iteration(self, compute_minutes: float = 0.0) -> ExperimentBudget:
        return self.model_copy(
            update={
                "iterations_used": self.iterations_used + 1,
                "compute_minutes_used": self.compute_minutes_used + compute_minutes,
            }
        )

    def record_hypothesis(self) -> ExperimentBudget:
        """Start a new hypothesis: consumes one slot and resets the iteration counter."""
        return self.model_copy(
            update={"hypotheses_used": self.hypotheses_used + 1, "iterations_used": 0}
        )
