"""Cohort assignment and per-cohort aggregation.

Cohort definitions are part of the frozen EvaluationContract: they must be
identical for every model in a campaign, which is why the buckets live here
(in the shared harness package) rather than in any model or job module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "COHORT_DIMENSIONS",
    "DEFAULT_HISTORY_BUCKETS",
    "DEFAULT_RECENCY_BUCKETS",
    "aggregate_by_cohort",
    "assign_cohorts",
    "cohort_metrics",
    "history_length_cohort",
    "recency_cohort",
]

# (upper_bound_exclusive, label) -- the final entry is the open-ended bucket.
DEFAULT_HISTORY_BUCKETS: tuple[tuple[float, str], ...] = (
    (1, "cold_start"),  # 0 known interactions
    (5, "hist_1_4"),
    (20, "hist_5_19"),
    (float("inf"), "hist_20_plus"),
)

DEFAULT_RECENCY_BUCKETS: tuple[tuple[float, str], ...] = (
    (7, "recency_0_7d"),
    (30, "recency_8_30d"),
    (float("inf"), "recency_31d_plus"),
)

COHORT_DIMENSIONS: tuple[str, ...] = ("history_length", "recency")


def _bucket(value: float, buckets: Sequence[tuple[float, str]], unknown: str) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return unknown
    for upper, label in buckets:
        if value < upper:
            return label
    return buckets[-1][1]


def history_length_cohort(
    n_history: int,
    buckets: Sequence[tuple[float, str]] = DEFAULT_HISTORY_BUCKETS,
) -> str:
    """Bucket a user by how many interactions they have in the train split."""
    return _bucket(float(n_history), buckets, unknown="hist_unknown")


def recency_cohort(
    days_since_last: float | None,
    buckets: Sequence[tuple[float, str]] = DEFAULT_RECENCY_BUCKETS,
) -> str:
    """Bucket a user by days between their last train interaction and the boundary."""
    return _bucket(
        float("nan") if days_since_last is None else float(days_since_last),
        buckets,
        unknown="recency_unknown",
    )


def assign_cohorts(
    histories: Sequence[Sequence[int]],
    days_since_last: Sequence[float] | None = None,
    history_buckets: Sequence[tuple[float, str]] = DEFAULT_HISTORY_BUCKETS,
    recency_buckets: Sequence[tuple[float, str]] = DEFAULT_RECENCY_BUCKETS,
) -> dict[str, list[str]]:
    """Assign every user a label on each cohort dimension.

    ``cold_start`` falls out of the history-length dimension (zero known
    history), so it needs no separate pass.  Returns
    ``{dimension: [label per user]}``.
    """
    hist_labels = [
        history_length_cohort(len(h or ()), history_buckets) for h in histories
    ]
    if days_since_last is None:
        rec_labels = ["recency_unknown"] * len(histories)
    else:
        if len(days_since_last) != len(histories):
            raise ValueError("days_since_last must align with histories")
        rec_labels = [recency_cohort(d, recency_buckets) for d in days_since_last]
    return {"history_length": hist_labels, "recency": rec_labels}


def aggregate_by_cohort(
    per_user: Mapping[str, np.ndarray],
    labels: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Average each per-user metric array within each cohort label."""
    n = len(labels)
    for name, arr in per_user.items():
        if len(arr) != n:
            raise ValueError(
                f"metric '{name}' has {len(arr)} rows but {n} cohort labels"
            )

    label_arr = np.asarray(labels, dtype=object)
    out: dict[str, dict[str, float]] = {}
    for label in sorted(set(labels)):
        mask = label_arr == label
        bucket: dict[str, float] = {"n_users": float(int(mask.sum()))}
        for name, arr in per_user.items():
            values = np.asarray(arr, dtype=np.float64)[mask]
            if values.size == 0 or np.all(np.isnan(values)):
                bucket[name] = 0.0
            else:
                bucket[name] = float(np.nanmean(values))
        out[label] = bucket
    return out


def cohort_metrics(
    per_user: Mapping[str, np.ndarray],
    cohorts: Mapping[str, Sequence[str]] | Iterable[str],
) -> dict[str, dict[str, float]]:
    """Aggregate per-user metrics over one or many cohort dimensions.

    ``cohorts`` may be a single list of labels or a mapping of
    ``{dimension: labels}`` as produced by :func:`assign_cohorts`; in the
    mapping case labels are namespaced ``"{dimension}={label}"`` so cohorts
    from different dimensions never collide in MLflow metric keys.
    """
    if isinstance(cohorts, Mapping):
        out: dict[str, dict[str, float]] = {}
        for dim, labels in cohorts.items():
            for label, values in aggregate_by_cohort(per_user, list(labels)).items():
                out[f"{dim}={label}"] = values
        return out
    return aggregate_by_cohort(per_user, list(cohorts))
