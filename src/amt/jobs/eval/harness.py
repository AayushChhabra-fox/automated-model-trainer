"""The single evaluation harness for the whole campaign.

Every metric used anywhere in AMT (trainer validation, baseline, QA phase 1)
is computed by the functions in this module.  The version string is recorded
in the :class:`EvaluationContract` and asserted by every job, so a change to
the math invalidates in-flight campaigns instead of silently changing the
ground truth (SPEC "Evaluation Metrics" / "Invariance").

Everything here is pure ``numpy``: no Spark, no torch, no MLflow.

Conventions
-----------
``ranked``  : ``list[list[int]]`` -- per user, item ids ordered best-first.
``truth``   : ``list[set[int]]``  -- per user, the relevant (held-out) items.
Users whose truth set is empty are excluded from the averages but still
appear (as ``nan``) in the per-user arrays so that arrays stay aligned with
the input order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

HARNESS_VERSION = "1.0.0"

DEFAULT_KS: tuple[int, ...] = (5, 10, 20)

__all__ = [
    "DEFAULT_KS",
    "HARNESS_VERSION",
    "evaluate",
    "hitrate_at_k",
    "item_coverage",
    "metric_names",
    "mrr",
    "ndcg_at_k",
    "per_user_metrics",
    "recall_at_k",
]


# --------------------------------------------------------------------------
# internal helpers
# --------------------------------------------------------------------------


def _as_hit_matrix(
    ranked: Sequence[Sequence[int]],
    truth: Sequence[Iterable[int]],
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(hits, n_truth, valid)``.

    ``hits`` is a ``(n_users, k)`` float array, 1.0 where the ranked item at
    that position is in the user's truth set.  Positions beyond the length of
    a short ranked list are 0.  ``n_truth`` is the truth-set size per user and
    ``valid`` marks users with a non-empty truth set.
    """
    if len(ranked) != len(truth):
        raise ValueError(
            f"ranked/truth length mismatch: {len(ranked)} != {len(truth)}"
        )
    if k <= 0:
        raise ValueError("k must be positive")

    n = len(ranked)
    hits = np.zeros((n, k), dtype=np.float64)
    n_truth = np.zeros(n, dtype=np.float64)

    for i, (rec, tru) in enumerate(zip(ranked, truth)):
        tset = tru if isinstance(tru, (set, frozenset)) else set(tru)
        n_truth[i] = len(tset)
        if not tset:
            continue
        seen: set[int] = set()
        for j, item in enumerate(rec[:k]):
            # A ranked list should not contain duplicates; if it does, only
            # the first occurrence may score (otherwise recall can exceed 1).
            if item in tset and item not in seen:
                hits[i, j] = 1.0
            seen.add(item)

    valid = n_truth > 0
    return hits, n_truth, valid


def _discount(k: int) -> np.ndarray:
    return 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))


def _masked(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Per-user array with ``nan`` for users that cannot be scored."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    out[valid] = values[valid]
    return out


def _mean(per_user: np.ndarray) -> float:
    if per_user.size == 0 or np.all(np.isnan(per_user)):
        return 0.0
    return float(np.nanmean(per_user))


# --------------------------------------------------------------------------
# metrics -- each returns a per-user array (nan for unscoreable users)
# --------------------------------------------------------------------------


def recall_at_k(
    ranked: Sequence[Sequence[int]], truth: Sequence[Iterable[int]], k: int
) -> np.ndarray:
    """Fraction of a user's relevant items retrieved in the top ``k``."""
    hits, n_truth, valid = _as_hit_matrix(ranked, truth, k)
    denom = np.where(valid, n_truth, 1.0)
    return _masked(hits.sum(axis=1) / denom, valid)


def hitrate_at_k(
    ranked: Sequence[Sequence[int]], truth: Sequence[Iterable[int]], k: int
) -> np.ndarray:
    """1.0 if at least one relevant item appears in the top ``k``."""
    hits, _n_truth, valid = _as_hit_matrix(ranked, truth, k)
    return _masked((hits.sum(axis=1) > 0).astype(np.float64), valid)


def ndcg_at_k(
    ranked: Sequence[Sequence[int]], truth: Sequence[Iterable[int]], k: int
) -> np.ndarray:
    """Binary-gain NDCG@k with the ideal DCG of ``min(|truth|, k)`` hits."""
    hits, n_truth, valid = _as_hit_matrix(ranked, truth, k)
    disc = _discount(k)
    dcg = (hits * disc).sum(axis=1)

    ideal_n = np.minimum(n_truth, k).astype(int)
    cum = np.concatenate([[0.0], np.cumsum(disc)])
    idcg = cum[ideal_n]
    idcg = np.where(idcg > 0, idcg, 1.0)
    return _masked(dcg / idcg, valid)


def mrr(
    ranked: Sequence[Sequence[int]],
    truth: Sequence[Iterable[int]],
    k: int | None = None,
) -> np.ndarray:
    """Reciprocal rank of the first relevant item (0 if none within ``k``)."""
    if k is None:
        k = max((len(r) for r in ranked), default=1) or 1
    hits, _n_truth, valid = _as_hit_matrix(ranked, truth, k)
    ranks = np.arange(1, k + 1, dtype=np.float64)
    rr = np.zeros(hits.shape[0], dtype=np.float64)
    any_hit = hits.sum(axis=1) > 0
    if np.any(any_hit):
        first = np.argmax(hits[any_hit] > 0, axis=1)
        rr[any_hit] = 1.0 / ranks[first]
    return _masked(rr, valid)


def item_coverage(
    ranked: Sequence[Sequence[int]], catalog_size: int, k: int
) -> float:
    """Share of the catalog that appears in any user's top ``k``."""
    if catalog_size <= 0:
        return 0.0
    distinct: set[int] = set()
    for rec in ranked:
        distinct.update(rec[:k])
    return float(len(distinct)) / float(catalog_size)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------

_PER_USER_FAMILIES = ("recall", "ndcg", "hitrate", "mrr")


def metric_names(ks: Sequence[int] = DEFAULT_KS) -> list[str]:
    """Names of every scalar metric :func:`evaluate` produces (minus coverage)."""
    return [f"{fam}@{k}" for k in ks for fam in _PER_USER_FAMILIES]


def per_user_metrics(
    ranked: Sequence[Sequence[int]],
    truth: Sequence[Iterable[int]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict[str, np.ndarray]:
    """Per-user arrays keyed ``"{metric}@{k}"``; ``nan`` where unscoreable.

    These arrays are the input to :mod:`amt.jobs.eval.paired` and
    :mod:`amt.jobs.eval.cohorts`, so paired comparisons and cohort
    breakdowns use exactly the same math as the headline numbers.
    """
    out: dict[str, np.ndarray] = {}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(ranked, truth, k)
        out[f"ndcg@{k}"] = ndcg_at_k(ranked, truth, k)
        out[f"hitrate@{k}"] = hitrate_at_k(ranked, truth, k)
        out[f"mrr@{k}"] = mrr(ranked, truth, k)
    return out


def evaluate(
    ranked: Sequence[Sequence[int]],
    truth: Sequence[Iterable[int]],
    ks: Sequence[int] = DEFAULT_KS,
    catalog_size: int = 0,
    include_per_user: bool = False,
) -> dict:
    """Compute every harness metric.

    Returns a flat ``{metric_name: float}`` dict.  ``coverage@k`` is a corpus
    level metric and has no per-user array.  ``n_users`` / ``n_scored`` are
    included so callers can tell how many users actually contributed.
    With ``include_per_user=True`` the returned dict additionally carries a
    ``"per_user"`` entry holding the raw arrays.
    """
    ks = tuple(int(k) for k in ks)
    per_user = per_user_metrics(ranked, truth, ks)

    metrics: dict[str, float] = {name: _mean(arr) for name, arr in per_user.items()}
    for k in ks:
        metrics[f"coverage@{k}"] = item_coverage(ranked, catalog_size, k)

    scored = 0
    if per_user:
        first = next(iter(per_user.values()))
        scored = int(np.count_nonzero(~np.isnan(first)))
    metrics["n_users"] = float(len(ranked))
    metrics["n_scored"] = float(scored)
    metrics["catalog_size"] = float(catalog_size)

    result: dict = dict(metrics)
    if include_per_user:
        result["per_user"] = per_user
    return result
