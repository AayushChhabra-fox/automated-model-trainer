"""The single evaluation harness for the whole campaign.

Every metric used anywhere in AMT (trainer validation, baseline, QA phase 1)
is computed by the functions in this module.  The version string is recorded
in the :class:`EvaluationContract` and asserted by every job, so a change to
the math invalidates in-flight campaigns instead of silently changing the
ground truth (SPEC "Evaluation Metrics" / "Invariance").

Everything here is pure ``numpy``: no Spark, no torch, no MLflow.

Conventions
-----------
``ranked``  : ``list[list[int]]`` -- per unit, item ids ordered best-first.
``truth``   : ``list[set[int]]``  -- per unit, the relevant (held-out) items.
``gains``   : ``list[dict[int, float]]`` -- optional per-unit relevance weight
              of each truth item. ``None`` means every hit counts 1, which is
              what every metric below reduces to.
Units whose truth set is empty are excluded from the averages but still
appear (as ``nan``) in the per-unit arrays so that arrays stay aligned with
the input order.

A "unit" is a user in ``next_item`` mode and one render of a container in
``grouped_rerank`` mode. Nothing here knows the difference: the inputs are a
ranked list and a truth set, and a render is both.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

#: 1.1.0, not 1.0.0: `evaluate` grew a `gains` argument, and every metric's
#: unweighted result is byte-identical (a unit test pins that), so campaigns
#: already in flight compare cleanly. The version still moves, because the
#: contract asserts it and a reader deserves to know which harness ran.
HARNESS_VERSION = "1.1.0"

DEFAULT_KS: tuple[int, ...] = (5, 10, 20)

__all__ = [
    "DEFAULT_KS",
    "HARNESS_VERSION",
    "binary_metrics",
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


class _Hits:
    """Per-unit gains of the top-k hits, plus the totals each metric divides by.

    ``matrix[i, j]`` is the gain of the item ranked ``j`` for unit ``i``, or 0
    when it is not relevant. With no weighting every gain is 1.0 and the
    fields below collapse to counts, which is why every metric reduces exactly
    to its unweighted form.
    """

    __slots__ = ("best", "ideal", "matrix", "n_truth", "total", "valid")

    def __init__(
        self,
        matrix: np.ndarray,
        total: np.ndarray,
        best: np.ndarray,
        ideal: np.ndarray,
        n_truth: np.ndarray,
    ) -> None:
        self.matrix = matrix
        self.total = total  # sum of every truth gain
        self.best = best  # largest single truth gain
        self.ideal = ideal  # DCG of the best achievable ordering
        self.n_truth = n_truth
        self.valid = n_truth > 0


def _as_hits(
    ranked: Sequence[Sequence[int]],
    truth: Sequence[Iterable[int]],
    k: int,
    gains: Sequence[Mapping[Any, float]] | None = None,
) -> _Hits:
    if len(ranked) != len(truth):
        raise ValueError(f"ranked/truth length mismatch: {len(ranked)} != {len(truth)}")
    if gains is not None and len(gains) != len(ranked):
        raise ValueError(f"ranked/gains length mismatch: {len(ranked)} != {len(gains)}")
    if k <= 0:
        raise ValueError("k must be positive")

    n = len(ranked)
    matrix = np.zeros((n, k), dtype=np.float64)
    total = np.zeros(n, dtype=np.float64)
    best = np.zeros(n, dtype=np.float64)
    ideal = np.zeros(n, dtype=np.float64)
    n_truth = np.zeros(n, dtype=np.float64)
    disc = _discount(k)

    for i, (rec, tru) in enumerate(zip(ranked, truth, strict=True)):
        tset = tru if isinstance(tru, (set, frozenset)) else set(tru)
        n_truth[i] = len(tset)
        if not tset:
            continue
        weights = gains[i] if gains is not None else None
        gain_of = {item: float((weights or {}).get(item, 1.0)) for item in tset}
        total[i] = sum(gain_of.values())
        best[i] = max(gain_of.values())
        # The best any ordering could do: the heaviest conversions first.
        top = sorted(gain_of.values(), reverse=True)[:k]
        ideal[i] = float(np.dot(top, disc[: len(top)]))

        seen: set[Any] = set()
        for j, item in enumerate(rec[:k]):
            # A ranked list should not contain duplicates; if it does, only
            # the first occurrence may score (otherwise recall can exceed 1).
            if item in gain_of and item not in seen:
                matrix[i, j] = gain_of[item]
            seen.add(item)

    return _Hits(matrix, total, best, ideal, n_truth)


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
    ranked: Sequence[Sequence[int]],
    truth: Sequence[Iterable[int]],
    k: int,
    gains: Sequence[Mapping[Any, float]] | None = None,
) -> np.ndarray:
    """Share of a unit's relevance retrieved in the top ``k``.

    Self-normalised: the recovered gain over the total available gain, so a
    weighted recall is still a fraction and reduces to ``hits / |truth|``.
    """
    return _recall(_as_hits(ranked, truth, k, gains))


def _recall(hits: _Hits) -> np.ndarray:
    denom = np.where(hits.valid, hits.total, 1.0)
    return _masked(hits.matrix.sum(axis=1) / denom, hits.valid)


def hitrate_at_k(
    ranked: Sequence[Sequence[int]],
    truth: Sequence[Iterable[int]],
    k: int,
    gains: Sequence[Mapping[Any, float]] | None = None,
) -> np.ndarray:
    """1.0 if at least one relevant item appears in the top ``k``.

    Weighted, it is the best hit's gain over the best available gain: finding
    the conversion that happened at slot 3 counts fully, and finding only the
    easy slot-0 one when a slot-3 one existed does not.
    """
    return _hitrate(_as_hits(ranked, truth, k, gains))


def _hitrate(hits: _Hits) -> np.ndarray:
    denom = np.where(hits.valid, hits.best, 1.0)
    return _masked(hits.matrix.max(axis=1) / denom, hits.valid)


def ndcg_at_k(
    ranked: Sequence[Sequence[int]],
    truth: Sequence[Iterable[int]],
    k: int,
    gains: Sequence[Mapping[Any, float]] | None = None,
) -> np.ndarray:
    """NDCG@k against the ideal ordering of whatever gains the unit carries."""
    return _ndcg(_as_hits(ranked, truth, k, gains), k)


def _ndcg(hits: _Hits, k: int) -> np.ndarray:
    dcg = (hits.matrix * _discount(k)).sum(axis=1)
    idcg = np.where(hits.ideal > 0, hits.ideal, 1.0)
    return _masked(dcg / idcg, hits.valid)


def mrr(
    ranked: Sequence[Sequence[int]],
    truth: Sequence[Iterable[int]],
    k: int | None = None,
    gains: Sequence[Mapping[Any, float]] | None = None,
) -> np.ndarray:
    """Reciprocal rank of the first relevant item (0 if none within ``k``).

    Weighted, the reciprocal rank is scaled by that hit's share of the best
    available gain, so ranking a low-propensity conversion first counts more.
    """
    if k is None:
        k = max((len(r) for r in ranked), default=1) or 1
    return _mrr(_as_hits(ranked, truth, k, gains), k)


def _mrr(hits: _Hits, k: int) -> np.ndarray:
    ranks = np.arange(1, k + 1, dtype=np.float64)
    rr = np.zeros(hits.matrix.shape[0], dtype=np.float64)
    any_hit = hits.matrix.sum(axis=1) > 0
    if np.any(any_hit):
        rows = hits.matrix[any_hit]
        first = np.argmax(rows > 0, axis=1)
        share = rows[np.arange(rows.shape[0]), first] / np.where(
            hits.best[any_hit] > 0, hits.best[any_hit], 1.0
        )
        rr[any_hit] = share / ranks[first]
    return _masked(rr, hits.valid)


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
    gains: Sequence[Mapping[Any, float]] | None = None,
) -> dict[str, np.ndarray]:
    """Per-unit arrays keyed ``"{metric}@{k}"``; ``nan`` where unscoreable.

    These arrays are the input to :mod:`amt.jobs.eval.paired` and
    :mod:`amt.jobs.eval.cohorts`, so paired comparisons and cohort
    breakdowns use exactly the same math as the headline numbers.

    The hit matrix is built once per ``k`` and shared by the four metrics that
    read it. Calling the four public functions instead would scan the ranked
    lists four times, which at the half a million renders a real container
    produces is four times a minute of Python rather than four times nothing.
    """
    out: dict[str, np.ndarray] = {}
    for k in ks:
        hits = _as_hits(ranked, truth, k, gains)
        out[f"recall@{k}"] = _recall(hits)
        out[f"ndcg@{k}"] = _ndcg(hits, k)
        out[f"hitrate@{k}"] = _hitrate(hits)
        out[f"mrr@{k}"] = _mrr(hits, k)
    return out


def binary_metrics(scores: Sequence[float], labels: Sequence[float]) -> dict[str, float]:
    """Pointwise quality of the scores themselves: ``auc``, ``pr_auc``, ``logloss``, ``ece``.

    These need a label on every candidate, which only ``grouped_rerank``
    provides. They are diagnostics: they say whether the model separates
    converters from non-converters at all, which a ranking metric can hide
    when the candidate set is only three items long. They are never the
    primary metric -- see ``is_secondary_only_metric``.

    Returns ``{}`` when one class is absent, because none of the four means
    anything then.
    """
    y = np.asarray(labels, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if y.size != s.size:
        raise ValueError(f"scores/labels length mismatch: {s.size} != {y.size}")
    keep = np.isfinite(s) & np.isfinite(y)
    y, s = y[keep], s[keep]
    positives = int(np.count_nonzero(y > 0))
    if y.size == 0 or positives in (0, y.size):
        return {}

    # AUC as the rank-sum (Mann-Whitney U) statistic, so ties count as half.
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    _unique, inverse, counts = np.unique(s, return_inverse=True, return_counts=True)
    mean_rank = np.zeros(counts.size, dtype=np.float64)
    np.add.at(mean_rank, inverse, ranks)
    ranks = (mean_rank / counts)[inverse]
    negatives = y.size - positives
    auc = (ranks[y > 0].sum() - positives * (positives + 1) / 2.0) / (positives * negatives)

    # Average precision: the step-wise integral of the PR curve.
    ordered = y[np.argsort(-s, kind="mergesort")]
    tp = np.cumsum(ordered)
    precision = tp / np.arange(1, ordered.size + 1, dtype=np.float64)
    pr_auc = float((precision * ordered).sum() / positives)

    probability = _to_probability(s)
    eps = 1e-12
    logloss = float(
        -np.mean(y * np.log(probability + eps) + (1 - y) * np.log(1 - probability + eps))
    )

    # Expected calibration error over ten equal-width probability bins.
    bins = np.clip((probability * 10).astype(int), 0, 9)
    ece = 0.0
    for b in range(10):
        mask = bins == b
        if not np.any(mask):
            continue
        ece += (mask.sum() / y.size) * abs(probability[mask].mean() - y[mask].mean())

    return {
        "auc": float(auc),
        "pr_auc": pr_auc,
        "logloss": logloss,
        "ece": float(ece),
    }


def _to_probability(scores: np.ndarray) -> np.ndarray:
    """Scores as probabilities, squashing only when they are not already in [0, 1].

    A classifier's output is a probability; a ranker's margin is not, and
    ``logloss`` on a raw margin is meaningless rather than merely large.
    """
    if scores.size and scores.min() >= 0.0 and scores.max() <= 1.0:
        return scores
    return 1.0 / (1.0 + np.exp(-scores))


def evaluate(
    ranked: Sequence[Sequence[int]],
    truth: Sequence[Iterable[int]],
    ks: Sequence[int] = DEFAULT_KS,
    catalog_size: int = 0,
    include_per_user: bool = False,
    gains: Sequence[Mapping[Any, float]] | None = None,
) -> dict:
    """Compute every harness metric.

    Returns a flat ``{metric_name: float}`` dict.  ``coverage@k`` is a corpus
    level metric and has no per-unit array.  ``n_users`` / ``n_scored`` are
    included so callers can tell how many units actually contributed.
    With ``include_per_user=True`` the returned dict additionally carries a
    ``"per_user"`` entry holding the raw arrays.

    ``gains`` weights each unit's relevant items. Metric *names* do not change
    when it is supplied -- ``ndcg@4`` is ``ndcg@4`` -- because the weighting is
    a contract-level fact stamped into ``contract_hash`` and therefore
    identical for the baseline, every candidate and QA.
    """
    ks = tuple(int(k) for k in ks)
    per_user = per_user_metrics(ranked, truth, ks, gains)

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
