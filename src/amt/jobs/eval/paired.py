"""Paired candidate-vs-baseline comparison (SPEC stage 9, phase 1).

Comparing two aggregate numbers throws away the fact that both models were
scored on *the same users*.  The paired view -- per-user deltas plus a
seeded paired bootstrap CI -- is far more sensitive and is what the QA agent
reports.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_N_BOOTSTRAP",
    "ab_design",
    "paired_bootstrap",
    "paired_comparison",
    "per_user_deltas",
    "spearman",
    "top_slot_conversions",
]

DEFAULT_BOOTSTRAP_SEED = 20240917
DEFAULT_N_BOOTSTRAP = 2000

#: Two-sided 95 % significance and 80 % power: (1.9600 + 0.8416)^2.
_Z_SUM_SQUARED = 7.849


def per_user_deltas(
    candidate: Sequence[float], baseline: Sequence[float]
) -> np.ndarray:
    """``candidate - baseline`` per user, dropping users either side cannot score.

    Both inputs come from :func:`amt.jobs.eval.harness.per_user_metrics` and
    must be aligned on the same user ordering.
    """
    cand = np.asarray(candidate, dtype=np.float64)
    base = np.asarray(baseline, dtype=np.float64)
    if cand.shape != base.shape:
        raise ValueError(
            f"paired arrays must align: {cand.shape} != {base.shape}"
        )
    keep = ~(np.isnan(cand) | np.isnan(base))
    return cand[keep] - base[keep]


def paired_bootstrap(
    candidate: Sequence[float],
    baseline: Sequence[float],
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Seeded paired bootstrap over users.

    Resamples *users* (keeping each user's candidate/baseline pair together),
    which is the correct unit of independence here.  Returns the mean delta,
    a percentile CI at ``1 - alpha``, the fraction of users improved and the
    number of paired users.
    """
    deltas = per_user_deltas(candidate, baseline)
    n = deltas.size
    if n == 0:
        return {
            "mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "frac_improved": 0.0,
            "n_users": 0,
        }

    mean = float(deltas.mean())
    boot_means = _bootstrap_means(deltas, n_boot, seed)
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "frac_improved": float(np.mean(deltas > 0)),
        "n_users": int(n),
    }


#: Resample indices are drawn in chunks of this many rows. A single
#: ``(n_boot, n)`` index array is 2000 x n int64: at the 500 000 QA renders a
#: real container produces that is 8 GB, which is how a QA job dies after the
#: campaign has already spent its entire compute budget. Chunking keeps the
#: peak at a few hundred MB and changes no result: the same seed draws the same
#: numbers in the same order either way.
_BOOTSTRAP_CHUNK_ROWS = 20_000_000


def _bootstrap_means(values: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """Means of ``n_boot`` bootstrap resamples of ``values``, memory-bounded."""
    n = values.size
    n_boot = int(n_boot)
    rng = np.random.default_rng(seed)
    per_chunk = max(1, _BOOTSTRAP_CHUNK_ROWS // max(n, 1))
    means = np.empty(n_boot, dtype=np.float64)
    done = 0
    while done < n_boot:
        rows = min(per_chunk, n_boot - done)
        idx = rng.integers(0, n, size=(rows, n))
        means[done : done + rows] = values[idx].mean(axis=1)
        done += rows
    return means


def paired_comparison(
    candidate_per_user: Mapping[str, Sequence[float]],
    baseline_per_user: Mapping[str, Sequence[float]],
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, dict[str, float]]:
    """Run :func:`paired_bootstrap` for every metric both sides computed."""
    out: dict[str, dict[str, float]] = {}
    for metric in sorted(set(candidate_per_user) & set(baseline_per_user)):
        out[metric] = paired_bootstrap(
            candidate_per_user[metric],
            baseline_per_user[metric],
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
        )
    return out


# --------------------------------------------------------------------------
# the report's arithmetic: product units, proxy correlation, A/B sizing
# --------------------------------------------------------------------------


def _ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not invent an ordering."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    _unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    totals = np.zeros(counts.size, dtype=np.float64)
    np.add.at(totals, inverse, ranks)
    return (totals / counts)[inverse]


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Rank correlation of two aligned daily series; ``nan`` when undefined.

    Rank rather than Pearson because the question is whether the two numbers
    move together, not whether they are on the same scale -- our label is
    playback-within-90s and the dashboard's is a click, so the levels differ
    by construction and only the co-movement is evidence.
    """
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(y, dtype=np.float64).ravel()
    if a.size != b.size:
        raise ValueError(f"series must align: {a.size} != {b.size}")
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3 or np.unique(a).size < 2 or np.unique(b).size < 2:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def top_slot_conversions(
    ranked: Sequence[Sequence[Any]],
    truth: Sequence[Iterable[Any]],
    gains: Sequence[Mapping[Any, float]] | None = None,
    per: float = 1000.0,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Expected conversions at the top slot per ``per`` renders, with a CI.

    For each render: if the model's first pick is an item that converted,
    credit it ``1 / theta`` for the slot that item was actually shown in. That
    is the position-based estimate of "would this have been taken had it been
    first". For the incumbent it collapses to its own observed slot-0 rate,
    because its first pick *is* the slot-0 item and ``theta_0`` is 1 -- which
    is why the incumbent's number in the report matches the dashboard's.

    This is the unit a product owner reads. ``ndcg@4`` is not.
    """
    credits = np.zeros(len(ranked), dtype=np.float64)
    for i, (picks, tru) in enumerate(zip(ranked, truth, strict=True)):
        if not picks:
            continue
        top = picks[0]
        relevant = tru if isinstance(tru, (set, frozenset)) else set(tru)
        if top in relevant:
            weights = gains[i] if gains is not None else None
            credits[i] = float((weights or {}).get(top, 1.0))

    if credits.size == 0:
        return {"per_1k": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_renders": 0}

    boot = _bootstrap_means(credits, n_boot, seed) * per
    return {
        "per_1k": float(credits.mean() * per),
        "ci_low": float(np.percentile(boot, 100.0 * (alpha / 2.0))),
        "ci_high": float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0))),
        "n_renders": int(credits.size),
    }


def ab_design(
    baseline_rate: float,
    candidate_rate: float,
    renders_per_day: float = 0.0,
) -> dict[str, float]:
    """What it would take to prove this offline effect online.

    Two-sample proportions at 95 % significance and 80 % power. The point is
    not precision -- traffic and novelty effects will move it -- but whether
    the effect is inside the range an experiment could ever resolve. An offline
    delta that needs six months of traffic is not a result, it is a rounding
    error with a confidence interval.
    """
    p0 = max(0.0, min(1.0, float(baseline_rate)))
    p1 = max(0.0, min(1.0, float(candidate_rate)))
    effect = p1 - p0
    pooled = (p0 + p1) / 2.0
    variance = 2.0 * pooled * (1.0 - pooled)

    out = {
        "baseline_rate": p0,
        "candidate_rate": p1,
        "absolute_effect": effect,
        "relative_effect": (effect / p0) if p0 > 0 else 0.0,
    }
    if effect == 0.0 or variance <= 0.0:
        # No effect to size, or a degenerate rate: say so rather than
        # dividing by zero and reporting an infinity as a plan.
        out.update({"required_renders": 0.0, "days_at_current_traffic": 0.0, "mde": 0.0})
        return out

    per_arm = _Z_SUM_SQUARED * variance / (effect**2)
    required = 2.0 * per_arm
    out["required_renders"] = float(required)
    out["days_at_current_traffic"] = (
        float(required / renders_per_day) if renders_per_day > 0 else 0.0
    )
    # The smallest effect this much traffic could resolve, for the reader who
    # wants to know whether the experiment is worth running at all.
    if renders_per_day > 0:
        out["mde"] = float(np.sqrt(_Z_SUM_SQUARED * variance / (renders_per_day / 2.0)))
    else:
        out["mde"] = 0.0
    return out
