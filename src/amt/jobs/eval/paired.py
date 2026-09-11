"""Paired candidate-vs-baseline comparison (SPEC stage 9, phase 1).

Comparing two aggregate numbers throws away the fact that both models were
scored on *the same users*.  The paired view -- per-user deltas plus a
seeded paired bootstrap CI -- is far more sensitive and is what the QA agent
reports.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

__all__ = [
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_N_BOOTSTRAP",
    "paired_bootstrap",
    "paired_comparison",
    "per_user_deltas",
]

DEFAULT_BOOTSTRAP_SEED = 20240917
DEFAULT_N_BOOTSTRAP = 2000


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
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    boot_means = deltas[idx].mean(axis=1)
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "mean": mean,
        "ci_low": lo,
        "ci_high": hi,
        "frac_improved": float(np.mean(deltas > 0)),
        "n_users": int(n),
    }


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
