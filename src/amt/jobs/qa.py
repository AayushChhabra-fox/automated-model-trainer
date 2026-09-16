"""Independent QA job (SPEC stages 9/10).

Two strictly ordered phases:

**Phase 0 -- integrity & reproducibility.** Loads the frozen champion the
way a serving system would (MLflow pyfunc, not the trainer's in-memory
object), checks the artifact, the schema, the contract stamp and the harness
version, and re-scores an approved *subset* of validation to confirm the
artifact reproduces its logged behaviour.  This is the only phase that may
return ``INVALID_EXPERIMENT``.

**Phase 1 -- holdout evaluation.** The only place in the system that reads
the QA split, and it must be granted *exactly* ``qa``.  Evaluates the frozen
candidate and the baseline on the identical QA population, produces paired
per-user deltas with bootstrap CIs, cohort breakdowns and operational
metrics.

Nothing in this module fits, refits or mutates the candidate; it has no
optimiser, no config search and no write path back into training.  A poor QA
result is evidence, not a signal (SPEC "QA Independence").
"""

from __future__ import annotations

import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

from amt.jobs.data.features import build_eval_frame
from amt.jobs.data.loaders import SplitGuard
from amt.jobs.eval import cohorts as cohort_mod
from amt.jobs.eval import paired as paired_mod
from amt.jobs.eval.harness import HARNESS_VERSION, evaluate
from amt.jobs.eval.paired import paired_comparison
from amt.jobs.models.registry import recommend_with_pyfunc
from amt.jobs.train import (
    REPRO_PREFIX,
    assert_contract,
    cohorts_for,
    contract_eval_settings,
)

__all__ = [
    "PHASE0_SPLITS",
    "PHASE1_SPLITS",
    "Check",
    "phase0",
    "phase1",
    "run_phase0",
    "run_phase1",
]

PHASE0_SPLITS = ("validation",)
PHASE1_SPLITS = ("qa",)


def Check(name: str, passed: bool, detail: str = "") -> dict[str, Any]:
    """One phase-0 integrity check result."""
    return {"name": name, "passed": bool(passed), "detail": str(detail)}


def _load_pyfunc(model_uri: str):
    import mlflow.pyfunc

    return mlflow.pyfunc.load_model(model_uri)


def _slice(frame, n: int):
    """The reproduction subset, sliced exactly as training sliced it.

    Training and QA must take the *same* prefix with the *same* parallel
    arrays -- candidates, positions, weights, cohorts -- or phase 0 re-scores a
    different problem and fails "reproduces_logged_validation" for reasons that
    have nothing to do with the artifact. One implementation, imported.
    """
    from amt.jobs.train import _slice_frame

    return _slice_frame(frame, n)


# --------------------------------------------------------------------------
# Phase 0
# --------------------------------------------------------------------------


def phase0(
    contract: Mapping[str, Any],
    frozen: Mapping[str, Any],
    allowed_splits: str | Sequence[str],
    spark: Any | None = None,
    local_dir: str | None = None,
    loader: Any = None,
) -> dict[str, Any]:
    """Integrity and reproducibility of the frozen champion."""
    from amt.jobs.entry import to_pandas

    checks: list[dict[str, Any]] = []
    reproduction_deltas: dict[str, float] = {}
    settings = contract_eval_settings(contract)
    ks, tolerance = settings["ks"], settings["tolerance"]
    k_max = max(ks)

    # -- contract stamps (cheap, and a mismatch invalidates everything else)
    try:
        contract_hash = assert_contract(contract, frozen.get("contract_hash"))
        checks.append(Check("contract_hash_matches", True, contract_hash))
    except Exception as exc:
        return _phase0_result(
            [Check("contract_hash_matches", False, str(exc))], {}, contract, False
        )

    frozen_harness = frozen.get("harness_version")
    checks.append(
        Check(
            "harness_version_matches",
            not frozen_harness or str(frozen_harness) == HARNESS_VERSION,
            f"frozen={frozen_harness!r} wheel={HARNESS_VERSION!r}",
        )
    )

    guard = SplitGuard(allowed_splits, contract, local_dir=local_dir)
    try:
        # Phase 0 may look at validation (for the reproduction subset) and
        # nothing else -- certainly not QA.
        guard.require_subset_of(PHASE0_SPLITS)
        checks.append(Check("split_scope", True, str(sorted(guard.allowed_splits))))
    except Exception as exc:
        checks.append(Check("split_scope", False, str(exc)))
        return _phase0_result(checks, {}, contract, False)

    # -- artifact loads
    model_uri = str(frozen.get("model_uri") or "")
    model = None
    try:
        model = (loader or _load_pyfunc)(model_uri)
        checks.append(Check("artifact_loads", True, model_uri))
    except Exception as exc:
        checks.append(Check("artifact_loads", False, f"{model_uri}: {exc}"))
        return _phase0_result(checks, {}, contract, False)

    # -- signature / schema
    checks.append(_signature_check(model))

    # -- inference executes on the approved validation subset
    try:
        val_df = to_pandas(guard.read("validation", spark))
        frame = build_eval_frame(val_df)
        n = int(frozen.get("reproduction_subset_size") or settings["subset_size"])
        subset = _slice(frame, min(n, len(frame.histories)))
        ranked = recommend_with_pyfunc(
            model,
            subset.histories,
            k_max,
            candidates=subset.candidates if subset.grouped else None,
            positions=subset.positions if subset.grouped else None,
            contexts=subset.contexts if subset.grouped else None,
        )
        shape_ok = len(ranked) == len(subset.histories)
        checks.append(
            Check(
                "inference_executes",
                shape_ok,
                f"{len(ranked)} rows for {len(subset.histories)} users",
            )
        )
        checks.append(
            Check(
                "output_ranked_lists",
                all(isinstance(r, list) for r in ranked),
                "predict returned ranked lists",
            )
        )
    except Exception as exc:
        checks.append(Check("inference_executes", False, str(exc)))
        return _phase0_result(checks, {}, contract, False)

    # -- reproduces the logged validation behaviour within tolerance
    logged = {
        key[len(REPRO_PREFIX) :]: value
        for key, value in (frozen.get("val_metrics") or {}).items()
        if key.startswith(REPRO_PREFIX)
    }
    catalog_size = int(
        frozen.get("catalog_size")
        or (frozen.get("val_metrics") or {}).get("catalog_size")
        or 0
    )
    if subset.grouped:
        # Same catalog and the same relevance weights training scored with.
        catalog_size = subset.distinct_candidates or catalog_size
    recomputed = evaluate(
        ranked,
        subset.truths,
        ks=ks,
        catalog_size=catalog_size,
        gains=subset.gains if subset.grouped else None,
    )

    if not logged:
        checks.append(
            Check(
                "reproduces_logged_validation",
                False,
                "training run logged no reproduction subset metrics "
                f"(expected keys prefixed {REPRO_PREFIX!r})",
            )
        )
        return _phase0_result(checks, {}, contract, False)

    worst = 0.0
    for metric, expected in logged.items():
        if metric in ("n_users", "catalog_size", "n_scored"):
            continue
        actual = float(recomputed.get(metric, float("nan")))
        delta = abs(actual - float(expected))
        reproduction_deltas[metric] = float(actual - float(expected))
        worst = max(worst, delta)
    checks.append(
        Check(
            "reproduces_logged_validation",
            worst <= tolerance,
            f"max |delta| {worst:.6g} vs tolerance {tolerance:.6g}",
        )
    )

    passed = all(c["passed"] for c in checks)
    return _phase0_result(checks, reproduction_deltas, contract, passed)


def _signature_check(model) -> dict[str, Any]:
    try:
        signature = getattr(getattr(model, "metadata", None), "signature", None)
    except Exception as exc:  # pragma: no cover
        return Check("signature_schema", False, str(exc))
    if signature is None:
        return Check(
            "signature_schema", False, "model logged without an input signature"
        )
    names = []
    try:
        names = [c.name for c in signature.inputs.inputs]
    except Exception:  # pragma: no cover
        names = []
    return Check(
        "signature_schema",
        "history" in names,
        f"input columns {names}",
    )


def _phase0_result(
    checks: Sequence[Mapping[str, Any]],
    reproduction_deltas: Mapping[str, float],
    contract: Mapping[str, Any],
    passed: bool,
) -> dict[str, Any]:
    return {
        "passed": bool(passed),
        "checks": [dict(c) for c in checks],
        "reproduction_deltas": dict(reproduction_deltas),
        "status": "ok" if passed else "INVALID_EXPERIMENT",
        "contract_hash": contract.get("contract_hash"),
        "harness_version": HARNESS_VERSION,
    }


# --------------------------------------------------------------------------
# Phase 1
# --------------------------------------------------------------------------


def phase1(
    contract: Mapping[str, Any],
    frozen: Mapping[str, Any],
    allowed_splits: str | Sequence[str],
    baseline_ref: Mapping[str, Any] | None = None,
    spark: Any | None = None,
    local_dir: str | None = None,
    loader: Any = None,
) -> dict[str, Any]:
    """Evaluate frozen candidate vs baseline on the untouched QA split."""
    from amt.jobs.entry import to_pandas

    contract_hash = assert_contract(contract, frozen.get("contract_hash"))
    settings = contract_eval_settings(contract)
    ks = settings["ks"]
    k_max = max(ks)

    guard = SplitGuard(allowed_splits, contract, local_dir=local_dir)
    # Exactly qa: phase 1 must not be able to re-read train or validation,
    # which is what would make "re-tune after seeing QA" physically possible.
    guard.require_exactly(PHASE1_SPLITS)

    qa_df = to_pandas(guard.read("qa", spark))
    frame = build_eval_frame(qa_df)
    catalog_size = int(
        frozen.get("catalog_size")
        or (frozen.get("val_metrics") or {}).get("catalog_size")
        or 0
    )

    load = loader or _load_pyfunc
    candidate = load(str(frozen.get("model_uri") or ""))

    baseline_ref = dict(baseline_ref or {})
    baseline_uri = baseline_ref.get("model_uri")
    if not baseline_uri:
        raise RuntimeError(
            "phase 1 needs the baseline's logged model_uri; re-fitting the "
            "baseline here would require the train split, which this job is "
            "not granted"
        )
    baseline = load(str(baseline_uri))

    grouped = frame.grouped
    if grouped:
        catalog_size = frame.distinct_candidates or catalog_size

    def rank(model) -> list[list[Any]]:
        return recommend_with_pyfunc(
            model,
            frame.histories,
            k_max,
            candidates=frame.candidates if grouped else None,
            positions=frame.positions if grouped else None,
            contexts=frame.contexts if grouped else None,
        )

    cand_ranked = rank(candidate)
    base_ranked = rank(baseline)

    # Proxy sanity runs *before* any candidate number is read. If the offline
    # metric does not track the product number for the ranker already in
    # production, nothing below it can certify a new one, and displaying the
    # candidate's score first is how a broken proxy gets approved anyway.
    proxy = _proxy_sanity(
        contract,
        frame,
        base_ranked,
        settings["primary_metric"],
        ks,
        catalog_size,
        spark,
    )

    cand = evaluate(
        cand_ranked, frame.truths, ks=ks, catalog_size=catalog_size,
        include_per_user=True, gains=frame.gains,
    )
    base = evaluate(
        base_ranked, frame.truths, ks=ks, catalog_size=catalog_size,
        include_per_user=True, gains=frame.gains,
    )
    cand_per_user = cand.pop("per_user")
    base_per_user = base.pop("per_user")

    paired = paired_comparison(cand_per_user, base_per_user)

    assignments = cohorts_for(frame)
    cohort_candidate = cohort_mod.cohort_metrics(cand_per_user, assignments)
    cohort_baseline = cohort_mod.cohort_metrics(base_per_user, assignments)
    cohort_out: dict[str, dict[str, float]] = {}
    for cohort, block in cohort_candidate.items():
        merged = {m: float(v) for m, v in block.items()}
        for metric, value in cohort_baseline.get(cohort, {}).items():
            merged[f"baseline/{metric}"] = float(value)
            if metric != "n_users":
                merged[f"delta/{metric}"] = float(block.get(metric, 0.0)) - float(value)
        cohort_out[cohort] = merged

    operational = _operational(candidate, frame, k_max, frozen)

    # Product units and the A/B sizing, in the unit the dashboard reports.
    candidate_units = paired_mod.top_slot_conversions(cand_ranked, frame.truths, frame.gains)
    baseline_units = paired_mod.top_slot_conversions(base_ranked, frame.truths, frame.gains)
    design = paired_mod.ab_design(
        baseline_units["per_1k"] / 1000.0,
        candidate_units["per_1k"] / 1000.0,
        renders_per_day=_renders_per_day(frame),
    )

    return {
        "candidate_metrics": {k: float(v) for k, v in cand.items()},
        "baseline_metrics": {k: float(v) for k, v in base.items()},
        "paired_deltas": {
            metric: {k: float(v) for k, v in stats.items()}
            for metric, stats in paired.items()
        },
        "cohort_metrics": cohort_out,
        "operational": operational,
        "proxy_sanity": proxy,
        "proxy_valid": bool(proxy.get("passed", True)),
        "product_units": {
            "candidate": candidate_units,
            "baseline": baseline_units,
            "unit": "top-slot conversions per 1,000 renders",
        },
        "ab_design": design,
        "contract_hash": contract_hash,
        "harness_version": HARNESS_VERSION,
        "primary_metric": settings["primary_metric"],
        "baseline_model_uri": str(baseline_uri),
        "candidate_model_uri": str(frozen.get("model_uri") or ""),
        "n_qa_users": len(frame.histories),
        "splits_read": guard.reads,
        "status": "success",
        "error": None,
    }


def _renders_per_day(frame) -> float:
    """Units per distinct day in the QA split -- the traffic the A/B sizing uses."""
    days = _day_keys(frame)
    distinct = len({d for d in days if d})
    return float(len(frame.histories)) / distinct if distinct else 0.0


def _day_keys(frame) -> list[str]:
    """The calendar day of each unit, as ``YYYY-MM-DD``, or ``""`` when unknown."""
    stamps = getattr(frame, "timestamps", None) or []
    out = []
    for value in stamps:
        text = str(value or "")
        out.append(text[:10] if len(text) >= 10 else "")
    return out


def _proxy_sanity(
    contract: Mapping[str, Any],
    frame,
    baseline_ranked: Sequence[Sequence[Any]],
    primary_metric: str,
    ks: Sequence[int],
    catalog_size: int,
    spark: Any | None,
) -> dict[str, Any]:
    """Does the offline metric move with the product number, day by day?

    Correlated on the *incumbent*: the model already in production, whose
    product number is already being watched. If its offline score and the
    dashboard's series disagree across days, the offline metric is not
    measuring what the campaign claims, and the candidate's score is
    uninterpretable however good it looks.

    Two correlations are reported. The first is against an external series the
    contract names (``product_metric_source``) -- a real cross-source test.
    The second is against the split's own daily top-slot conversion rate, which
    is the fallback when no external series is configured and is deliberately
    the weaker evidence, because it shares our label.
    """
    criteria = dict(contract.get("acceptance_criteria") or {})
    threshold = float(criteria.get("min_proxy_correlation", 0.5))
    min_days = int(criteria.get("min_proxy_days", 7))

    days = _day_keys(frame)
    if not any(days):
        return {
            "passed": True,
            "reason": "the split carries no timestamps, so no daily series exists",
            "rho_external": None,
            "rho_own_label": None,
            "days": 0,
            "threshold": threshold,
            "checked": False,
        }

    offline = _daily_offline(frame, baseline_ranked, days, primary_metric, ks, catalog_size)
    own = _daily_top_slot_rate(frame, baseline_ranked, days)
    ordered = sorted(offline)
    rho_own = paired_mod.spearman(
        [offline[d] for d in ordered], [own[d] for d in ordered]
    )

    external = _external_series(contract, spark)
    rho_external: float | None = None
    overlap: list[str] = []
    if external:
        overlap = sorted(set(offline) & set(external))
        rho_external = paired_mod.spearman(
            [offline[d] for d in overlap], [external[d] for d in overlap]
        )

    rho = rho_external if rho_external is not None else rho_own
    days_used = len(overlap) if external else len(ordered)
    passed = True
    reason = ""
    if rho is None or rho != rho:  # nan
        passed, reason = False, "the correlation is undefined: too few distinct daily values"
    elif days_used < min_days:
        passed, reason = (
            False,
            f"{days_used} overlapping day(s) is below the {min_days} the contract requires",
        )
    elif rho < threshold:
        passed, reason = (
            False,
            f"rho={rho:.3f} is below the {threshold} the contract pre-registered: the "
            "offline metric does not track the product number for the incumbent",
        )

    return {
        "passed": passed,
        "reason": reason,
        "rho": None if rho is None else float(rho),
        "rho_external": None if rho_external is None else float(rho_external),
        "rho_own_label": None if rho_own != rho_own else float(rho_own),
        "days": days_used,
        "threshold": threshold,
        "min_days": min_days,
        "source": "external" if external else "own_label",
        "offline_by_day": {d: float(offline[d]) for d in ordered},
        "product_by_day": (
            {d: float(external[d]) for d in overlap}
            if external
            else {d: float(own[d]) for d in ordered}
        ),
        "checked": True,
    }


def _daily_offline(
    frame,
    ranked: Sequence[Sequence[Any]],
    days: Sequence[str],
    primary_metric: str,
    ks: Sequence[int],
    catalog_size: int,
) -> dict[str, float]:
    """The incumbent's primary metric, recomputed on each day's units."""
    buckets: dict[str, list[int]] = {}
    for i, day in enumerate(days):
        if day:
            buckets.setdefault(day, []).append(i)
    out: dict[str, float] = {}
    for day, rows in buckets.items():
        scored = evaluate(
            [ranked[i] for i in rows],
            [frame.truths[i] for i in rows],
            ks=ks,
            catalog_size=catalog_size,
            gains=[frame.gains[i] for i in rows] if frame.gains else None,
        )
        out[day] = float(scored.get(primary_metric, 0.0))
    return out


def _daily_top_slot_rate(
    frame, ranked: Sequence[Sequence[Any]], days: Sequence[str]
) -> dict[str, float]:
    """The split's own daily top-slot conversion rate, unweighted."""
    hits: dict[str, list[float]] = {}
    for i, day in enumerate(days):
        if not day:
            continue
        picks = ranked[i] if i < len(ranked) else []
        truth = frame.truths[i]
        hits.setdefault(day, []).append(
            1.0 if picks and picks[0] in truth else 0.0
        )
    return {day: sum(values) / len(values) for day, values in hits.items() if values}


def _external_series(contract: Mapping[str, Any], spark: Any | None) -> dict[str, float]:
    """The product metric's own daily series, read from the table the contract names.

    Read-only and best effort: a missing grant or an unreachable table makes
    the check fall back to the split's own label rather than failing QA for a
    reason that has nothing to do with the model.
    """
    source = dict(contract.get("product_metric_source") or {})
    table = str(source.get("table") or "")
    if not table or spark is None:
        return {}
    date_column = str(source.get("date_column") or "")
    value_column = str(source.get("value_column") or "")
    if not (date_column and value_column):
        return {}
    where = str(source.get("filter_sql") or "") or "TRUE"
    try:
        rows = spark.sql(
            f"SELECT cast({date_column} AS string) AS d, "
            f"avg(cast({value_column} AS double)) AS v "
            f"FROM {table} WHERE {where} GROUP BY 1"
        ).collect()
    except Exception:  # pragma: no cover - diagnostics, not a QA failure
        return {}
    return {
        str(row["d"])[:10]: float(row["v"])
        for row in rows
        if row["d"] is not None and row["v"] is not None
    }


def _operational(model, frame, k: int, frozen: Mapping[str, Any]) -> dict[str, float]:
    """Serving-shaped measurements: single-request latency, size, params.

    One request is one unit *with everything a request carries*: in grouped
    mode that is the render's candidates and context, or the model is being
    timed on a code path it never serves (and refuses to run).
    """
    import numpy as np

    n = min(64, len(frame.histories)) or 1
    timings = []
    for i in range(n):
        history = frame.histories[i] if i < len(frame.histories) else []
        kwargs: dict[str, Any] = {}
        if frame.grouped:
            kwargs["candidates"] = [frame.candidates[i]]
            if frame.positions is not None:
                kwargs["positions"] = [frame.positions[i]]
            if frame.contexts is not None:
                kwargs["contexts"] = [frame.contexts[i]]
        start = time.perf_counter()
        recommend_with_pyfunc(model, [history], k, **kwargs)
        timings.append((time.perf_counter() - start) * 1000.0)

    return {
        "latency_ms_p50": float(np.percentile(np.asarray(timings), 50)),
        "latency_ms_p95": float(np.percentile(np.asarray(timings), 95)),
        "model_size_mb": _model_size_mb(model, frozen),
        "params": _param_count(model),
    }


def _model_size_mb(model, frozen: Mapping[str, Any]) -> float:
    root = None
    for attr in ("_model_meta", "metadata"):
        meta = getattr(model, attr, None)
        root = getattr(meta, "_model_root", None) or root
    root = root or getattr(model, "_model_impl", None)
    path = getattr(root, "_model_root", None) if root is not None else None
    if not isinstance(path, str):
        path = str(frozen.get("model_local_path") or "")
    if not path or not os.path.isdir(path):
        return 0.0
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:  # pragma: no cover
                pass
    return float(total) / (1024.0 * 1024.0)


def _param_count(model) -> float:
    inner = None
    unwrap = getattr(model, "unwrap_python_model", None)
    if callable(unwrap):
        try:
            inner = getattr(unwrap(), "model", None)
        except Exception:  # pragma: no cover
            inner = None
    if inner is None:
        # A RecommenderPyfunc used directly (local dry runs, tests).
        inner = getattr(model, "model", None)
    if inner is None:
        return 0.0
    if hasattr(inner, "n_params"):
        try:
            return float(inner.n_params())
        except Exception:  # pragma: no cover
            return 0.0
    return float(getattr(inner, "catalog_size", 0) or 0)


# --------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------


def run_phase0(args: Any) -> dict[str, Any]:
    from amt.jobs.entry import get_spark, load_json_arg

    return phase0(
        load_json_arg(args.contract_json) or {},
        load_json_arg(args.frozen_json) or {},
        args.allowed_splits,
        spark=get_spark(),
        local_dir=getattr(args, "local_dir", None),
    )


def run_phase1(args: Any) -> dict[str, Any]:
    from amt.jobs.entry import get_spark, load_json_arg

    frozen = load_json_arg(args.frozen_json) or {}
    baseline_ref = frozen.get("baseline") or {}
    if args.baseline_run_id and not baseline_ref.get("model_uri"):
        baseline_ref = {
            **baseline_ref,
            "run_id": args.baseline_run_id,
            "model_uri": f"runs:/{args.baseline_run_id}/model",
        }
    return phase1(
        load_json_arg(args.contract_json) or {},
        frozen,
        args.allowed_splits,
        baseline_ref=baseline_ref,
        spark=get_spark(),
        local_dir=getattr(args, "local_dir", None),
    )
