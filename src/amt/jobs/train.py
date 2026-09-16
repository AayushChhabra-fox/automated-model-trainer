"""Training job (SPEC stage 5) -- runs on Databricks, never sees QA.

Reads the frozen train and validation splits through ``SplitGuard``, fits the
family named in the ``ExperimentConfig``, evaluates on validation through the
shared harness, breaks the result down by cohort, and logs everything to
MLflow.  The same module serves ``--role baseline`` (the popularity
reference) so the baseline is produced by exactly the same code path and is
therefore comparable by construction.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from typing import Any

from amt.jobs.data.features import build_eval_frame, build_item_index
from amt.jobs.data.loaders import SplitGuard
from amt.jobs.eval import cohorts as cohort_mod
from amt.jobs.eval.harness import DEFAULT_KS, HARNESS_VERSION, evaluate
from amt.jobs.models import registry

__all__ = [
    "TRAIN_SPLITS",
    "ContractMismatch",
    "assert_contract",
    "contract_eval_settings",
    "run",
    "train_and_evaluate",
]

TRAIN_SPLITS = ("train", "validation")

# Users (in sorted id order) whose validation behaviour QA phase 0 will try to
# reproduce from the logged artifact.  Deterministic, and a subset only --
# phase 0 must not become a second validation pass.
DEFAULT_REPRODUCTION_SUBSET = 200
REPRO_PREFIX = "reproduction/"


class ContractMismatch(RuntimeError):
    """The job was handed a contract it is not allowed to run against."""


def assert_contract(
    contract: Mapping[str, Any], expected_hash: str | None = None
) -> str:
    """Enforcement mechanism #3 and #6 from PLAN §9."""
    contract_hash = contract.get("contract_hash")
    if not contract_hash:
        raise ContractMismatch("contract has no contract_hash; refusing to run")
    if expected_hash and str(expected_hash) != str(contract_hash):
        raise ContractMismatch(
            f"contract_hash mismatch: job was given {expected_hash!r} but the "
            f"contract hashes to {contract_hash!r}"
        )
    pinned = contract.get("harness_version")
    if pinned and str(pinned) != HARNESS_VERSION:
        raise ContractMismatch(
            f"harness_version mismatch: contract pins {pinned!r}, this wheel "
            f"ships {HARNESS_VERSION!r}"
        )
    return str(contract_hash)


def contract_ks(contract: Mapping[str, Any]) -> tuple[int, ...]:
    """The cutoffs the contract's own metrics name, e.g. ``ndcg@4`` -> ``4``.

    Previously the job fell back to 5/10/20 whenever ``metrics`` was a list --
    which it always is, because the contract declares ``list[MetricDef]``. A
    campaign whose primary metric is ``ndcg@4`` then never computed ``ndcg@4``
    at all: champion selection had nothing to select on, and every number in
    the run was at a cutoff nobody asked for. On a rail with three and a half
    candidates per render, ``@10`` is also trivially 1.0, so the fallback was
    not merely wrong but flattering.
    """
    metrics = contract.get("metrics")
    # A dry run may override with `{"ks": [...]}`; a real contract carries a
    # list of MetricDef and the cutoffs have to be read out of their names.
    explicit = metrics.get("ks") if isinstance(metrics, Mapping) else contract.get("ks")
    if explicit:
        return tuple(sorted({int(k) for k in explicit}))

    found: set[int] = set()
    for metric in metrics if isinstance(metrics, Sequence) else ():
        name = str((metric or {}).get("name") if isinstance(metric, Mapping) else metric or "")
        _family, _, cutoff = name.partition("@")
        if cutoff.isdigit():
            found.add(int(cutoff))
    primary = str(contract.get("primary_metric") or "")
    _family, _, cutoff = primary.partition("@")
    if cutoff.isdigit():
        found.add(int(cutoff))
    return tuple(sorted(found)) or DEFAULT_KS


def contract_eval_settings(contract: Mapping[str, Any]) -> dict[str, Any]:
    """ks / primary metric / tolerance, wherever the contract records them."""
    metrics = contract.get("metrics")
    block = metrics if isinstance(metrics, Mapping) else {}
    ks = contract_ks(contract)
    primary = (
        block.get("primary_metric")
        or contract.get("primary_metric")
        or "ndcg@10"
    )
    return {
        "ks": tuple(int(k) for k in ks),
        "primary_metric": str(primary),
        "tolerance": float(
            block.get("reproduction_tolerance")
            or contract.get("reproduction_tolerance")
            or 1e-6
        ),
        "subset_size": int(
            block.get("reproduction_subset_size")
            or contract.get("reproduction_subset_size")
            or DEFAULT_REPRODUCTION_SUBSET
        ),
    }


def _flatten(prefix: str, payload: Any, out: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(payload, (list, tuple)):
        out[prefix] = ",".join(str(v) for v in payload)
    else:
        out[prefix] = payload
    return out


def _recipe_attribute_frames(
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    item_col: str,
    spark: Any | None,
) -> dict[str, Any]:
    """Read the item-attribute tables a ``feature_recipe`` names, at item grain.

    The frozen splits still come from ``SplitGuard``; this only fetches the
    descriptive columns the approved recipe asked for. A recipe that points at
    the QA split's table -- by name rather than by split -- is refused here.
    """
    from amt.jobs.data.loaders import split_ref
    from amt.jobs.entry import to_pandas

    recipe = config.get("feature_recipe")
    if not isinstance(recipe, Mapping):
        return {}

    qa_tables = set()
    try:
        qa_tables.add(str(split_ref(contract, "qa").get("table") or "").lower())
    except Exception:  # pragma: no cover - contract without a qa ref
        pass

    frames: dict[str, Any] = {}
    for source in recipe.get("sources") or []:
        table = str(source.get("table") or "")
        if not table:
            continue
        if table.lower() in qa_tables:
            raise ContractMismatch(
                f"feature recipe source {source.get('name')!r} reads the QA split table "
                f"{table!r}; training may never see QA"
            )
        if spark is None:
            raise ContractMismatch(
                f"feature recipe source {source.get('name')!r} needs table {table!r} "
                "but this job has no Spark session"
            )
        # Attribute sources attach at the item grain; the recipe no longer
        # declares join keys because nothing else was ever supported.
        keys = [item_col]
        columns = [str(c) for c in (source.get("columns") or [])]
        wanted = ", ".join(dict.fromkeys([*keys, *columns]))
        frames[str(source.get("name") or table)] = to_pandas(
            spark.sql(f"SELECT DISTINCT {wanted} FROM {table}")
        )
    return frames


def _recipe_params(config: Mapping[str, Any]) -> dict[str, Any]:
    """Compact params identifying an approved feature/fit shape on the MLflow run."""
    recipe = config.get("feature_recipe")
    if not isinstance(recipe, Mapping):
        return {}
    label = recipe.get("label") or {}
    candidates = recipe.get("candidate_generation") or {}
    return {
        "feature_fit_hash": str(config.get("feature_fit_hash") or ""),
        "recipe.n_features": len(recipe.get("feature_columns") or []),
        "recipe.label_strategy": str(label.get("strategy") or ""),
        "recipe.candidates": str(candidates.get("strategy") or ""),
        "recipe.group_key": str(recipe.get("group_key") or ""),
    }


def _evaluate_frame(model, frame, ks: Sequence[int], catalog_size: int, k_max: int):
    """Rank the frame's units and score them, in whichever mode the frame is in.

    In ``grouped_rerank`` the model is handed the closed candidate set and the
    incumbent's slots, the truth weights come off the split, and the catalog
    ``coverage@k`` divides by is the distinct impressed set rather than the
    train catalog -- a re-ranker cannot surface an item nobody showed.
    """
    if frame.grouped:
        ranked = model.recommend(
            frame.histories, k_max, candidates=frame.candidates, positions=frame.positions
        )
        catalog_size = frame.distinct_candidates or catalog_size
        gains = frame.gains
    else:
        ranked = model.recommend(frame.histories, k_max)
        gains = None
    result = evaluate(
        ranked,
        frame.truths,
        ks=ks,
        catalog_size=catalog_size,
        include_per_user=True,
        gains=gains,
    )
    per_user = result.pop("per_user")
    if frame.grouped:
        result.update(_pointwise(model, frame))
    return result, per_user, ranked


def _pointwise(model, frame) -> dict[str, float]:
    """``auc`` / ``pr_auc`` / ``logloss`` / ``ece`` over every scored candidate.

    Only possible in ``grouped_rerank``, where every candidate carries a label.
    A ranking metric over a three-item candidate set can look healthy while the
    scores separate nothing; these say whether they do.
    """
    from amt.jobs.eval.harness import binary_metrics

    scorer = getattr(model, "score", None)
    if scorer is None or not frame.candidates:
        return {}
    try:
        scores, _groups, items = scorer(frame.histories, frame.candidates)
    except Exception:  # pragma: no cover - a family that cannot score candidates
        return {}
    if len(items) != len(scores):
        return {}
    labels = [
        1.0 if item in truth else 0.0
        for truth, row in zip(frame.truths, frame.candidates, strict=True)
        for item in row
    ]
    if len(labels) != len(scores):
        return {}
    return binary_metrics(scores, labels)


def train_and_evaluate(
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    allowed_splits: str | Sequence[str],
    spark: Any | None = None,
    local_dir: str | None = None,
    mlflow: Any | None = None,
    run_id: str = "",
    expected_contract_hash: str | None = None,
) -> dict[str, Any]:
    """Fit one config and return the ``TrainingRunResult`` payload."""
    from amt.jobs.entry import stable_hash, to_pandas

    started = time.time()
    contract_hash = assert_contract(contract, expected_contract_hash)
    settings = contract_eval_settings(contract)
    ks = settings["ks"]
    k_max = max(ks)

    guard = SplitGuard(allowed_splits, contract, local_dir=local_dir)
    # A training job that could read QA would make the whole campaign
    # unfalsifiable, so refuse before touching any data.
    guard.require_subset_of(TRAIN_SPLITS)

    train_df = to_pandas(guard.read("train", spark))
    val_df = to_pandas(guard.read("validation", spark))

    model_cfg = dict(config.get("model") or {})
    family = str(model_cfg.get("family") or config.get("family") or "popularity")
    item_col = str(config.get("item_col", "item_id"))

    index = _item_index(
        train_df, item_col, int(config.get("min_item_count", 1))
    )
    val_frame = build_eval_frame(val_df)
    extra_frames = _recipe_attribute_frames(contract, config, item_col, spark)

    def _log_curve(step: int, metrics: Mapping[str, float]) -> None:
        if mlflow is None:
            return
        mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
            step=int(step),
        )

    def _evaluator(model) -> dict[str, float]:
        scores, _pu, _r = _evaluate_frame(
            model, val_frame, ks, index.catalog_size, k_max
        )
        return scores

    model = registry.get_builder(family)(
        train_df,
        val_df,
        config,
        index=index,
        logger=_log_curve,
        evaluator=_evaluator if family != "popularity" else None,
        extra_frames=extra_frames,
    )

    val_metrics, val_per_user, _ranked = _evaluate_frame(
        model, val_frame, ks, index.catalog_size, k_max
    )

    # Reproduction subset: deterministic prefix of the sorted validation
    # users, re-scored at QA phase 0 from the logged artifact.
    subset = min(int(settings["subset_size"]), len(val_frame.histories))
    if subset > 0:
        sub_scores, _sub_pu, _ = _evaluate_frame(
            model,
            _slice_frame(val_frame, subset),
            ks,
            index.catalog_size,
            k_max,
        )
        for key, value in sub_scores.items():
            val_metrics[f"{REPRO_PREFIX}{key}"] = value
    val_metrics[f"{REPRO_PREFIX}n_users"] = float(subset)

    cohort_metrics = cohort_mod.cohort_metrics(val_per_user, cohorts_for(val_frame))

    train_metrics = _train_metrics(model, train_df, index)

    model_uri = ""
    if mlflow is not None:
        model_uri = registry.log_pyfunc_model(
            model,
            artifact_path="model",
            metadata={
                "family": family,
                "contract_hash": contract_hash,
                "harness_version": HARNESS_VERSION,
            },
        )
        # The recipe is logged whole as the config artifact; flattening it into
        # params would bury the run under one param per transform.
        mlflow.log_params(
            _flatten("", {k: v for k, v in config.items() if k != "feature_recipe"}, {})
        )
        mlflow.log_params(
            {
                "model.family": family,
                "catalog_size": index.catalog_size,
                "harness_version": HARNESS_VERSION,
                **_recipe_params(config),
            }
        )
        mlflow.log_metrics({f"train/{k}": float(v) for k, v in train_metrics.items()})
        mlflow.log_metrics(
            {f"val/{k}": float(v) for k, v in val_metrics.items()}
        )
        mlflow.log_metrics(
            {
                f"val/cohort/{cohort}/{metric}": float(value)
                for cohort, block in cohort_metrics.items()
                for metric, value in block.items()
            }
        )
        _log_config_yaml(mlflow, config)

    return {
        "run_id": run_id,
        "mlflow_run_id": (
            mlflow.active_run().info.run_id
            if mlflow is not None and mlflow.active_run() is not None
            else None
        ),
        "config": dict(config),
        "config_hash": stable_hash(config),
        "train_metrics": {k: float(v) for k, v in train_metrics.items()},
        "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        "cohort_metrics": {
            cohort: {m: float(v) for m, v in block.items()}
            for cohort, block in cohort_metrics.items()
        },
        "runtime_seconds": time.time() - started,
        "model_uri": model_uri,
        "status": "success",
        "error": None,
        "contract_hash": contract_hash,
        "harness_version": HARNESS_VERSION,
        "primary_metric": settings["primary_metric"],
        "splits_read": guard.reads,
    }


def _item_index(train_df, item_col: str, min_count: int):
    """The catalog `coverage@k` divides by, in whichever grain the split is.

    A render-grain split has no item column: its catalog is the set of titles
    the rail actually impressed, which is also the only set a re-ranker could
    ever surface.
    """
    import pandas as pd

    from amt.jobs.data.features import CANDIDATES_COLUMN, as_item_list

    frame = train_df
    if CANDIDATES_COLUMN in getattr(train_df, "columns", ()):
        frame = pd.DataFrame(
            {
                item_col: [
                    item
                    for row in train_df[CANDIDATES_COLUMN]
                    for item in as_item_list(row)
                ]
            }
        )
    return build_item_index(frame, item_col=item_col, min_count=min_count)


def cohorts_for(frame) -> dict[str, list[str]]:
    """The reported slices for a frame, in whichever grain it is.

    ``next_item`` derives them from history length and recency. A render has
    neither -- its "history" is the candidate set -- so a grouped frame reports
    the dimensions the contract named and materialization carried onto the row,
    and reports nothing if it carries none. Inventing a history-length cohort
    from a candidate count would be a number that looks like evidence.
    """
    if frame.grouped:
        return {k: [str(v) for v in labels] for k, labels in (frame.cohorts or {}).items()}
    return cohort_mod.assign_cohorts(frame.histories, frame.days_since_last)


def _slice_frame(frame, n: int):
    """The first ``n`` units, with every parallel array kept aligned."""
    from amt.jobs.data.features import EvalFrame

    def head(values):
        return values[:n] if values is not None else None

    return EvalFrame(
        user_ids=frame.user_ids[:n],
        histories=frame.histories[:n],
        truths=frame.truths[:n],
        days_since_last=head(frame.days_since_last),
        # Dropping these silently re-scored the reproduction subset in
        # `next_item` mode: no candidates, no weights, different numbers, and
        # QA phase 0 then failing to reproduce its own logged validation.
        candidates=head(frame.candidates),
        positions=head(frame.positions),
        gains=head(frame.gains),
        timestamps=head(frame.timestamps),
        cohorts=(
            {k: v[:n] for k, v in frame.cohorts.items()}
            if frame.cohorts is not None
            else None
        ),
    )


def _train_metrics(model, train_df, index) -> dict[str, float]:
    """Training-side summary: fit diagnostics, not a second evaluation."""
    user_col = "user_id" if "user_id" in train_df.columns else train_df.columns[0]
    metrics: dict[str, float] = {
        "n_interactions": float(len(train_df)),
        "n_users": float(train_df[user_col].nunique() if len(train_df) else 0),
        "catalog_size": float(index.catalog_size),
    }
    history = getattr(model, "history", None)
    if history:
        last = history[-1]
        metrics["epochs"] = float(len(history))
        for key in ("train/loss", "val/primary"):
            if key in last:
                metrics[key.replace("/", "_")] = float(last[key])
    cfg = getattr(model, "cfg", {}) or {}
    if "best_epoch" in cfg:
        metrics["best_epoch"] = float(cfg["best_epoch"])
    if hasattr(model, "n_params"):
        try:
            metrics["params"] = float(model.n_params())
        except Exception:  # pragma: no cover - diagnostics only
            pass
    return metrics


def _log_config_yaml(mlflow: Any, config: Mapping[str, Any]) -> None:
    try:
        import yaml

        text = yaml.safe_dump(dict(config), sort_keys=True)
    except Exception:  # pragma: no cover - yaml is a job-cluster dependency
        import json

        text = json.dumps(dict(config), indent=2, default=str)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "config.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        mlflow.log_artifact(path)


def run(args: Any) -> dict[str, Any]:
    """``--role train`` / ``--role baseline`` entry point."""
    from amt.jobs.entry import get_spark, load_json_arg

    contract = load_json_arg(args.contract_json) or {}
    config = load_json_arg(args.config_json) or {}
    if args.role == "baseline" and not config:
        config = {"model": {"family": "popularity"}}

    mlflow = None
    try:
        import mlflow as mlflow_mod

        mlflow = mlflow_mod if mlflow_mod.active_run() is not None else None
    except Exception:
        mlflow = None

    return train_and_evaluate(
        contract,
        config,
        args.allowed_splits,
        spark=get_spark(),
        local_dir=getattr(args, "local_dir", None),
        mlflow=mlflow,
        run_id=getattr(args, "job_run_id", "") or "",
    )
