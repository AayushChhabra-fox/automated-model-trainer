"""Shared runtime for the gradient-boosting families (XGBoost / LightGBM).

The four booster families differ only in backend and objective; everything that
decides what the model actually sees -- sources, joins, features, label, candidate
set -- arrives as a ``feature_recipe`` dict inside the ``ExperimentConfig`` and is
executed here.  That recipe was approved by a human at the feature/fit checkpoint,
so this module treats it as the specification and refuses anything outside the
closed transform vocabulary rather than improvising.

Feature construction runs through :func:`build_rows` at fit time *and* at
inference time. There is exactly one implementation so that QA phase 0's
train/inference consistency check has something real to verify, and so that a
feature which cannot be computed from a serving-time history cannot be trained on.

The recipe is read as a plain mapping: nothing under ``amt.jobs`` may import the
orchestrator's Pydantic schemas (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

__all__ = [
    "BASE_COLUMNS",
    "TabularModel",
    "UnsupportedRecipe",
    "build_rows",
    "build_state",
    "fit_family",
    "load",
    "resolve_cfg",
]

BASE_COLUMNS: tuple[str, ...] = (
    "item_id",
    "history_length",
    "item_train_count",
    "item_train_rank",
)

TRANSFORM_KINDS = frozenset(
    {"column", "cast", "bucket", "hash", "count", "ratio", "literal"}
)

FILTER_OPS = frozenset(
    {"eq", "ne", "gt", "gte", "lt", "lte", "in", "not_in", "not_null"}
)

DEFAULTS: dict[str, Any] = {
    "user_col": "user_id",
    "item_col": "item_id",
    "ts_col": "ts",
    "num_boost_round": 200,
    "early_stopping_rounds": 20,
    "negative_samples_per_positive": 4,
    "seed": 42,
    "max_train_rows": 500_000,
    "max_validation_users": 2_000,
}

# Backend defaults per family: (objective, eval_metric).
FAMILY_OBJECTIVES: dict[str, tuple[str, str]] = {
    "xgboost_classifier": ("binary:logistic", "logloss"),
    "xgboost_reranker": ("rank:pairwise", "ndcg"),
    "lightgbm_classifier": ("binary", "binary_logloss"),
    "lightgbm_reranker": ("lambdarank", "ndcg"),
}

UNKNOWN_CODE = 0


def _is_missing(value: Any) -> bool:
    """None, NaN or pandas NA -- attributes are sparse in event tables."""
    if value is None:
        return True
    try:
        return bool(value != value)
    except Exception:  # pragma: no cover - exotic dtypes
        return False


class UnsupportedRecipe(ValueError):
    """The recipe asks for something this runtime does not implement."""


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


FIT_KNOBS = frozenset(set(DEFAULTS) | {"objective", "eval_metric"})


def resolve_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Split the ExperimentConfig into AMT knobs, booster params and the recipe.

    Precedence, weakest first: defaults, the approved ``fit_spec``, then the
    Trainer's ``model`` / ``training`` sections. Anything that is not a known
    AMT knob is a booster hyperparameter and nothing else is passed to the
    booster -- campaign plumbing (ids, prose, hashes) must not leak into it.
    ``loss.type`` applies only to configs without a recipe; see below.
    """
    cfg = dict(cfg or {})
    flat = dict(DEFAULTS)
    booster: dict[str, Any] = {}
    has_recipe = bool(cfg.get("feature_recipe"))

    fit_spec = cfg.get("fit_spec")
    if isinstance(fit_spec, Mapping):
        flat.update({k: v for k, v in fit_spec.items() if k in FIT_KNOBS})
        params = fit_spec.get("booster_params")
        if isinstance(params, Mapping):
            booster.update(dict(params))

    for section in ("model", "training"):
        block = cfg.get(section)
        if not isinstance(block, Mapping):
            continue
        for key, value in block.items():
            if key == "family":
                continue
            if key in FIT_KNOBS:
                # Every AMT knob is part of the shape a human approved, so with a
                # recipe the Trainer's sections cannot move one -- same reason
                # `loss.type` is ignored below. Dropped, not demoted to a booster
                # param: `max_train_rows` is not a hyperparameter.
                if not has_recipe:
                    flat[key] = value
            else:
                booster[key] = value

    loss = cfg.get("loss")
    if isinstance(loss, Mapping) and loss.get("type") and not has_recipe:
        # With a recipe the objective is part of the shape a human approved, so
        # the Trainer's free-form `loss` cannot redirect it -- it names losses
        # the boosters do not have (`pairwise_ranking`).
        flat["objective"] = loss["type"]

    flat.update({k: v for k, v in cfg.items() if k in FIT_KNOBS})
    flat["feature_recipe"] = cfg.get("feature_recipe")
    flat["booster_params"] = booster
    return flat


def _booster_params(flat: Mapping[str, Any]) -> dict[str, Any]:
    """Scalar hyperparameters only: the boosters reject structured values."""
    params = flat.get("booster_params")
    if not isinstance(params, Mapping):
        return {}
    return {
        k: v
        for k, v in params.items()
        if isinstance(v, (int, float, bool, str, list, tuple))
    }


# --------------------------------------------------------------------------
# recipe validation
# --------------------------------------------------------------------------


def _require_recipe(flat: Mapping[str, Any]) -> dict[str, Any]:
    recipe = flat.get("feature_recipe")
    if not isinstance(recipe, Mapping):
        raise UnsupportedRecipe(
            "the tabular families require an approved feature_recipe in the ExperimentConfig"
        )
    recipe = json.loads(json.dumps(recipe, default=str))
    transforms = recipe.get("feature_columns") or []
    if not transforms:
        raise UnsupportedRecipe("feature_recipe has no feature_columns")
    for transform in transforms:
        kind = str(transform.get("kind") or "")
        if kind not in TRANSFORM_KINDS:
            raise UnsupportedRecipe(
                f"transform {transform.get('name')!r} uses unsupported kind {kind!r}; "
                f"supported: {sorted(TRANSFORM_KINDS)}"
            )
    for source in recipe.get("sources") or []:
        if str(source.get("split") or "") == "qa":
            raise UnsupportedRecipe("a feature recipe may never read the qa split")
    for rule in recipe.get("filters") or []:
        if str(rule.get("op") or "") not in FILTER_OPS:
            raise UnsupportedRecipe(f"filter uses unsupported op {rule.get('op')!r}")
    return recipe


def _attribute_columns(recipe: Mapping[str, Any]) -> list[str]:
    """Columns the recipe needs joined onto each candidate row."""
    wanted: list[str] = []
    for source in recipe.get("sources") or []:
        if str(source.get("role") or "interactions") != "item_attributes":
            continue
        for column in source.get("columns") or []:
            if column not in wanted:
                wanted.append(str(column))
    for transform in recipe.get("feature_columns") or []:
        for key in ("source", "of"):
            name = transform.get(key)
            if name and name not in BASE_COLUMNS and name not in wanted:
                wanted.append(str(name))
    for rule in recipe.get("filters") or []:
        name = rule.get("column")
        if name and name not in BASE_COLUMNS and name not in wanted:
            wanted.append(str(name))
    return wanted


# --------------------------------------------------------------------------
# fitted side state
# --------------------------------------------------------------------------


class _State:
    """Everything learned from train that inference needs. Serialized as JSON."""

    def __init__(
        self,
        catalog: list[Any],
        counts: dict[str, float],
        ranks: dict[str, int],
        attributes: dict[str, dict[str, Any]],
        attribute_columns: list[str],
        train_value_counts: dict[str, dict[str, float]],
        category_codes: dict[str, dict[str, int]],
        feature_names: list[str],
    ) -> None:
        self.catalog = catalog
        self.counts = counts
        self.ranks = ranks
        self.attributes = attributes
        self.attribute_columns = attribute_columns
        self.train_value_counts = train_value_counts
        self.category_codes = category_codes
        self.feature_names = feature_names

    def to_dict(self) -> dict[str, Any]:
        return {
            "catalog": self.catalog,
            "counts": self.counts,
            "ranks": self.ranks,
            "attributes": self.attributes,
            "attribute_columns": self.attribute_columns,
            "train_value_counts": self.train_value_counts,
            "category_codes": self.category_codes,
            "feature_names": self.feature_names,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> _State:
        return cls(
            catalog=list(payload.get("catalog") or []),
            counts=dict(payload.get("counts") or {}),
            ranks={k: int(v) for k, v in (payload.get("ranks") or {}).items()},
            attributes=dict(payload.get("attributes") or {}),
            attribute_columns=list(payload.get("attribute_columns") or []),
            train_value_counts=dict(payload.get("train_value_counts") or {}),
            category_codes={
                col: {str(k): int(v) for k, v in mapping.items()}
                for col, mapping in (payload.get("category_codes") or {}).items()
            },
            feature_names=list(payload.get("feature_names") or []),
        )

    def attribute_of(self, item: Any, column: str) -> Any:
        return (self.attributes.get(str(item)) or {}).get(column)


# --------------------------------------------------------------------------
# feature construction (one implementation, fit and inference)
# --------------------------------------------------------------------------


def _as_float(values: Sequence[Any]) -> np.ndarray:
    import pandas as pd

    return pd.to_numeric(pd.Series(list(values)), errors="coerce").fillna(0.0).to_numpy(
        dtype=np.float64
    )


def _encode(values: Sequence[Any], column: str, state: _State) -> np.ndarray:
    codes = state.category_codes.get(column) or {}
    return np.asarray(
        [float(codes.get(str(v), UNKNOWN_CODE)) for v in values], dtype=np.float64
    )


def _hash_bucket(value: Any, buckets: int) -> float:
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
    return float(int(digest[:12], 16) % int(buckets))


def _column_values(
    name: str,
    raw: dict[str, list[Any]],
    built: dict[str, np.ndarray],
) -> list[Any] | np.ndarray:
    if name in raw:
        return raw[name]
    if name in built:
        return built[name]
    raise UnsupportedRecipe(
        f"transform references unknown column {name!r}; available: "
        f"{sorted(set(raw) | set(built))}"
    )


def build_rows(
    recipe: Mapping[str, Any],
    state: _State,
    histories: Sequence[Sequence[Any]],
    candidates: Sequence[Sequence[Any]],
):
    """Build the numeric feature matrix for ``(history, candidate)`` pairs.

    Returns ``(X, group_index, item_ids)`` where ``group_index[i]`` is the index of
    the history row ``i`` belongs to -- the grouping the rankers train on and the
    grouping inference re-ranks within.
    """
    import pandas as pd

    attribute_columns = state.attribute_columns
    raw: dict[str, list[Any]] = {name: [] for name in BASE_COLUMNS}
    raw.update({name: [] for name in attribute_columns})
    group_index: list[int] = []
    history_values: dict[str, list[list[Any]]] = {name: [] for name in attribute_columns}
    history_items: list[list[Any]] = []

    for row, (history, items) in enumerate(zip(histories, candidates, strict=True)):
        history = list(history or ())
        seen_attrs = {
            column: [state.attribute_of(item, column) for item in history]
            for column in attribute_columns
        }
        for item in items:
            key = str(item)
            group_index.append(row)
            raw["item_id"].append(item)
            raw["history_length"].append(float(len(history)))
            raw["item_train_count"].append(float(state.counts.get(key, 0.0)))
            raw["item_train_rank"].append(
                float(state.ranks.get(key, len(state.catalog) + 1))
            )
            for column in attribute_columns:
                raw[column].append(state.attribute_of(item, column))
                history_values[column].append(seen_attrs[column])
            history_items.append(history)

    built: dict[str, np.ndarray] = {}
    for transform in recipe.get("feature_columns") or []:
        name = str(transform["name"])
        kind = str(transform["kind"])
        if kind == "literal":
            built[name] = np.full(len(group_index), float(transform["value"]))
            continue
        if kind == "ratio":
            num = np.asarray(
                _as_float(_column_values(str(transform["numerator"]), raw, built)),
                dtype=np.float64,
            )
            den = np.asarray(
                _as_float(_column_values(str(transform["denominator"]), raw, built)),
                dtype=np.float64,
            )
            built[name] = np.divide(
                num, den, out=np.zeros_like(num), where=np.abs(den) > 1e-12
            )
            continue
        if kind == "count":
            of = str(transform["of"])
            over = str(transform["over"])
            built[name] = _count_feature(of, over, raw, state, history_values, history_items)
            continue

        source = str(transform["source"])
        values = _column_values(source, raw, built)
        if kind == "column":
            built[name] = _numeric_or_code(values, source, state)
        elif kind == "cast":
            target = str(transform["to"])
            built[name] = (
                _encode(list(values), source, state)
                if target == "code"
                else _as_float(list(values))
                if target == "float"
                else np.floor(_as_float(list(values)))
            )
        elif kind == "bucket":
            boundaries = [float(b) for b in transform["boundaries"]]
            built[name] = np.digitize(_as_float(list(values)), boundaries).astype(np.float64)
        elif kind == "hash":
            buckets = int(transform["buckets"])
            built[name] = np.asarray(
                [_hash_bucket(v, buckets) for v in list(values)], dtype=np.float64
            )

    names = state.feature_names or [str(t["name"]) for t in recipe["feature_columns"]]
    frame = pd.DataFrame({name: built[name] for name in names})
    return frame, np.asarray(group_index, dtype=np.int64), list(raw["item_id"])


def _numeric_or_code(values: Sequence[Any], column: str, state: _State) -> np.ndarray:
    """Numeric columns pass through; anything categorical uses the fitted codes."""
    if column in state.category_codes:
        return _encode(list(values), column, state)
    return _as_float(list(values))


def _count_feature(
    of: str,
    over: str,
    raw: Mapping[str, list[Any]],
    state: _State,
    history_values: Mapping[str, list[list[Any]]],
    history_items: Sequence[Sequence[Any]],
) -> np.ndarray:
    if over == "train":
        if of == "item_id":
            return np.asarray(
                [float(state.counts.get(str(v), 0.0)) for v in raw["item_id"]],
                dtype=np.float64,
            )
        table = state.train_value_counts.get(of) or {}
        if of not in raw:
            raise UnsupportedRecipe(f"count over train needs column {of!r} in the recipe sources")
        return np.asarray([float(table.get(str(v), 0.0)) for v in raw[of]], dtype=np.float64)
    if over != "history":
        raise UnsupportedRecipe(f"count scope {over!r} is not supported")
    if of == "item_id":
        return np.asarray(
            [
                float(list(history).count(item))
                for item, history in zip(raw["item_id"], history_items, strict=True)
            ],
            dtype=np.float64,
        )
    if of not in history_values:
        raise UnsupportedRecipe(f"count over history needs column {of!r} in the recipe sources")
    return np.asarray(
        [
            float(sum(1 for seen in seen_values if seen is not None and seen == value))
            for value, seen_values in zip(raw[of], history_values[of], strict=True)
        ],
        dtype=np.float64,
    )


# --------------------------------------------------------------------------
# candidates
# --------------------------------------------------------------------------


def candidate_strategy(recipe: Mapping[str, Any]) -> str:
    return str((recipe.get("candidate_generation") or {}).get("strategy") or "popularity_topn")


def uses_logged_candidates(recipe: Mapping[str, Any]) -> bool:
    """Whether the candidate set is the one the container actually showed.

    There is nothing to generate then, and no ``n``: the set is closed at three
    to five items per render, which is what retires the
    ``n x validation_rows`` feature frame for this mode.
    """
    return candidate_strategy(recipe) == "logged_impressions"


def _candidates_for(
    recipe: Mapping[str, Any], state: _State, history: Sequence[Any]
) -> list[Any]:
    block = recipe.get("candidate_generation") or {}
    strategy = str(block.get("strategy") or "popularity_topn")
    n = int(block.get("n") or 200)
    exclude_seen = bool(block.get("exclude_seen", True))
    seen = set(map(str, history or ()))

    if strategy == "logged_impressions":
        # The candidates arrive on the group row; a generator asked for them
        # here has lost them, which is a bug rather than a fallback.
        raise UnsupportedRecipe(
            "candidate strategy 'logged_impressions' takes the candidates from the "
            "group row; nothing should be generating them"
        )

    picks: list[Any] = []
    if strategy == "history_attribute_expansion":
        attribute = str(block.get("attribute") or "")
        wanted = {
            str(state.attribute_of(item, attribute))
            for item in (history or ())
            if state.attribute_of(item, attribute) is not None
        }
        for item in state.catalog:
            if exclude_seen and str(item) in seen:
                continue
            if str(state.attribute_of(item, attribute)) in wanted:
                picks.append(item)
            if len(picks) >= n:
                break
    elif strategy != "popularity_topn":
        raise UnsupportedRecipe(f"candidate strategy {strategy!r} is not supported")

    if len(picks) < n:
        chosen = set(map(str, picks))
        for item in state.catalog:
            if exclude_seen and str(item) in seen:
                continue
            if str(item) in chosen:
                continue
            picks.append(item)
            if len(picks) >= n:
                break
    return picks[:n]


# --------------------------------------------------------------------------
# backends
# --------------------------------------------------------------------------


class _Backend:
    name = ""

    def fit(self, *args, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError

    def predict(self, booster, frame):  # pragma: no cover - interface
        raise NotImplementedError


class _XGBoostBackend(_Backend):
    name = "xgboost"
    artifact = "booster.ubj"

    def _module(self):
        try:
            import xgboost
        except Exception as exc:  # pragma: no cover - job-cluster dependency
            raise ImportError(
                "the xgboost families need a working xgboost (`pip install 'amt[jobs]'`; "
                f"on macOS it also needs libomp). Import failed with: {exc}"
            ) from exc
        return xgboost

    def fit(self, train, validation, params, rounds, early_stopping_rounds, ranker):
        xgboost = self._module()
        dtrain = xgboost.DMatrix(train["X"], label=train["y"])
        if ranker:
            dtrain.set_group(train["group_sizes"])
        evals = []
        dval = None
        if validation is not None:
            dval = xgboost.DMatrix(validation["X"], label=validation["y"])
            if ranker:
                dval.set_group(validation["group_sizes"])
            evals = [(dtrain, "train"), (dval, "validation")]
        curve: dict[str, dict[str, list[float]]] = {}
        kwargs: dict[str, Any] = {"num_boost_round": int(rounds), "verbose_eval": False}
        if evals:
            kwargs["evals"] = evals
            kwargs["evals_result"] = curve
            if early_stopping_rounds:
                kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        booster = xgboost.train(params, dtrain, **kwargs)
        best = int(getattr(booster, "best_iteration", 0) or 0) + 1
        return booster, curve, best

    def predict(self, booster, frame):
        xgboost = self._module()
        return np.asarray(booster.predict(xgboost.DMatrix(frame)), dtype=np.float64)

    def save(self, booster, path: str) -> None:
        booster.save_model(os.path.join(path, self.artifact))

    def load(self, path: str):
        xgboost = self._module()
        booster = xgboost.Booster()
        booster.load_model(os.path.join(path, self.artifact))
        return booster


class _LightGBMBackend(_Backend):
    name = "lightgbm"
    artifact = "booster.txt"

    def _module(self):
        try:
            import lightgbm
        except Exception as exc:  # pragma: no cover - job-cluster dependency
            raise ImportError(
                "the lightgbm families need a working lightgbm (`pip install 'amt[jobs]'`; "
                f"on macOS it also needs libomp). Import failed with: {exc}"
            ) from exc
        return lightgbm

    def fit(self, train, validation, params, rounds, early_stopping_rounds, ranker):
        lightgbm = self._module()
        dtrain = lightgbm.Dataset(
            train["X"],
            label=train["y"],
            group=train["group_sizes"] if ranker else None,
            free_raw_data=False,
        )
        valid_sets, valid_names = [], []
        if validation is not None:
            valid_sets.append(
                lightgbm.Dataset(
                    validation["X"],
                    label=validation["y"],
                    group=validation["group_sizes"] if ranker else None,
                    reference=dtrain,
                    free_raw_data=False,
                )
            )
            valid_names.append("validation")
        curve: dict[str, dict[str, list[float]]] = {}
        callbacks = [lightgbm.record_evaluation(curve)]
        if valid_sets and early_stopping_rounds:
            callbacks.append(
                lightgbm.early_stopping(int(early_stopping_rounds), verbose=False)
            )
        booster = lightgbm.train(
            params,
            dtrain,
            num_boost_round=int(rounds),
            valid_sets=valid_sets or None,
            valid_names=valid_names or None,
            callbacks=callbacks,
        )
        best = int(getattr(booster, "best_iteration", 0) or 0) or int(rounds)
        return booster, curve, best

    def predict(self, booster, frame):
        return np.asarray(booster.predict(frame), dtype=np.float64)

    def save(self, booster, path: str) -> None:
        booster.save_model(os.path.join(path, self.artifact))

    def load(self, path: str):
        lightgbm = self._module()
        return lightgbm.Booster(model_file=os.path.join(path, self.artifact))


BACKENDS: dict[str, _Backend] = {
    "xgboost": _XGBoostBackend(),
    "lightgbm": _LightGBMBackend(),
}


def _backend_for(family: str) -> _Backend:
    return BACKENDS["lightgbm" if family.startswith("lightgbm") else "xgboost"]


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------


class TabularModel:
    """A fitted booster plus the approved recipe it was built from."""

    def __init__(
        self,
        family: str,
        booster: Any,
        recipe: Mapping[str, Any],
        state: _State,
        cfg: Mapping[str, Any] | None = None,
        history: Sequence[Mapping[str, float]] = (),
    ) -> None:
        self.family = family
        self.booster = booster
        self.recipe = dict(recipe)
        self.state = state
        self.cfg: dict[str, Any] = dict(cfg or {})
        self.history: list[dict[str, float]] = [dict(h) for h in history]
        self.backend = _backend_for(family)

    # -- inference -----------------------------------------------------

    @property
    def catalog_size(self) -> int:
        return len(self.state.catalog)

    @property
    def feature_names(self) -> list[str]:
        return list(self.state.feature_names)

    def score(self, histories: Sequence[Sequence[Any]], candidates: Sequence[Sequence[Any]]):
        frame, groups, items = build_rows(self.recipe, self.state, histories, candidates)
        if len(frame) == 0:
            return np.zeros(0, dtype=np.float64), groups, items
        return self.backend.predict(self.booster, frame), groups, items

    def recommend(
        self,
        histories: Sequence[Sequence[Any]],
        k: int = 10,
        candidates: Sequence[Sequence[Any]] | None = None,
        positions: Sequence[Sequence[float]] | None = None,
    ) -> list[list[Any]]:
        """Top-``k`` items per unit: take or generate candidates, score, rank.

        ``candidates`` is supplied in ``grouped_rerank`` mode, where the set is
        closed at what the container showed; ``positions`` is accepted for
        interface symmetry with the incumbent and deliberately ignored, because
        a model that scored on the slot it was given could not be served.
        """
        k = int(k)
        histories = [list(h or ()) for h in histories]
        if candidates is None:
            candidates = [
                _candidates_for(self.recipe, self.state, history) for history in histories
            ]
        else:
            candidates = [list(row or ()) for row in candidates]
        scores, groups, items = self.score(histories, candidates)

        out: list[list[Any]] = [[] for _ in histories]
        if len(items) == 0:
            return out
        order = np.argsort(-scores, kind="stable")
        for idx in order:
            row = int(groups[idx])
            if len(out[row]) >= k:
                continue
            out[row].append(items[int(idx)])
        return out

    # -- persistence ---------------------------------------------------

    def save(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"family": self.family, "cfg": self.cfg}, fh, default=str)
        with open(os.path.join(path, "state.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "recipe": self.recipe,
                    "state": self.state.to_dict(),
                    "history": self.history,
                },
                fh,
                default=str,
            )
        self.backend.save(self.booster, path)
        return path

    @classmethod
    def load(cls, path: str) -> TabularModel:
        with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        with open(os.path.join(path, "state.json"), encoding="utf-8") as fh:
            payload = json.load(fh)
        family = str(meta.get("family") or "")
        backend = _backend_for(family)
        return cls(
            family=family,
            booster=backend.load(path),
            recipe=payload.get("recipe") or {},
            state=_State.from_dict(payload.get("state") or {}),
            cfg=meta.get("cfg") or {},
            history=payload.get("history") or [],
        )


def load(path: str) -> TabularModel:
    return TabularModel.load(path)


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------


def _item_stats(train_df, item_col: str) -> tuple[list[Any], dict[str, float], dict[str, int]]:
    """Catalog, impression counts and popularity ranks from the train split.

    A render-grain split has no item column: its items are the ``candidates``
    arrays, so they are exploded first and a count is an impression, not an
    interaction. (These feed ``item_train_count`` / ``item_train_rank``; a
    contract that bans title identity simply never references them.)
    """
    from amt.jobs.data.features import CANDIDATES_COLUMN, as_item_list

    if item_col not in train_df.columns and CANDIDATES_COLUMN in train_df.columns:
        import pandas as pd

        train_df = pd.DataFrame(
            {item_col: [i for row in train_df[CANDIDATES_COLUMN] for i in as_item_list(row)]}
        )
    counts = train_df[item_col].value_counts()
    ordered = sorted(counts.index.tolist(), key=lambda i: (-int(counts[i]), str(i)))
    return (
        ordered,
        {str(item): float(counts[item]) for item in ordered},
        {str(item): rank + 1 for rank, item in enumerate(ordered)},
    )


def _attribute_table(
    frames: Sequence[Any], item_col: str, columns: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Last non-null value per item for each requested attribute column."""
    table: dict[str, dict[str, Any]] = {}
    for frame in frames:
        if frame is None or item_col not in getattr(frame, "columns", []):
            continue
        present = [c for c in columns if c in frame.columns]
        if not present:
            continue
        for row in frame[[item_col, *present]].itertuples(index=False):
            key = str(row[0])
            bucket = table.setdefault(key, {})
            for column, value in zip(present, row[1:], strict=True):
                if not _is_missing(value):
                    bucket[column] = value
    return table


def _category_codes(
    attributes: Mapping[str, Mapping[str, Any]], columns: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Stable 1-based codes per categorical column; 0 is reserved for unseen."""
    codes: dict[str, dict[str, int]] = {}
    for column in columns:
        values = {
            str(bucket[column])
            for bucket in attributes.values()
            if column in bucket and not isinstance(bucket[column], (int, float, bool))
        }
        if values:
            codes[column] = {value: i + 1 for i, value in enumerate(sorted(values))}
    return codes


def _train_value_counts(
    recipe: Mapping[str, Any],
    attributes: Mapping[str, Mapping[str, Any]],
    counts: Mapping[str, float],
) -> dict[str, dict[str, float]]:
    """Train-frequency of each attribute value, for ``count(over='train')``."""
    wanted = {
        str(t.get("of"))
        for t in recipe.get("feature_columns") or []
        if str(t.get("kind")) == "count"
        and str(t.get("over")) == "train"
        and t.get("of") != "item_id"
    }
    table: dict[str, dict[str, float]] = {}
    for column in wanted:
        bucket: dict[str, float] = {}
        for item, attrs in attributes.items():
            if column not in attrs:
                continue
            key = str(attrs[column])
            bucket[key] = bucket.get(key, 0.0) + float(counts.get(item, 0.0))
        table[column] = bucket
    return table


def _sequences(train_df, user_col: str, item_col: str, ts_col: str) -> dict[Any, list[Any]]:
    from amt.jobs.data.features import build_sequences

    if ts_col not in train_df.columns:
        return {
            user: group[item_col].tolist()
            for user, group in train_df.groupby(user_col, sort=True)
        }
    return build_sequences(train_df, user_col=user_col, item_col=item_col, ts_col=ts_col)


def grouped_examples(
    frame, max_rows: int
) -> tuple[list[list[Any]], list[list[Any]], list[list[float]]]:
    """``(candidates, positives, labels)`` read off a render-grain split.

    One row per render, carrying the closed candidate set, the incumbent's
    slots and the conversion labels. Nothing is generated and nothing is
    sampled: the candidates are what the container showed.
    """
    candidates: list[list[Any]] = []
    positives: list[list[Any]] = []
    labels: list[list[float]] = []
    for row in frame.itertuples(index=False):
        items = list(getattr(row, "candidates", ()) or ())
        flags = [float(x) for x in (getattr(row, "labels", ()) or ())]
        if len(items) < 2 or len(flags) != len(items):
            continue
        converted = [item for item, flag in zip(items, flags, strict=True) if flag > 0]
        if not converted:
            # A render nobody converted in carries no signal for a ranker and
            # no label for a classifier; the harness excludes it too.
            continue
        candidates.append(items)
        positives.append(converted)
        labels.append(flags)
        if len(candidates) >= int(max_rows):
            break
    return candidates, positives, labels


def _labelled_examples(
    recipe: Mapping[str, Any],
    train_df,
    sequences: Mapping[Any, list[Any]],
    user_col: str,
    item_col: str,
    ts_col: str,
    max_rows: int,
) -> tuple[list[list[Any]], list[list[Any]]]:
    """``(histories, positives)`` implied by the recipe's label strategy."""
    label = recipe.get("label") or {}
    strategy = str(label.get("strategy") or "holdout_next_item")

    histories: list[list[Any]] = []
    positives: list[list[Any]] = []

    if strategy == "holdout_next_item":
        for _user, items in sequences.items():
            if len(items) < 2:
                continue
            histories.append(items[:-1])
            positives.append([items[-1]])
    elif strategy == "interaction_positive":
        for _user, items in sequences.items():
            if not items:
                continue
            histories.append(items)
            positives.append(list(dict.fromkeys(items)))
    elif strategy == "label_column":
        column = str(label.get("source_column") or "")
        if column not in train_df.columns:
            raise UnsupportedRecipe(
                f"label column {column!r} is not in the train split; "
                "the recipe must name a column the frozen split carries"
            )
        wanted = {str(v) for v in (label.get("positive_values") or [])}
        frame = train_df
        if ts_col in frame.columns:
            frame = frame.sort_values([user_col, ts_col], kind="mergesort")
        for _user, group in frame.groupby(user_col, sort=True):
            items = group[item_col].tolist()
            flags = group[column].tolist()
            hits = [
                items[i]
                for i, flag in enumerate(flags)
                if (str(flag) in wanted if wanted else bool(flag))
            ]
            if not hits:
                continue
            histories.append([i for i in items if i not in set(hits)] or items)
            positives.append(list(dict.fromkeys(hits)))
    else:
        raise UnsupportedRecipe(f"label strategy {strategy!r} is not supported")

    total = 0
    kept_histories: list[list[Any]] = []
    kept_positives: list[list[Any]] = []
    for history, items in zip(histories, positives, strict=True):
        kept_histories.append(history)
        kept_positives.append(items)
        total += len(items)
        if total >= max_rows:
            break
    return kept_histories, kept_positives


def _sample_negatives(
    state: _State,
    history: Sequence[Any],
    positives: Sequence[Any],
    n: int,
    rng: Any,
    exclude_seen: bool,
) -> list[Any]:
    """Popularity-weighted negatives, never a positive (and never a seen item)."""
    if n <= 0 or not state.catalog:
        return []
    forbidden = set(map(str, positives))
    if exclude_seen:
        forbidden |= set(map(str, history or ()))
    pool = [item for item in state.catalog if str(item) not in forbidden]
    if not pool:
        return []
    weights = np.asarray([state.counts.get(str(i), 0.0) + 1.0 for i in pool], dtype=np.float64)
    weights /= weights.sum()
    size = min(int(n), len(pool))
    idx = rng.choice(len(pool), size=size, replace=False, p=weights)
    return [pool[int(i)] for i in np.atleast_1d(idx)]


def _grouped_dataset(
    recipe: Mapping[str, Any],
    state: _State,
    candidates: Sequence[Sequence[Any]],
    labels: Sequence[Sequence[float]],
) -> dict[str, Any] | None:
    """One labelled group per render, over the candidates the container showed.

    No negative sampling: every candidate is a real impression, and the rows
    that did not convert are the negatives. Sampling them would throw away the
    only thing that makes this a re-ranking problem.
    """
    if not candidates:
        return None
    frame, groups, _items = build_rows(recipe, state, candidates, candidates)
    y = np.asarray([label for row in labels for label in row], dtype=np.float64)
    frame, y, groups = _apply_filters(recipe, frame, y, groups)
    if len(frame) == 0:
        return None
    _, sizes = np.unique(groups, return_counts=True)
    return {"X": frame, "y": y, "group_sizes": sizes.tolist()}


def _dataset(
    recipe: Mapping[str, Any],
    state: _State,
    histories: Sequence[Sequence[Any]],
    positives: Sequence[Sequence[Any]],
    negatives_per_positive: int,
    rng: Any,
) -> dict[str, Any] | None:
    """Assemble one labelled, grouped dataset from histories and their positives."""
    exclude_seen = bool((recipe.get("candidate_generation") or {}).get("exclude_seen", True))
    rows_histories: list[list[Any]] = []
    rows_candidates: list[list[Any]] = []
    labels: list[list[float]] = []

    for history, pos in zip(histories, positives, strict=True):
        if not pos:
            continue
        negatives = _sample_negatives(
            state,
            history,
            pos,
            negatives_per_positive * max(1, len(pos)),
            rng,
            exclude_seen,
        )
        items = list(pos) + negatives
        if len(items) < 2:
            continue
        rows_histories.append(list(history))
        rows_candidates.append(items)
        labels.append([1.0] * len(pos) + [0.0] * len(negatives))

    if not rows_candidates:
        return None

    frame, groups, _items = build_rows(recipe, state, rows_histories, rows_candidates)
    y = np.asarray([label for row in labels for label in row], dtype=np.float64)
    frame, y, groups = _apply_filters(recipe, frame, y, groups)
    if len(frame) == 0:
        return None
    _, sizes = np.unique(groups, return_counts=True)
    return {"X": frame, "y": y, "group_sizes": sizes.tolist()}


def _apply_filters(recipe: Mapping[str, Any], frame, y: np.ndarray, groups: np.ndarray):
    """Row filters, applied only to columns the recipe actually produced."""
    rules = [r for r in (recipe.get("filters") or []) if str(r.get("column")) in frame.columns]
    if not rules:
        return frame, y, groups

    keep = np.ones(len(frame), dtype=bool)
    for rule in rules:
        column = frame[str(rule["column"])]
        op = str(rule["op"])
        value = rule.get("value")
        if op == "eq":
            keep &= (column == value).to_numpy()
        elif op == "ne":
            keep &= (column != value).to_numpy()
        elif op == "gt":
            keep &= (column > value).to_numpy()
        elif op == "gte":
            keep &= (column >= value).to_numpy()
        elif op == "lt":
            keep &= (column < value).to_numpy()
        elif op == "lte":
            keep &= (column <= value).to_numpy()
        elif op == "in":
            keep &= column.isin(list(value)).to_numpy()
        elif op == "not_in":
            keep &= (~column.isin(list(value))).to_numpy()
        elif op == "not_null":
            keep &= column.notna().to_numpy()
    return frame.loc[keep].reset_index(drop=True), y[keep], groups[keep]


def _validation_examples(val_df, max_users: int) -> tuple[list[list[Any]], list[list[Any]]]:
    """Histories/targets from the frozen validation split, for early stopping only."""
    from amt.jobs.data.features import build_eval_frame

    if val_df is None or not len(val_df):
        return [], []
    try:
        frame = build_eval_frame(val_df)
    except Exception:  # pragma: no cover - validation shape is contract-defined
        return [], []
    n = min(int(max_users), len(frame.histories))
    return (
        [list(h) for h in frame.histories[:n]],
        [list(t) for t in frame.truths[:n]],
    )


def build_state(
    recipe: Mapping[str, Any],
    train_df,
    item_col: str = "item_id",
    extra_frames: Mapping[str, Any] | None = None,
) -> _State:
    """Everything the recipe needs that is learned from the train split only.

    Catalog, popularity, item attributes and category codes. Kept separate from
    fitting so the same state can be rebuilt for inspection without a booster.
    """
    catalog, counts, ranks = _item_stats(train_df, item_col)
    attribute_columns = _attribute_columns(recipe)
    attributes = _attribute_table(
        [train_df, *list((extra_frames or {}).values())], item_col, attribute_columns
    )
    return _State(
        catalog=catalog,
        counts=counts,
        ranks=ranks,
        attributes=attributes,
        attribute_columns=attribute_columns,
        train_value_counts=_train_value_counts(recipe, attributes, counts),
        category_codes=_category_codes(attributes, attribute_columns),
        feature_names=[str(t["name"]) for t in recipe["feature_columns"]],
    )


def fit_family(
    family: str,
    train_df,
    val_df=None,
    cfg: Mapping[str, Any] | None = None,
    index: Any = None,
    logger: Callable[..., None] | None = None,
    evaluator: Callable[[Any], Mapping[str, float]] | None = None,
    extra_frames: Mapping[str, Any] | None = None,
    **_: Any,
) -> TabularModel:
    """Fit one booster family from the approved recipe. Reads train + validation only."""
    flat = resolve_cfg(cfg)
    recipe = _require_recipe(flat)
    objective_default, metric_default = FAMILY_OBJECTIVES[family]
    ranker = family.endswith("_reranker")

    user_col = str(flat["user_col"])
    item_col = str(flat["item_col"])
    ts_col = str(flat["ts_col"])
    rng = np.random.default_rng(int(flat["seed"]))

    grouped = uses_logged_candidates(recipe)
    state = build_state(recipe, train_df, item_col, extra_frames)
    negatives = int(flat["negative_samples_per_positive"])

    if grouped:
        # Render grain: the candidates, their order and their labels are all on
        # the row, so there is nothing to generate or sample.
        candidates, _positives, labels = grouped_examples(
            train_df, int(flat["max_train_rows"])
        )
        train_set = _grouped_dataset(recipe, state, candidates, labels)
        val_set = None
        if val_df is not None and len(val_df):
            val_candidates, _vp, val_labels = grouped_examples(
                val_df, int(flat["max_validation_users"])
            )
            val_set = _grouped_dataset(recipe, state, val_candidates, val_labels)
    else:
        sequences = _sequences(train_df, user_col, item_col, ts_col)
        histories, positives = _labelled_examples(
            recipe,
            train_df,
            sequences,
            user_col,
            item_col,
            ts_col,
            int(flat["max_train_rows"]),
        )
        train_set = _dataset(recipe, state, histories, positives, negatives, rng)
        val_histories, val_targets = _validation_examples(
            val_df, int(flat["max_validation_users"])
        )
        val_set = (
            _dataset(recipe, state, val_histories, val_targets, negatives, rng)
            if val_histories
            else None
        )

    if train_set is None:
        raise ValueError(
            "the recipe produced no labelled training rows: every group ended up with "
            "fewer than two candidates. Check the label strategy, and whether "
            "candidate_generation.exclude_seen leaves any negatives for this catalog "
            f"({len(state.catalog)} items)."
        )

    params = {
        "objective": str(flat.get("objective") or objective_default),
        "seed": int(flat["seed"]),
        **_booster_params(flat),
    }
    eval_metric = str(flat.get("eval_metric") or metric_default)
    backend = _backend_for(family)
    if backend.name == "xgboost":
        params["eval_metric"] = eval_metric
    else:
        params["metric"] = eval_metric
        params.setdefault("verbosity", -1)

    booster, curve, best_iteration = backend.fit(
        train_set,
        val_set,
        params,
        int(flat["num_boost_round"]),
        int(flat["early_stopping_rounds"]),
        ranker,
    )

    history = _curve_to_history(curve, eval_metric)
    if logger is not None:
        for step, entry in enumerate(history, start=1):
            logger(step=step, metrics=entry)

    model = TabularModel(
        family=family,
        booster=booster,
        recipe=recipe,
        state=state,
        cfg={
            **{k: v for k, v in flat.items() if k != "feature_recipe"},
            "best_epoch": best_iteration,
            "n_train_rows": len(train_set["X"]),
            "n_features": len(state.feature_names),
        },
        history=history,
    )

    # Convergence is judged on the harness metric, never on the booster's own loss.
    if evaluator is not None:
        scores = {f"val/{k}": float(v) for k, v in dict(evaluator(model)).items()}
        model.history.append({"epoch": float(best_iteration), **scores})
        if logger is not None:
            logger(step=max(1, len(history)), metrics=scores)
    return model


def _curve_to_history(
    curve: Mapping[str, Mapping[str, Sequence[float]]], eval_metric: str
) -> list[dict[str, float]]:
    """Flatten the backend's eval history into one dict per boosting round."""
    series: dict[str, Sequence[float]] = {}
    for split, metrics in (curve or {}).items():
        for metric, values in (metrics or {}).items():
            label = "train/loss" if split in {"train", "training"} else f"val/{metric}"
            if split in {"train", "training"} and metric != eval_metric:
                continue
            series[label] = list(values)
    if not series:
        return []
    length = min(len(v) for v in series.values())
    return [
        {"epoch": float(i + 1), **{k: float(v[i]) for k, v in series.items()}}
        for i in range(length)
    ]
