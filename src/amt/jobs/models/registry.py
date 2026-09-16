"""family -> builder registry, plus the single MLflow pyfunc wrapper.

Adding a model family means adding a module with ``fit`` / ``load`` and one
entry in :data:`FAMILIES`.  The trainer agent selects a family by name in the
``ExperimentConfig``; it does not generate a training system (SPEC stage 4).

The pyfunc wrapper is shared by every family so that QA phase 0 has exactly
one serialization/inference path to validate, whatever the candidate is.
"""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from typing import Any

__all__ = [
    "FAMILIES",
    "PYFUNC_INPUT_COLUMNS",
    "RECIPE_FAMILIES",
    "REQUIRES_PIP",
    "RecommenderPyfunc",
    "UnknownFamily",
    "fit_model",
    "get_builder",
    "get_module",
    "load_model",
    "log_pyfunc_model",
    "pyfunc_signature",
    "recommend_with_pyfunc",
    "requires_pip",
    "save_model",
]

FAMILIES: dict[str, str] = {
    "popularity": "amt.jobs.models.popularity",
    "sasrec": "amt.jobs.models.sasrec",
    "xgboost_classifier": "amt.jobs.models.xgboost_classifier",
    "xgboost_reranker": "amt.jobs.models.xgboost_reranker",
    "lightgbm_classifier": "amt.jobs.models.lightgbm_classifier",
    "lightgbm_reranker": "amt.jobs.models.lightgbm_reranker",
}

# Families whose input shape is declared by an approved FeatureRecipe rather than
# fixed in code. Mirrors ``amt.schemas.feature_recipe.TABULAR_FAMILIES``, which the
# orchestrator uses; the two are kept in step by a unit test (jobs code must not
# import the orchestrator's schemas).
RECIPE_FAMILIES: frozenset[str] = frozenset(
    {
        "xgboost_classifier",
        "xgboost_reranker",
        "lightgbm_classifier",
        "lightgbm_reranker",
    }
)

PYFUNC_INPUT_COLUMNS = ("history", "k")

# PyPI packages a family needs that the runtime image is not guaranteed to carry.
# The executor passes these as `extra_libraries` on the job submission, so a fresh
# job cluster installs the family's backend per run instead of someone having to
# prepare a long-lived cluster by hand. `sasrec` is absent because `torch` ships
# with every ML runtime; `popularity` needs nothing beyond pandas.
REQUIRES_PIP: dict[str, tuple[str, ...]] = {
    "xgboost_classifier": ("xgboost",),
    "xgboost_reranker": ("xgboost",),
    "lightgbm_classifier": ("lightgbm",),
    "lightgbm_reranker": ("lightgbm",),
}


class UnknownFamily(KeyError):
    """Raised when an ExperimentConfig names a family that does not exist."""


def requires_pip(family: str) -> tuple[str, ...]:
    """Packages to install alongside the wheel for ``family``.

    Deliberately tolerant of an unknown or empty name: the caller is the
    executor deciding what to install, and a bad family must fail on the
    cluster with :class:`UnknownFamily` rather than here, where the message
    would be about libraries.
    """
    return REQUIRES_PIP.get(str(family).strip().lower(), ())


def _normalize(family: str) -> str:
    key = str(family).strip().lower()
    if key not in FAMILIES:
        raise UnknownFamily(
            f"unknown model family {family!r}; known: {sorted(FAMILIES)}"
        )
    return key


def get_module(family: str):
    """Import a family module lazily (so torch is only needed for sasrec)."""
    return importlib.import_module(FAMILIES[_normalize(family)])


def get_builder(family: str) -> Callable[..., Any]:
    """Return the family's ``fit(train_df, val_df, cfg) -> model`` callable."""
    return get_module(family).fit


def fit_model(
    family: str,
    train_df,
    val_df=None,
    cfg: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    return get_builder(family)(train_df, val_df, cfg or {}, **kwargs)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def save_model(model: Any, path: str) -> str:
    os.makedirs(path, exist_ok=True)
    model.save(path)
    # save() writes meta.json, but be defensive: the family must be recoverable.
    meta_path = os.path.join(path, "meta.json")
    meta: dict[str, Any] = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
    meta.setdefault("family", getattr(model, "family", "unknown"))
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return path


def load_model(path: str) -> Any:
    """Load a saved model directory, dispatching on the recorded family."""
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    return get_module(meta["family"]).load(path)


# --------------------------------------------------------------------------
# MLflow pyfunc
# --------------------------------------------------------------------------

try:  # pragma: no cover - exercised only where mlflow is installed
    from mlflow.pyfunc import PythonModel as _PythonModel
except Exception:  # pragma: no cover
    _PythonModel = object  # type: ignore[assignment,misc]


class RecommenderPyfunc(_PythonModel):  # type: ignore[valid-type,misc]
    """Serving wrapper: a DataFrame of histories in, top-K lists out.

    Input DataFrame columns:
      * ``history`` -- the user's past item ids, as a list or a JSON array
        string (the JSON form is what the logged signature declares, so the
        model is servable over REST);
      * ``k`` -- optional, per-row top-K (defaults to 10).

    Output: a DataFrame with one ``items`` column holding the ranked item ids
    -- JSON strings when the input used JSON strings, plain lists otherwise.
    """

    def __init__(self) -> None:
        self._model: Any = None

    def load_context(self, context) -> None:  # pragma: no cover - mlflow hook
        self._model = load_model(context.artifacts["state"])

    @property
    def model(self) -> Any:
        if self._model is None:
            raise RuntimeError("pyfunc context not loaded")
        return self._model

    def predict(self, context, model_input, params=None):
        from amt.jobs.data.features import as_item_list

        if self._model is None and context is not None:
            self.load_context(context)

        default_k = 10
        if isinstance(params, Mapping) and params.get("k"):
            default_k = int(params["k"])

        raw_histories, ks, json_mode = _extract_histories(model_input, default_k)
        histories = [as_item_list(h) for h in raw_histories]

        k = max(ks) if ks else default_k
        ranked = self.model.recommend(histories, k)
        trimmed = [list(r)[: ks[i]] for i, r in enumerate(ranked)]

        import pandas as pd

        if json_mode:
            return pd.DataFrame(
                {"items": [json.dumps(r, default=str) for r in trimmed]}
            )
        return pd.DataFrame({"items": trimmed})


def _extract_histories(model_input, default_k: int):
    """Normalize the accepted pyfunc inputs into (histories, ks, json_mode)."""
    if hasattr(model_input, "columns"):
        cols = list(model_input.columns)
        hist_col = "history" if "history" in cols else cols[0]
        raw = model_input[hist_col].tolist()
        if "k" in cols:
            ks = [int(v) if v is not None else default_k for v in model_input["k"]]
        else:
            ks = [default_k] * len(raw)
    else:
        raw = list(model_input)
        ks = [default_k] * len(raw)
    json_mode = bool(raw) and isinstance(raw[0], str)
    return raw, ks, json_mode


def pyfunc_signature():
    """Declared signature: ``(history: string, k: long) -> items: string``."""
    try:
        from mlflow.models.signature import ModelSignature
        from mlflow.types.schema import ColSpec, Schema
    except Exception:  # pragma: no cover
        return None
    return ModelSignature(
        inputs=Schema([ColSpec("string", "history"), ColSpec("long", "k")]),
        outputs=Schema([ColSpec("string", "items")]),
    )


def log_pyfunc_model(
    model: Any,
    artifact_path: str = "model",
    metadata: Mapping[str, Any] | None = None,
    input_example=None,
) -> str:
    """Log ``model`` as an MLflow pyfunc and return its ``runs:/`` URI."""
    import mlflow

    with tempfile.TemporaryDirectory() as tmp:
        state_dir = os.path.join(tmp, "state")
        save_model(model, state_dir)
        kwargs: dict[str, Any] = {
            "python_model": RecommenderPyfunc(),
            "artifacts": {"state": state_dir},
            "signature": pyfunc_signature(),
            "metadata": dict(metadata or {}),
        }
        if input_example is not None:
            kwargs["input_example"] = input_example
        try:
            info = mlflow.pyfunc.log_model(artifact_path=artifact_path, **kwargs)
        except TypeError:  # mlflow >= 3 renamed artifact_path -> name
            info = mlflow.pyfunc.log_model(name=artifact_path, **kwargs)

    uri = getattr(info, "model_uri", None)
    if uri:
        return uri
    run = mlflow.active_run()
    return f"runs:/{run.info.run_id}/{artifact_path}"


def recommend_with_pyfunc(
    pyfunc_model: Any, histories: Sequence[Sequence[Any]], k: int = 10
) -> list[list[Any]]:
    """Call a loaded pyfunc model through its declared serving interface.

    QA uses this rather than the trainer's in-memory object: same math,
    different code path (PLAN §6, "QA independence").
    """
    import pandas as pd

    frame = pd.DataFrame(
        {
            "history": [json.dumps(list(h), default=str) for h in histories],
            "k": [int(k)] * len(histories),
        }
    )
    try:
        out = pyfunc_model.predict(frame)
    except TypeError:
        # A bare RecommenderPyfunc (not wrapped by MLflow) takes the context.
        out = pyfunc_model.predict(None, frame)
    if hasattr(out, "columns"):
        values = out[out.columns[0]].tolist()
    else:
        values = list(out)
    return [json.loads(v) if isinstance(v, str) else list(v) for v in values]
