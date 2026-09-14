"""XGBoost pointwise click model.

Scores each candidate independently (``binary:logistic``) and ranks by the
predicted probability.  Everything about the input shape comes from the approved
``feature_recipe``; see :mod:`amt.jobs.models.tabular_common`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from amt.jobs.models import tabular_common

FAMILY = "xgboost_classifier"

__all__ = ["FAMILY", "fit", "load", "log_model", "resolve_cfg"]

resolve_cfg = tabular_common.resolve_cfg


def fit(
    train_df,
    val_df=None,
    cfg: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> tabular_common.TabularModel:
    return tabular_common.fit_family(FAMILY, train_df, val_df, cfg, **kwargs)


def load(path: str) -> tabular_common.TabularModel:
    return tabular_common.load(path)


def log_model(model: Any, artifact_path: str = "model", **kwargs: Any) -> str:
    from amt.jobs.models.registry import log_pyfunc_model

    return log_pyfunc_model(model, artifact_path=artifact_path, **kwargs)
