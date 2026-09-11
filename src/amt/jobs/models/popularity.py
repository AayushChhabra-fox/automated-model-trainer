"""Popularity baseline.

The SPEC requires every campaign to have a baseline evaluated through the
*same* EvaluationContract as the candidates.  Ranking by train-set item
frequency, excluding items the user has already seen, is the cheapest
defensible one and is deliberately boring: it has no hyperparameters to
tune, so it cannot be accidentally optimised against validation.

Works entirely in raw item-id space (see ``build_eval_frame``).
"""

from __future__ import annotations

import json
import os
from typing import Any, Mapping, Sequence

import numpy as np

FAMILY = "popularity"

__all__ = ["FAMILY", "PopularityModel", "fit", "load", "log_model"]


class PopularityModel:
    """Rank by train frequency; never re-recommend an item in the history."""

    family = FAMILY

    def __init__(
        self,
        items: Sequence[Any],
        counts: Sequence[float],
        cfg: Mapping[str, Any] | None = None,
    ) -> None:
        order = np.argsort(-np.asarray(counts, dtype=np.float64), kind="stable")
        self.items: list[Any] = [items[i] for i in order]
        self.counts: list[float] = [float(counts[i]) for i in order]
        self.cfg: dict[str, Any] = dict(cfg or {})
        self._score = {item: c for item, c in zip(self.items, self.counts)}

    # -- inference --------------------------------------------------------

    @property
    def catalog_size(self) -> int:
        return len(self.items)

    def score(self, item: Any) -> float:
        return self._score.get(item, 0.0)

    def recommend(
        self, histories: Sequence[Sequence[Any]], k: int = 10
    ) -> list[list[Any]]:
        """Top-``k`` unseen items per user, in descending popularity."""
        k = int(k)
        out: list[list[Any]] = []
        for history in histories:
            seen = set(history or ())
            picks: list[Any] = []
            for item in self.items:
                if item in seen:
                    continue
                picks.append(item)
                if len(picks) >= k:
                    break
            out.append(picks)
        return out

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"family": FAMILY, "cfg": self.cfg}, fh)
        with open(os.path.join(path, "state.json"), "w", encoding="utf-8") as fh:
            json.dump({"items": self.items, "counts": self.counts}, fh, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> "PopularityModel":
        with open(os.path.join(path, "state.json"), encoding="utf-8") as fh:
            state = json.load(fh)
        cfg: dict[str, Any] = {}
        meta_path = os.path.join(path, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                cfg = json.load(fh).get("cfg", {})
        return cls(state["items"], state["counts"], cfg)


def fit(
    train_df,
    val_df=None,
    cfg: Mapping[str, Any] | None = None,
    logger: Any = None,
    **_: Any,
) -> PopularityModel:
    """Fit from the frozen train split only.

    ``val_df`` is accepted for interface symmetry with
    :func:`amt.jobs.models.sasrec.fit` but is never read: a baseline that
    peeked at validation would not be a clean reference point.
    """
    cfg = dict(cfg or {})
    item_col = cfg.get("item_col", "item_id")
    min_count = int(cfg.get("min_count", 1))

    counts = train_df[item_col].value_counts()
    counts = counts[counts >= min_count]
    model = PopularityModel(
        items=counts.index.tolist(),
        counts=counts.to_numpy(dtype=float).tolist(),
        cfg=cfg,
    )
    if logger is not None:
        # One "epoch" so the curve plumbing is uniform across families.
        logger(step=0, metrics={"train/catalog_size": float(model.catalog_size)})
    return model


def load(path: str) -> PopularityModel:
    return PopularityModel.load(path)


def log_model(model: PopularityModel, artifact_path: str = "model", **kwargs) -> str:
    from amt.jobs.models.registry import log_pyfunc_model

    return log_pyfunc_model(model, artifact_path=artifact_path, **kwargs)
