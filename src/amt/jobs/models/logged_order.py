"""The incumbent: the order the container actually rendered.

In ``grouped_rerank`` mode the baseline is not a model AMT fits. It is the
ranking the production system already chose, read back off the impression log.
That makes it the only honest reference point for "does a new ranker beat what
is live" -- beating a popularity baseline says nothing, because popularity is
not what users saw.

There is nothing to learn, so ``fit`` reads no data at all: the ranking is
carried on every group row as the logged slot of each candidate. Registered
like any other family, so ``run_baseline`` and the pyfunc wrapper are
untouched.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from typing import Any

FAMILY = "logged_order"

__all__ = ["FAMILY", "LoggedOrderModel", "fit", "load", "log_model"]


class LoggedOrderModel:
    """Rank each group's candidates by the position they were shown in."""

    family = FAMILY

    def __init__(self, cfg: Mapping[str, Any] | None = None) -> None:
        self.cfg: dict[str, Any] = dict(cfg or {})

    @property
    def catalog_size(self) -> int:
        # The candidate set is per render, so there is no catalog to cover.
        # `coverage@k` is computed by the harness from the split instead.
        return 0

    def recommend(
        self,
        histories: Sequence[Sequence[Any]],
        k: int = 10,
        candidates: Sequence[Sequence[Any]] | None = None,
        positions: Sequence[Sequence[float]] | None = None,
        contexts: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[list[Any]]:
        """Top-``k`` candidates per group, in the incumbent's own order.

        ``positions`` is the logged slot of each candidate. Without it the
        candidate list is already assumed to be in render order, which is how
        materialization writes it -- the argument exists so a caller that
        reordered the candidates can still recover the truth. ``contexts`` is
        accepted for interface symmetry with the learned models and ignored:
        the incumbent's order does not depend on anything we could hand it.
        """
        k = int(k)
        groups = candidates if candidates is not None else histories
        out: list[list[Any]] = []
        for row, items in enumerate(groups):
            items = list(items or ())
            if positions is not None and row < len(positions):
                slots = list(positions[row] or ())
                if len(slots) == len(items):
                    items = [
                        item
                        for _slot, item in sorted(
                            zip(slots, items, strict=True),
                            key=lambda pair: (_as_float(pair[0]), str(pair[1])),
                        )
                    ]
            out.append(items[:k])
        return out

    # -- persistence ------------------------------------------------------

    def save(self, path: str) -> str:
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"family": FAMILY, "cfg": self.cfg}, fh, default=str)
        return path

    @classmethod
    def load(cls, path: str) -> LoggedOrderModel:
        meta_path = os.path.join(path, "meta.json")
        cfg: dict[str, Any] = {}
        if os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                cfg = json.load(fh).get("cfg", {})
        return cls(cfg)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("inf")


def fit(
    train_df=None,
    val_df=None,
    cfg: Mapping[str, Any] | None = None,
    logger: Any = None,
    **_: Any,
) -> LoggedOrderModel:
    """Return the incumbent. Reads nothing: the order is already in the log."""
    model = LoggedOrderModel(cfg)
    if logger is not None:
        # One "epoch" so the curve plumbing is uniform across families.
        logger(step=0, metrics={"train/rows": 0.0})
    return model


def load(path: str) -> LoggedOrderModel:
    return LoggedOrderModel.load(path)


def log_model(model: LoggedOrderModel, artifact_path: str = "model", **kwargs) -> str:
    from amt.jobs.models.registry import log_pyfunc_model

    return log_pyfunc_model(model, artifact_path=artifact_path, **kwargs)
