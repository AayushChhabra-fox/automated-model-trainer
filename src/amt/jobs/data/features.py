"""Feature construction shared by training and QA.

The important property here is that training and inference build sequences
the same way: QA phase 0 checks train/inference consistency, and it can only
do that if there is one implementation.  Everything is pandas/numpy so it
runs identically on a Databricks driver and in unit tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "ItemIndex",
    "build_item_index",
    "build_sequences",
    "as_item_list",
    "encode",
    "pad_left",
    "EvalFrame",
    "build_eval_frame",
    "history_stats",
]

PAD_ID = 0  # reserved; encoded item ids start at 1


@dataclass
class ItemIndex:
    """Bidirectional map between raw item ids and contiguous model ids.

    Model id ``0`` is reserved for padding, so ``size`` (the embedding table
    size) is ``len(items) + 1`` while ``catalog_size`` -- what the coverage
    metric divides by -- is ``len(items)``.
    """

    items: list[Any] = field(default_factory=list)
    item_to_idx: dict[Any, int] = field(default_factory=dict)

    @property
    def catalog_size(self) -> int:
        return len(self.items)

    @property
    def size(self) -> int:
        return len(self.items) + 1

    def encode(self, item: Any, default: int = PAD_ID) -> int:
        return self.item_to_idx.get(item, default)

    def decode(self, idx: int) -> Any:
        if idx <= 0 or idx > len(self.items):
            return None
        return self.items[idx - 1]

    def to_dict(self) -> dict[str, Any]:
        return {"items": list(self.items)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ItemIndex":
        return cls.from_items(payload.get("items", []))

    @classmethod
    def from_items(cls, items: Iterable[Any]) -> "ItemIndex":
        ordered = list(items)
        return cls(
            items=ordered,
            item_to_idx={item: i + 1 for i, item in enumerate(ordered)},
        )


def build_item_index(
    train_df,
    item_col: str = "item_id",
    min_count: int = 1,
) -> ItemIndex:
    """Index every item that occurs at least ``min_count`` times in train.

    Items are ordered by descending frequency then by id, so the index is a
    deterministic function of the frozen train split (reproducibility).
    """
    counts = train_df[item_col].value_counts()
    counts = counts[counts >= int(min_count)]
    ordered = sorted(counts.index.tolist(), key=lambda i: (-int(counts[i]), str(i)))
    return ItemIndex.from_items(ordered)


def build_sequences(
    df,
    user_col: str = "user_id",
    item_col: str = "item_id",
    ts_col: str = "ts",
) -> dict[Any, list[Any]]:
    """Per-user chronologically ordered item sequences from interaction rows."""
    ordered = df.sort_values([user_col, ts_col], kind="mergesort")
    return {
        user: group[item_col].tolist()
        for user, group in ordered.groupby(user_col, sort=True)
    }


def as_item_list(value: Any) -> list[Any]:
    """Coerce a stored history/targets cell into a plain list.

    Delta arrays arrive as lists or numpy arrays; the local parquet and the
    pyfunc signature paths carry JSON strings.  Both are accepted.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [p for p in text.split(",") if p]
        return list(parsed) if isinstance(parsed, (list, tuple)) else [parsed]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, float) and np.isnan(value):
        return []
    return [value]


def encode(items: Iterable[Any], index: ItemIndex, drop_unknown: bool = True) -> list[int]:
    """Map raw item ids to model ids, dropping ids unseen in train."""
    out = []
    for item in items:
        idx = index.encode(item)
        if idx == PAD_ID and drop_unknown:
            continue
        out.append(idx)
    return out


def pad_left(seq: Sequence[int], maxlen: int, pad: int = PAD_ID) -> list[int]:
    """Left-pad/truncate to ``maxlen``, keeping the most recent items."""
    trimmed = list(seq)[-maxlen:]
    return [pad] * (maxlen - len(trimmed)) + trimmed


@dataclass
class EvalFrame:
    """Everything the harness needs for one split, in a fixed user order."""

    user_ids: list[Any]
    histories: list[list[int]]
    truths: list[set[int]]
    days_since_last: list[float] | None = None

    def __len__(self) -> int:
        return len(self.user_ids)


def build_eval_frame(
    df,
    index: ItemIndex | None = None,
    user_col: str = "user_id",
    history_col: str = "history",
    target_col: str = "targets",
    recency_col: str = "days_since_last",
) -> EvalFrame:
    """Turn a materialized validation/QA split into an :class:`EvalFrame`.

    Rows are at user grain and carry ``history`` (items up to the split
    boundary) and ``targets`` (items after it) -- see
    :mod:`amt.jobs.materialize_splits`.  Users are ordered by id so that a
    candidate and a baseline evaluated in separate jobs stay aligned for the
    paired comparison.

    Ids stay **raw** unless an ``index`` is supplied.  The harness is id
    agnostic, and evaluating in raw id space means a target the model has
    never seen in training counts as a miss (which it is) rather than
    silently vanishing from the truth set -- so train-time and QA-time
    numbers mean the same thing.
    """
    frame = df.sort_values([user_col], kind="mergesort")
    user_ids = frame[user_col].tolist()
    if index is None:
        histories = [as_item_list(v) for v in frame[history_col].tolist()]
        truths = [set(as_item_list(v)) for v in frame[target_col].tolist()]
    else:
        histories = [
            encode(as_item_list(v), index) for v in frame[history_col].tolist()
        ]
        truths = [
            set(encode(as_item_list(v), index)) for v in frame[target_col].tolist()
        ]
    recency = None
    if recency_col in frame.columns:
        recency = [
            float(v) if v is not None and not _isnan(v) else float("nan")
            for v in frame[recency_col].tolist()
        ]
    return EvalFrame(
        user_ids=user_ids,
        histories=histories,
        truths=truths,
        days_since_last=recency,
    )


def _isnan(value: Any) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def history_stats(histories: Sequence[Sequence[int]]) -> dict[str, float]:
    """Summary stats logged as train params for later interpretation."""
    lengths = np.array([len(h) for h in histories], dtype=np.float64)
    if lengths.size == 0:
        return {"n_users": 0.0, "hist_mean": 0.0, "hist_p50": 0.0, "hist_max": 0.0}
    return {
        "n_users": float(lengths.size),
        "hist_mean": float(lengths.mean()),
        "hist_p50": float(np.percentile(lengths, 50)),
        "hist_max": float(lengths.max()),
    }
