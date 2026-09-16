"""Feature construction shared by training and QA.

The important property here is that training and inference build sequences
the same way: QA phase 0 checks train/inference consistency, and it can only
do that if there is one implementation.  Everything is pandas/numpy so it
runs identically on a Databricks driver and in unit tests.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "EvalFrame",
    "ItemIndex",
    "as_item_list",
    "build_eval_frame",
    "build_item_index",
    "build_sequences",
    "encode",
    "history_stats",
    "pad_left",
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
    def from_dict(cls, payload: dict[str, Any]) -> ItemIndex:
        return cls.from_items(payload.get("items", []))

    @classmethod
    def from_items(cls, items: Iterable[Any]) -> ItemIndex:
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
    """Everything the harness needs for one split, in a fixed unit order.

    A unit is a user in ``next_item`` mode and one render of a container in
    ``grouped_rerank`` mode. The extra fields are populated only in the second:
    ``candidates`` is the closed set the model may rank (what the container
    actually showed), ``positions`` is the incumbent's slot for each of them,
    and ``gains`` is the relevance weight of each converted item under the
    contract's ``relevance_weighting``.
    """

    user_ids: list[Any]
    histories: list[list[int]]
    truths: list[set[int]]
    days_since_last: list[float] | None = None
    candidates: list[list[Any]] | None = None
    positions: list[list[float]] | None = None
    gains: list[dict[Any, float]] | None = None

    def __len__(self) -> int:
        return len(self.user_ids)

    @property
    def grouped(self) -> bool:
        return self.candidates is not None

    @property
    def distinct_candidates(self) -> int:
        """Items that appear in any render -- the catalog `coverage@k` divides by."""
        if self.candidates is None:
            return 0
        seen: set[Any] = set()
        for row in self.candidates:
            seen.update(row)
        return len(seen)


def build_eval_frame(
    df,
    index: ItemIndex | None = None,
    user_col: str = "user_id",
    history_col: str = "history",
    target_col: str = "targets",
    recency_col: str = "days_since_last",
) -> EvalFrame:
    """Turn a materialized validation/QA split into an :class:`EvalFrame`.

    In ``next_item`` mode rows are at user grain and carry ``history`` (items
    up to the split boundary) and ``targets`` (items after it). In
    ``grouped_rerank`` mode rows are at render grain and carry ``candidates``,
    ``positions``, ``labels`` and optionally ``weights``; the truth set is then
    the labelled candidates and the history is the candidate set itself, since
    a re-ranker conditions on what it is ranking. See
    :mod:`amt.jobs.materialize_splits` for both shapes.

    Units are ordered by id so that a candidate and a baseline evaluated in
    separate jobs stay aligned for the paired comparison.

    Ids stay **raw** unless an ``index`` is supplied.  The harness is id
    agnostic, and evaluating in raw id space means a target the model has
    never seen in training counts as a miss (which it is) rather than
    silently vanishing from the truth set -- so train-time and QA-time
    numbers mean the same thing.
    """
    key = user_col if user_col in df.columns else GROUP_ID_COLUMN
    frame = df.sort_values([key], kind="mergesort")
    unit_ids = frame[key].tolist()

    if CANDIDATES_COLUMN in frame.columns:
        return _grouped_eval_frame(frame, unit_ids, index)

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
        user_ids=unit_ids,
        histories=histories,
        truths=truths,
        days_since_last=recency,
    )


#: Column names the `grouped_rerank` split shape uses. Plain strings rather
#: than contract fields: `amt.jobs` may not import the orchestrator's schemas,
#: and materialization writes these names whatever the source columns were.
GROUP_ID_COLUMN = "group_id"
CANDIDATES_COLUMN = "candidates"
POSITIONS_COLUMN = "positions"
LABELS_COLUMN = "labels"
WEIGHTS_COLUMN = "weights"


def _grouped_eval_frame(frame, unit_ids: list[Any], index: ItemIndex | None) -> EvalFrame:
    """One render per row: candidates, the incumbent's order, and the labels."""
    candidates = [as_item_list(v) for v in frame[CANDIDATES_COLUMN].tolist()]
    if index is not None:
        candidates = [encode(row, index) for row in candidates]
    positions = [
        [float(p) for p in as_item_list(v)] for v in frame[POSITIONS_COLUMN].tolist()
    ] if POSITIONS_COLUMN in frame.columns else None
    labels = [
        [float(x) for x in as_item_list(v)] for v in frame[LABELS_COLUMN].tolist()
    ]
    weights = (
        [[float(x) for x in as_item_list(v)] for v in frame[WEIGHTS_COLUMN].tolist()]
        if WEIGHTS_COLUMN in frame.columns
        else None
    )

    truths: list[set[Any]] = []
    gains: list[dict[Any, float]] = []
    for row, items in enumerate(candidates):
        flags = labels[row] if row < len(labels) else []
        if len(flags) != len(items):
            # A render whose labels do not line up with its candidates is a
            # malformed split; scoring it on whichever prefix happens to match
            # would be worse than treating it as unscoreable.
            truths.append(set())
            gains.append({})
            continue
        converted = {item for item, flag in zip(items, flags, strict=True) if flag > 0}
        truths.append(converted)
        row_weights = weights[row] if weights is not None and row < len(weights) else None
        gains.append(
            {
                item: float(row_weights[i])
                for i, item in enumerate(items)
                if item in converted and row_weights is not None and i < len(row_weights)
            }
            if row_weights is not None
            else {}
        )

    return EvalFrame(
        user_ids=unit_ids,
        # A re-ranker conditions on the set it is ranking, so the "history" a
        # model is handed is the candidate list itself.
        histories=candidates,
        truths=truths,
        candidates=candidates,
        positions=positions,
        gains=gains if weights is not None else None,
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
