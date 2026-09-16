"""Materialize the frozen train / validation / QA splits (SPEC stage 2).

This is enforcement mechanism #1 from PLAN §9.  The splits are written once,
as Delta tables, and the EvaluationContract then stores the table name *and*
the Delta version.  Every later read goes through ``SplitGuard`` with
``VERSION AS OF``, so no model in the campaign can be evaluated against a
different population than any other -- even if the source table keeps
receiving data.

Split shapes
------------
``split_train``       interaction grain: ``user_id, item_id, ts``
``split_validation``  user grain: ``user_id, history[], targets[], history_len,
                      days_since_last``
``split_qa``          same shape as validation, one boundary later

``history`` is everything the model is allowed to condition on at that
boundary; ``targets`` is what it must predict.  Building them here (rather
than in each job) is what makes candidate and baseline comparable.

Each name is prefixed with the campaign (``<campaign>_split_train``): campaigns
share a schema, so without it two running at once would overwrite the frozen
population the other is being judged against.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SPLIT_TABLE_NAMES",
    "SplitPlan",
    "materialize",
    "membership_hash",
    "plan_from_contract",
    "run",
    "table_prefix_for",
]

SPLIT_TABLE_NAMES = {
    "train": "split_train",
    "validation": "split_validation",
    "qa": "split_qa",
}


def table_prefix_for(campaign_id: str) -> str:
    """A campaign id as the leading part of a table name.

    Campaigns share one schema and each writes the same three splits, so the
    campaign has to be in the name or two campaigns running at once overwrite
    each other's frozen population.  Identifiers cannot carry the hyphens a
    campaign id usually has, nor start with a digit.
    """
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(campaign_id).strip().lower()).strip("_")
    if not cleaned:
        return ""
    return cleaned if cleaned[0].isalpha() else f"c_{cleaned}"


def membership_hash(user_ids: Sequence[Any]) -> str:
    """sha256 over the sorted user population -- the split's identity.

    Recomputing this later proves the population has not drifted, which is
    the invariance the SPEC demands of the EvaluationContract.
    """
    joined = ",".join(sorted(str(u) for u in user_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


@dataclass
class SplitPlan:
    """The dataset definition, normalized out of the EvaluationContract."""

    source_table: str = ""
    source_path: str = ""
    catalog: str = "main"
    schema: str = "amt"
    user_col: str = "user_id"
    item_col: str = "item_id"
    ts_col: str = "ts"
    filters: list[str] = field(default_factory=list)
    min_interactions_per_user: int = 2
    strategy: str = "temporal"
    train_end: Any = None
    val_end: Any = None
    qa_end: Any = None
    n_val: int = 1
    n_qa: int = 1
    table_prefix: str = ""
    # `grouped_rerank` only.
    evaluation_mode: str = "next_item"
    group_key: list[str] = field(default_factory=list)
    position_column: str = ""
    label_column: str = ""
    positive_values: list[Any] = field(default_factory=list)
    relevance_weighting: str = "none"
    propensity_floor: float = 0.02
    #: Column the propensities are estimated per, when it exists in the source.
    #: Position mechanics differ by device -- a D-pad and a swipe carousel are
    #: not the same rail -- so one propensity table across clients would be an
    #: average of two different behaviours.
    propensity_by: str = "client_name"
    cohort_columns: list[str] = field(default_factory=list)
    min_candidates_per_group: int = 2

    @property
    def grouped(self) -> bool:
        return self.evaluation_mode == "grouped_rerank"

    def table(self, split: str) -> str:
        leaf = SPLIT_TABLE_NAMES[split]
        if self.table_prefix:
            leaf = f"{self.table_prefix}_{leaf}"
        return f"{self.catalog}.{self.schema}.{leaf}"

    def leaf(self, split: str) -> str:
        return self.table(split).split(".")[-1]


def plan_from_contract(contract: Mapping[str, Any]) -> SplitPlan:
    """Read the executable fields of an EvaluationContract into a plan.

    Those are ``dataset`` (source table, column map, filters, eligibility) and
    ``split_policy`` (strategy and its boundaries or per-user counts).  A
    ``dataset.split`` block overrides both, which is how local dry runs point
    the same code at a file without writing a whole contract.
    """
    dataset = dict(contract.get("dataset") or contract.get("dataset_definition") or {})
    policy = dict(contract.get("split_policy") or {})
    override = dict(dataset.get("split") or dataset.get("splits") or {})
    known = set(SplitPlan.__dataclass_fields__)

    kwargs = {k: v for k, v in dataset.items() if k in known}
    columns = dict(dataset.get("columns") or {})
    for field_name, column in (
        ("user_col", columns.get("user")),
        ("item_col", columns.get("item")),
        ("ts_col", columns.get("timestamp")),
    ):
        if column:
            kwargs[field_name] = column
    kwargs.update(
        {
            k: v
            for k, v in (
                ("strategy", policy.get("strategy")),
                ("train_end", policy.get("train_end")),
                ("val_end", policy.get("validation_end")),
                ("qa_end", policy.get("qa_end")),
                ("n_val", policy.get("n_validation")),
                ("n_qa", policy.get("n_qa")),
            )
            if v
        }
    )
    # `grouped_rerank`: the render key, the incumbent's slot and the label.
    label = dict(contract.get("label_spec") or {})
    kwargs.update(
        {
            k: v
            for k, v in (
                ("evaluation_mode", contract.get("evaluation_mode")),
                ("relevance_weighting", contract.get("relevance_weighting")),
                ("propensity_floor", contract.get("propensity_floor")),
                ("group_key", dataset.get("group_key")),
                ("position_column", dataset.get("position_column")),
                ("label_column", label.get("source_column")),
                ("positive_values", label.get("positive_values")),
                ("cohort_columns", contract.get("cohort_columns")),
            )
            if v
        }
    )
    kwargs.update({k: v for k, v in override.items() if k in known})

    plan = SplitPlan(**kwargs)
    plan.filters = list(plan.filters or [])
    if not plan.source_table and not plan.source_path:
        raise ValueError("the contract names no dataset.source_table, so there is nothing to split")
    if plan.grouped and not (plan.group_key and plan.position_column):
        raise ValueError(
            "evaluation_mode 'grouped_rerank' needs dataset.group_key and "
            "dataset.position_column: one render is one evaluation unit"
        )
    return plan


# --------------------------------------------------------------------------
# pandas / local implementation (also the reference semantics)
# --------------------------------------------------------------------------


def _eligible(df, plan: SplitPlan):
    counts = df.groupby(plan.user_col)[plan.item_col].transform("size")
    return df[counts >= int(plan.min_interactions_per_user)]


def _user_frame(history_map, target_map, plan: SplitPlan, boundary):
    """Assemble the user-grain validation/QA rows."""
    import pandas as pd

    rows = []
    for user, targets in sorted(target_map.items(), key=lambda kv: str(kv[0])):
        if not targets:
            continue
        history, last_ts = history_map.get(user, ([], None))
        rows.append(
            {
                plan.user_col: user,
                "history": list(history),
                "targets": list(targets),
                "history_len": len(history),
                "days_since_last": _days_between(last_ts, boundary),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[plan.user_col, "history", "targets", "history_len", "days_since_last"],
    )


def _days_between(last_ts, boundary) -> float:
    if last_ts is None or boundary is None:
        return float("nan")
    try:
        import pandas as pd

        delta = pd.to_datetime(boundary) - pd.to_datetime(last_ts)
        return float(delta.total_seconds()) / 86400.0
    except Exception:
        try:
            return float(boundary) - float(last_ts)
        except (TypeError, ValueError):
            return float("nan")


def _ordered_items(df, plan: SplitPlan):
    """``{user: (items in time order, last ts)}``."""
    out: dict[Any, tuple[list[Any], Any]] = {}
    if len(df) == 0:
        return out
    ordered = df.sort_values([plan.user_col, plan.ts_col], kind="mergesort")
    for user, group in ordered.groupby(plan.user_col, sort=True):
        out[user] = (group[plan.item_col].tolist(), group[plan.ts_col].iloc[-1])
    return out


def _align_timestamps(df, plan: SplitPlan, boundaries):
    """Make the ts column comparable with the contract's boundary literals.

    Local formats (notably CSV, and JSON without ISO dates) can hand back
    timestamps as strings or epoch integers; comparing those against a
    ``'2024-01-05'`` boundary would raise rather than split.
    """
    import pandas as pd

    ts = df[plan.ts_col]
    wants_datetime = any(isinstance(b, str) for b in boundaries if b is not None)
    if wants_datetime and not pd.api.types.is_datetime64_any_dtype(ts):
        df = df.copy()
        unit = "ms" if pd.api.types.is_numeric_dtype(ts) else None
        df[plan.ts_col] = pd.to_datetime(ts, unit=unit) if unit else pd.to_datetime(ts)
        ts = df[plan.ts_col]
    return df, ts


def _temporal_local(df, plan: SplitPlan):
    import pandas as pd

    train_end, val_end, qa_end = plan.train_end, plan.val_end, plan.qa_end
    if train_end is None or val_end is None:
        raise ValueError("temporal split requires train_end and val_end")

    df, ts = _align_timestamps(df, plan, (train_end, val_end, qa_end))
    if qa_end is None:
        qa_end = ts.max()

    def _cast(bound):
        if bound is None:
            return None
        return pd.to_datetime(bound) if pd.api.types.is_datetime64_any_dtype(ts) else bound

    train_end, val_end, qa_end = _cast(train_end), _cast(val_end), _cast(qa_end)

    train = df[ts <= train_end]
    val_window = df[(ts > train_end) & (ts <= val_end)]
    qa_window = df[(ts > val_end) & (ts <= qa_end)]

    val_hist = _ordered_items(train, plan)
    qa_hist = _ordered_items(df[ts <= val_end], plan)

    validation = _user_frame(
        val_hist,
        {u: items for u, (items, _) in _ordered_items(val_window, plan).items()},
        plan,
        train_end,
    )
    qa = _user_frame(
        qa_hist,
        {u: items for u, (items, _) in _ordered_items(qa_window, plan).items()},
        plan,
        val_end,
    )
    return train.reset_index(drop=True), validation, qa


def _leave_last_n_local(df, plan: SplitPlan):
    """Per user: last ``n_qa`` events are QA, the ``n_val`` before are validation."""
    import pandas as pd

    n_val, n_qa = int(plan.n_val), int(plan.n_qa)
    seqs = _ordered_items(df, plan)
    train_rows, val_map, qa_map, val_hist, qa_hist = [], {}, {}, {}, {}

    ordered = df.sort_values([plan.user_col, plan.ts_col], kind="mergesort")
    for user, group in ordered.groupby(plan.user_col, sort=True):
        items = group[plan.item_col].tolist()
        stamps = group[plan.ts_col].tolist()
        if len(items) < n_val + n_qa + 1:
            train_rows.append(group)
            continue
        cut_qa = len(items) - n_qa
        cut_val = cut_qa - n_val
        train_rows.append(group.iloc[:cut_val])
        val_hist[user] = (items[:cut_val], stamps[cut_val - 1] if cut_val else None)
        qa_hist[user] = (items[:cut_qa], stamps[cut_qa - 1] if cut_qa else None)
        val_map[user] = items[cut_val:cut_qa]
        qa_map[user] = items[cut_qa:]

    train = (
        pd.concat(train_rows).reset_index(drop=True)
        if train_rows
        else df.iloc[0:0].copy()
    )
    del seqs
    return (
        train,
        _user_frame(val_hist, val_map, plan, None),
        _user_frame(qa_hist, qa_map, plan, None),
    )


# --------------------------------------------------------------------------
# grouped re-ranking: one render is one row
# --------------------------------------------------------------------------


def parse_position(value: Any) -> float:
    """The tile's column out of a logged ``item_index``.

    The source records ``"rail_rank,column"`` (and sometimes just the column),
    so the slot is the last element. A value that parses to nothing is
    ``inf``, which sorts it last rather than pretending it was first.
    """
    if value is None:
        return float("inf")
    text = str(value).strip()
    if not text:
        return float("inf")
    try:
        return float(text.split(",")[-1])
    except (TypeError, ValueError):
        return float("inf")


def estimate_propensities(
    rows: Sequence[Mapping[str, Any]],
    floor: float = 0.02,
    by: str = "",
) -> dict[str, float]:
    """Examination propensity of each slot, relative to slot 0.

    A position-based model: ``theta_slot = rate_slot / rate_0``, where a rate
    is the share of impressions at that slot that converted. Estimated from
    **train only**, so the weights are frozen ground truth rather than
    something derived from the split being judged.

    This corrects for *where* an item was shown. It does not correct for
    *which* items the incumbent chose to show -- there is no logged
    randomisation to support that, and the report says so.
    """
    totals: dict[tuple[str, float], list[float]] = {}
    for row in rows:
        segment = str(row.get(by) or "") if by else ""
        slot = float(row["slot"])
        bucket = totals.setdefault((segment, slot), [0.0, 0.0])
        bucket[0] += 1.0
        bucket[1] += 1.0 if row["label"] else 0.0

    rates = {
        key: (converted / impressions)
        for key, (impressions, converted) in totals.items()
        if impressions > 0
    }
    reference = {
        segment: rate for (segment, slot), rate in rates.items() if slot == 0.0
    }

    out: dict[str, float] = {}
    for (segment, slot), rate in sorted(rates.items()):
        base = reference.get(segment, 0.0)
        # Without a slot-0 rate to divide by there is no reference examination
        # probability, so the honest answer is "no correction" rather than a
        # ratio against an arbitrary denominator.
        theta = 1.0 if base <= 0 else rate / base
        out[f"{segment}|{slot:g}"] = float(max(min(theta, 1.0), float(floor)))
    return out


def relevance_weight(
    propensities: Mapping[str, float], segment: str, slot: float, floor: float
) -> float:
    """``1 / theta`` for a conversion at ``slot``, capped by the floor."""
    theta = propensities.get(f"{segment}|{slot:g}")
    if theta is None:
        theta = propensities.get(f"|{slot:g}", 1.0)
    return 1.0 / max(float(theta), float(floor))


def _grouped_rows(df, plan: SplitPlan, propensities: Mapping[str, float] | None):
    """Collapse impression rows into one row per render.

    Renders with fewer than ``min_candidates_per_group`` distinct candidates
    are dropped: there is nothing to re-rank, so they can only add noise and
    flatter whichever ranker is being scored.
    """
    import pandas as pd

    keys = list(plan.group_key)
    positive = {str(v) for v in (plan.positive_values or [])}
    rows: list[dict[str, Any]] = []

    for key, group in df.groupby(keys, sort=True, dropna=False):
        key_values = key if isinstance(key, tuple) else (key,)
        ordered = group.assign(
            _slot=[parse_position(v) for v in group[plan.position_column]]
        ).sort_values("_slot", kind="mergesort")
        # One impression per item: a render that showed a title twice is one
        # candidate, and it keeps the better (earlier) slot.
        ordered = ordered.drop_duplicates(subset=[plan.item_col], keep="first")
        if len(ordered) < int(plan.min_candidates_per_group):
            continue

        labels = [
            1.0 if (str(v) in positive if positive else bool(v)) else 0.0
            for v in (
                ordered[plan.label_column]
                if plan.label_column in ordered.columns
                else [False] * len(ordered)
            )
        ]
        slots = [float(s) for s in ordered["_slot"]]
        segment = (
            str(ordered[plan.propensity_by].iloc[0])
            if plan.propensity_by in ordered.columns
            else ""
        )
        row: dict[str, Any] = {
            "group_id": "|".join(str(v) for v in key_values),
            "candidates": ordered[plan.item_col].tolist(),
            "positions": slots,
            "labels": labels,
            "user_id": str(ordered[plan.user_col].iloc[0])
            if plan.user_col in ordered.columns
            else "",
            "ts": ordered[plan.ts_col].iloc[0] if plan.ts_col in ordered.columns else None,
        }
        if propensities is not None:
            row["weights"] = [
                relevance_weight(propensities, segment, slot, plan.propensity_floor)
                if label > 0
                else 0.0
                for slot, label in zip(slots, labels, strict=True)
            ]
        for column in plan.cohort_columns:
            if column in ordered.columns:
                row[column] = ordered[column].iloc[0]
        if plan.propensity_by in ordered.columns:
            row[plan.propensity_by] = segment
        rows.append(row)

    columns = ["group_id", "candidates", "positions", "labels", "user_id", "ts"]
    if propensities is not None:
        columns.append("weights")
    extra = [c for c in (*plan.cohort_columns, plan.propensity_by) if c]
    columns += [c for c in dict.fromkeys(extra) if any(c in r for r in rows)]
    return pd.DataFrame(rows, columns=columns)


def _grouped_local(df, plan: SplitPlan):
    """``(train, validation, qa, propensities)`` at render grain."""
    import pandas as pd

    df, ts = _align_timestamps(df, plan, (plan.train_end, plan.val_end, plan.qa_end))
    train_end, val_end = plan.train_end, plan.val_end
    qa_end = plan.qa_end if plan.qa_end is not None else ts.max()
    if train_end is None or val_end is None:
        raise ValueError("a grouped_rerank split requires train_end and validation_end")

    def _cast(bound):
        if bound is None:
            return None
        return pd.to_datetime(bound) if pd.api.types.is_datetime64_any_dtype(ts) else bound

    train_end, val_end, qa_end = _cast(train_end), _cast(val_end), _cast(qa_end)
    windows = {
        "train": df[ts <= train_end],
        "validation": df[(ts > train_end) & (ts <= val_end)],
        "qa": df[(ts > val_end) & (ts <= qa_end)],
    }

    propensities: dict[str, float] | None = None
    if plan.relevance_weighting == "pbm":
        positive = {str(v) for v in (plan.positive_values or [])}
        train = windows["train"]
        propensities = estimate_propensities(
            [
                {
                    "slot": parse_position(slot),
                    "label": (str(label) in positive if positive else bool(label)),
                    plan.propensity_by: segment,
                }
                for slot, label, segment in zip(
                    train[plan.position_column],
                    train[plan.label_column]
                    if plan.label_column in train.columns
                    else [False] * len(train),
                    train[plan.propensity_by]
                    if plan.propensity_by in train.columns
                    else [""] * len(train),
                    strict=True,
                )
            ],
            floor=plan.propensity_floor,
            by=plan.propensity_by,
        )

    frames = {
        name: _grouped_rows(window, plan, propensities)
        for name, window in windows.items()
    }
    return frames["train"], frames["validation"], frames["qa"], (propensities or {})


def _pandas_predicate(predicate: str) -> str:
    """A contract filter in ``DataFrame.query`` syntax.

    Contract filters are SQL, because Spark is what runs them on a cluster.
    Rewriting SQL equality is enough for the dry-run path; a predicate that
    needs more than that simply cannot be checked locally.
    """
    return re.sub(r"(?<![=<>!])=(?!=)", "==", predicate)


def _materialize_local(
    contract: Mapping[str, Any], plan: SplitPlan, output_dir: str
) -> dict[str, Any]:
    import os

    from amt.jobs.data.loaders import LOCAL_FORMATS, read_local_table, write_local_table

    path = plan.source_path
    if not path:
        leaf = plan.source_table.split(".")[-1]
        for ext in LOCAL_FORMATS:
            cand = os.path.join(output_dir, f"{leaf}{ext}")
            if os.path.exists(cand):
                path = cand
                break
    if not path:
        raise FileNotFoundError(
            f"no local source table for {plan.source_table!r} under {output_dir}"
        )

    df = read_local_table(path)
    for predicate in plan.filters:
        df = df.query(_pandas_predicate(predicate))

    propensities: dict[str, float] = {}
    if plan.grouped:
        # No per-user eligibility: the unit is a render, and a render is
        # eligible when it showed something worth re-ordering.
        train, validation, qa, propensities = _grouped_local(df, plan)
        identity = "group_id"
    else:
        df = _eligible(df, plan)
        if plan.strategy == "leave_last_n":
            train, validation, qa = _leave_last_n_local(df, plan)
        else:
            train, validation, qa = _temporal_local(df, plan)
        identity = plan.user_col

    ext = ".parquet" if _parquet_available() else ".jsonl"
    refs: dict[str, Any] = {}
    for name, frame in (("train", train), ("validation", validation), ("qa", qa)):
        out_path = os.path.join(output_dir, f"{plan.leaf(name)}{ext}")
        write_local_table(frame, out_path)
        refs[name] = {
            "name": name,
            "table": plan.table(name),
            "path": out_path,
            "delta_version": 0,
            "row_count": len(frame),
            "membership_hash": membership_hash(
                frame[identity].tolist() if identity in frame.columns else []
            ),
        }
    return _result(contract, plan, refs, backend="local", propensities=propensities)


def _parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401

        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401

            return True
        except Exception:
            return False


# --------------------------------------------------------------------------
# Spark / Unity Catalog implementation
# --------------------------------------------------------------------------


def _where(plan: SplitPlan) -> str:
    return " AND ".join(f"({f})" for f in plan.filters) if plan.filters else "TRUE"


def _base_cte(plan: SplitPlan) -> str:
    u, i, t = plan.user_col, plan.item_col, plan.ts_col
    return (
        f"base AS (SELECT {u} AS user_id, {i} AS item_id, {t} AS ts "
        f"FROM {plan.source_table} WHERE {_where(plan)} "
        f"AND {u} IS NOT NULL AND {i} IS NOT NULL AND {t} IS NOT NULL), "
        f"elig AS (SELECT user_id FROM base GROUP BY user_id "
        f"HAVING count(*) >= {int(plan.min_interactions_per_user)}), "
        "f AS (SELECT b.* FROM base b JOIN elig e ON b.user_id = e.user_id)"
    )


_SEQ = "transform(array_sort(collect_list(struct(ts, item_id))), x -> x.item_id)"


def _user_grain_sql(plan: SplitPlan, hist_upper: str, lo: str, hi: str) -> str:
    return (
        f"WITH {_base_cte(plan)}, "
        f"hist AS (SELECT user_id, {_SEQ} AS history, max(ts) AS last_ts "
        f"FROM f WHERE ts <= {hist_upper} GROUP BY user_id), "
        f"tgt AS (SELECT user_id, {_SEQ} AS targets "
        f"FROM f WHERE ts > {lo} AND ts <= {hi} GROUP BY user_id) "
        "SELECT t.user_id, coalesce(h.history, array()) AS history, t.targets, "
        "size(coalesce(h.history, array())) AS history_len, "
        f"datediff({hist_upper}, h.last_ts) AS days_since_last "
        "FROM tgt t LEFT JOIN hist h ON t.user_id = h.user_id "
        "WHERE size(t.targets) > 0"
    )


def _ensure_schema(spark: Any, plan: SplitPlan) -> None:
    """Create the output schema only when it is actually missing.

    Unity Catalog checks CREATE SCHEMA on the catalog even for
    ``IF NOT EXISTS``, so an unconditional create fails for the common case of
    a user who owns one schema but cannot create schemas in the catalog.
    """
    try:
        existing = spark.sql(f"SHOW SCHEMAS IN {plan.catalog} LIKE '{plan.schema}'").collect()
    except Exception:
        existing = []
    if not existing:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {plan.catalog}.{plan.schema}")


def _slot_expr(plan: SplitPlan) -> str:
    """The tile's column out of ``item_index``; mirrors :func:`parse_position`."""
    column = plan.position_column
    return (
        f"try_cast(element_at(split(cast({column} AS string), ','), -1) AS DOUBLE)"
    )


def _label_expr(plan: SplitPlan) -> str:
    column = plan.label_column
    if not column:
        return "FALSE"
    if not plan.positive_values:
        return f"coalesce(cast({column} AS BOOLEAN), FALSE)"
    values = ", ".join(_lit(str(v)) for v in plan.positive_values)
    return f"(cast({column} AS string) IN ({values}))"


def _grouped_base_cte(plan: SplitPlan) -> str:
    """Impression rows, one per (render, item), with the slot and label resolved."""
    extra = [c for c in dict.fromkeys([*plan.cohort_columns, plan.propensity_by]) if c]
    carried = "".join(f", {c}" for c in extra)
    return (
        "base AS (SELECT concat_ws('|', "
        + ", ".join(f"cast({k} AS string)" for k in plan.group_key)
        + ") AS group_id, "
        f"{plan.item_col} AS item_id, {plan.ts_col} AS ts, "
        f"{plan.user_col} AS user_id, "
        f"{_slot_expr(plan)} AS slot, "
        f"cast({_label_expr(plan)} AS DOUBLE) AS label{carried} "
        f"FROM {plan.source_table} WHERE {_where(plan)} "
        f"AND {plan.item_col} IS NOT NULL AND {plan.ts_col} IS NOT NULL "
        f"AND {plan.position_column} IS NOT NULL), "
        # One impression per item per render, keeping the better slot: a rail
        # that showed a title twice offers one candidate, not two.
        "dedup AS (SELECT * FROM (SELECT *, row_number() OVER ("
        "PARTITION BY group_id, item_id ORDER BY slot) AS rn FROM base) WHERE rn = 1), "
        f"grp AS (SELECT group_id, min(ts) AS ts, min(user_id) AS user_id{carried}, "
        "transform(array_sort(collect_list(struct(slot, item_id, label)), "
        "(l, r) -> CASE WHEN l.slot < r.slot THEN -1 WHEN l.slot > r.slot THEN 1 "
        "ELSE 0 END), x -> x.item_id) AS candidates, "
        "transform(array_sort(collect_list(struct(slot, item_id, label)), "
        "(l, r) -> CASE WHEN l.slot < r.slot THEN -1 WHEN l.slot > r.slot THEN 1 "
        "ELSE 0 END), x -> x.slot) AS positions, "
        "transform(array_sort(collect_list(struct(slot, item_id, label)), "
        "(l, r) -> CASE WHEN l.slot < r.slot THEN -1 WHEN l.slot > r.slot THEN 1 "
        "ELSE 0 END), x -> x.label) AS labels "
        f"FROM dedup GROUP BY group_id{carried} "
        f"HAVING size(candidates) >= {int(plan.min_candidates_per_group)})"
    )


def _weights_expr(plan: SplitPlan, propensities: Mapping[str, float]) -> str:
    """``labels`` turned into gains by the frozen per-slot propensities.

    Built as a literal CASE over the estimates rather than recomputed per
    split, because the weighting has to be identical for train, validation and
    QA -- it is part of the contract, not of the split.
    """
    if plan.relevance_weighting != "pbm" or not propensities:
        return ""
    segment = (
        f"cast({plan.propensity_by} AS string)" if plan.propensity_by else "''"
    )
    branches = []
    for key, theta in sorted(propensities.items()):
        seg, _, slot = key.rpartition("|")
        weight = 1.0 / max(float(theta), float(plan.propensity_floor))
        branches.append(
            f"WHEN {segment} = {_lit(seg)} AND p = {slot} THEN {weight:.6f}"
        )
    case = "CASE " + " ".join(branches) + " ELSE 1.0 END"
    return (
        ", zip_with(labels, positions, (l, p) -> "
        f"CASE WHEN l > 0 THEN {case} ELSE 0.0 END) AS weights"
    )


def _estimate_propensities_spark(plan: SplitPlan, spark: Any) -> dict[str, float]:
    """Per-slot conversion rates on the *train* window, as ratios to slot 0."""
    segment = plan.propensity_by if plan.propensity_by else "''"
    rows = spark.sql(
        f"WITH {_grouped_base_cte(plan)} "
        f"SELECT cast({segment} AS string) AS segment, slot, "
        "count(*) AS impressions, sum(label) AS converted "
        f"FROM dedup WHERE ts <= {_lit(plan.train_end)} "
        "GROUP BY 1, 2"
    ).collect()
    return estimate_propensities(
        _expand_rates(rows, plan.propensity_by),
        floor=plan.propensity_floor,
        by=plan.propensity_by,
    )


def _expand_rates(rows: Sequence[Any], by: str) -> list[dict[str, Any]]:
    """Aggregate ``(segment, slot, impressions, converted)`` rows as pseudo-rows.

    Keeps one estimator: the aggregation happens in Spark for scale, and the
    arithmetic that turns rates into propensities stays the pure function the
    pandas path and the unit tests use.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        slot = row["slot"]
        if slot is None:
            continue
        converted = int(row["converted"] or 0)
        impressions = int(row["impressions"] or 0)
        out.extend(
            {"slot": float(slot), "label": True, by: row["segment"]}
            for _ in range(converted)
        )
        out.extend(
            {"slot": float(slot), "label": False, by: row["segment"]}
            for _ in range(max(0, impressions - converted))
        )
    return out


def _grouped_sql(plan: SplitPlan, propensities: Mapping[str, float]) -> dict[str, str]:
    """One query per split; rows are renders, ordered by the incumbent's slots."""
    train_end = _lit(plan.train_end)
    val_end = _lit(plan.val_end)
    qa_end = _lit(plan.qa_end) if plan.qa_end is not None else "(SELECT max(ts) FROM dedup)"
    weights = _weights_expr(plan, propensities)
    carried = "".join(
        f", {c}" for c in dict.fromkeys([*plan.cohort_columns, plan.propensity_by]) if c
    )
    select = (
        "SELECT group_id, candidates, positions, labels, user_id, ts"
        f"{carried}{weights} FROM grp"
    )
    return {
        "train": f"WITH {_grouped_base_cte(plan)} {select} WHERE ts <= {train_end}",
        "validation": (
            f"WITH {_grouped_base_cte(plan)} {select} "
            f"WHERE ts > {train_end} AND ts <= {val_end}"
        ),
        "qa": (
            f"WITH {_grouped_base_cte(plan)} {select} "
            f"WHERE ts > {val_end} AND ts <= {qa_end}"
        ),
    }


def _materialize_spark(
    contract: Mapping[str, Any], plan: SplitPlan, spark: Any
) -> dict[str, Any]:
    propensities: dict[str, float] = {}
    if plan.grouped:
        if plan.relevance_weighting == "pbm":
            propensities = _estimate_propensities_spark(plan, spark)
        queries = _grouped_sql(plan, propensities)
        identity = "group_id"
    elif plan.strategy == "leave_last_n":
        queries = _leave_last_n_sql(plan)
        identity = "user_id"
    else:
        train_end = _lit(plan.train_end)
        val_end = _lit(plan.val_end)
        qa_end = _lit(plan.qa_end) if plan.qa_end is not None else None
        queries = {
            "train": (
                f"WITH {_base_cte(plan)} "
                f"SELECT user_id, item_id, ts FROM f WHERE ts <= {train_end}"
            ),
            "validation": _user_grain_sql(plan, train_end, train_end, val_end),
            "qa": _user_grain_sql(
                plan, val_end, val_end, qa_end or "(SELECT max(ts) FROM f)"
            ),
        }
        identity = "user_id"

    _ensure_schema(spark, plan)
    refs: dict[str, Any] = {}
    for name, query in queries.items():
        table = plan.table(name)
        spark.sql(f"DROP TABLE IF EXISTS {table}")
        spark.sql(f"CREATE TABLE {table} USING DELTA AS {query}")
        row_count = int(spark.sql(f"SELECT count(*) AS c FROM {table}").collect()[0][0])
        version = int(
            spark.sql(f"DESCRIBE HISTORY {table} LIMIT 1").collect()[0]["version"]
        )
        digest = spark.sql(
            f"SELECT sha2(concat_ws(',', array_sort(collect_list(cast({identity} AS "
            f"string)))), 256) AS h FROM {table}"
        ).collect()[0][0]
        refs[name] = {
            "name": name,
            "table": table,
            "delta_version": version,
            "row_count": row_count,
            "membership_hash": digest,
        }
    return _result(contract, plan, refs, backend="spark", propensities=propensities)


def _leave_last_n_sql(plan: SplitPlan) -> dict[str, str]:
    n_val, n_qa = int(plan.n_val), int(plan.n_qa)
    ranked = (
        f"WITH {_base_cte(plan)}, "
        "r AS (SELECT user_id, item_id, ts, "
        "row_number() OVER (PARTITION BY user_id ORDER BY ts DESC) AS rn, "
        "count(*) OVER (PARTITION BY user_id) AS n FROM f)"
    )
    min_len = n_val + n_qa + 1
    user_grain = (
        "SELECT t.user_id, coalesce(h.history, array()) AS history, t.targets, "
        "size(coalesce(h.history, array())) AS history_len, "
        "CAST(NULL AS DOUBLE) AS days_since_last "
        "FROM tgt t LEFT JOIN hist h ON t.user_id = h.user_id "
        "WHERE size(t.targets) > 0"
    )
    return {
        "train": (
            f"{ranked} SELECT user_id, item_id, ts FROM r "
            f"WHERE n < {min_len} OR rn > {n_val + n_qa}"
        ),
        "validation": (
            f"{ranked}, "
            f"hist AS (SELECT user_id, {_SEQ} AS history FROM r "
            f"WHERE n >= {min_len} AND rn > {n_val + n_qa} GROUP BY user_id), "
            f"tgt AS (SELECT user_id, {_SEQ} AS targets FROM r "
            f"WHERE n >= {min_len} AND rn > {n_qa} AND rn <= {n_val + n_qa} "
            "GROUP BY user_id) " + user_grain
        ),
        "qa": (
            f"{ranked}, "
            f"hist AS (SELECT user_id, {_SEQ} AS history FROM r "
            f"WHERE n >= {min_len} AND rn > {n_qa} GROUP BY user_id), "
            f"tgt AS (SELECT user_id, {_SEQ} AS targets FROM r "
            f"WHERE n >= {min_len} AND rn <= {n_qa} GROUP BY user_id) " + user_grain
        ),
    }


def _lit(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def _result(
    contract: Mapping[str, Any],
    plan: SplitPlan,
    refs: Mapping[str, Any],
    backend: str,
    propensities: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    from amt.jobs.eval.harness import HARNESS_VERSION

    return {
        "splits": dict(refs),
        "catalog": plan.catalog,
        "schema": plan.schema,
        "source_table": plan.source_table,
        "strategy": plan.strategy,
        "evaluation_mode": plan.evaluation_mode,
        "relevance_weighting": plan.relevance_weighting,
        "propensities": dict(propensities or {}),
        "backend": backend,
        "contract_hash": contract.get("contract_hash"),
        "harness_version": HARNESS_VERSION,
        "status": "success",
        "error": None,
    }


def materialize(
    contract: Mapping[str, Any],
    spark: Any | None = None,
    output_dir: str | None = None,
    catalog: str | None = None,
    schema: str | None = None,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Build the three frozen splits and return their immutable references."""
    plan = plan_from_contract(contract)
    if catalog:
        plan.catalog = str(catalog)
    if schema:
        plan.schema = str(schema)
    if campaign_id:
        plan.table_prefix = table_prefix_for(campaign_id)
    if spark is not None:
        return _materialize_spark(contract, plan, spark)
    if not output_dir:
        raise ValueError("no Spark session and no output_dir for local materialization")
    return _materialize_local(contract, plan, output_dir)


def run(args: Any) -> dict[str, Any]:
    """``--role materialize`` entry point (see :mod:`amt.jobs.entry`)."""
    from amt.jobs.entry import get_spark, load_json_arg

    contract = load_json_arg(args.contract_json) or {}
    return materialize(
        contract,
        spark=get_spark(),
        output_dir=getattr(args, "local_dir", None),
        catalog=getattr(args, "split_catalog", None),
        schema=getattr(args, "split_schema", None),
        campaign_id=getattr(args, "campaign_id", None),
    )
