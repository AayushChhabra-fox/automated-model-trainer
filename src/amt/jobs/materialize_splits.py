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
    kwargs.update({k: v for k, v in override.items() if k in known})

    plan = SplitPlan(**kwargs)
    plan.filters = list(plan.filters or [])
    if not plan.source_table and not plan.source_path:
        raise ValueError("the contract names no dataset.source_table, so there is nothing to split")
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
    df = _eligible(df, plan)

    if plan.strategy == "leave_last_n":
        train, validation, qa = _leave_last_n_local(df, plan)
    else:
        train, validation, qa = _temporal_local(df, plan)

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
            "membership_hash": membership_hash(frame[plan.user_col].tolist()),
        }
    return _result(contract, plan, refs, backend="local")


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


def _materialize_spark(
    contract: Mapping[str, Any], plan: SplitPlan, spark: Any
) -> dict[str, Any]:
    if plan.strategy == "leave_last_n":
        queries = _leave_last_n_sql(plan)
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
            "SELECT sha2(concat_ws(',', array_sort(collect_list(cast(user_id AS "
            f"string)))), 256) AS h FROM {table}"
        ).collect()[0][0]
        refs[name] = {
            "name": name,
            "table": table,
            "delta_version": version,
            "row_count": row_count,
            "membership_hash": digest,
        }
    return _result(contract, plan, refs, backend="spark")


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
) -> dict[str, Any]:
    from amt.jobs.eval.harness import HARNESS_VERSION

    return {
        "splits": dict(refs),
        "catalog": plan.catalog,
        "schema": plan.schema,
        "source_table": plan.source_table,
        "strategy": plan.strategy,
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
