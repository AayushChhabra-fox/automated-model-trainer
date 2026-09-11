"""``SplitGuard`` -- the only way job code is allowed to touch a split.

This is enforcement mechanism #2 from PLAN §9.  A job is launched with
``--allowed-splits`` and constructs a guard from it; asking the guard for a
split outside that set raises :class:`SplitAccessViolation` rather than
returning data.  Train jobs are launched with ``train,validation`` and so
*cannot* read ``qa`` even if a model or a generated snippet tries to.

Reads are pinned to the Delta version recorded in the frozen
EvaluationContract (``VERSION AS OF``), and the row count returned is checked
against the contract, so a split that was rewritten underneath the campaign
fails loudly instead of silently changing the ground truth.

A pandas/parquet fallback is used when Spark is unavailable (unit tests and
local dry-runs); it honours exactly the same guard and row-count checks.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

__all__ = [
    "CANONICAL_SPLITS",
    "LOCAL_FORMATS",
    "SplitAccessViolation",
    "SplitContractMismatch",
    "normalize_split_name",
    "parse_allowed_splits",
    "split_ref",
    "read_local_table",
    "write_local_table",
    "SplitGuard",
]

# Preference order for the no-Spark fallback.  Parquet first (what UC holds),
# then jsonl, which keeps array columns intact without needing pyarrow.
LOCAL_FORMATS: tuple[str, ...] = (".parquet", ".jsonl", ".json", ".csv")

CANONICAL_SPLITS: tuple[str, ...] = ("train", "validation", "qa")

_ALIASES = {
    "train": "train",
    "training": "train",
    "split_train": "train",
    "val": "validation",
    "valid": "validation",
    "validation": "validation",
    "split_validation": "validation",
    "qa": "qa",
    "holdout": "qa",
    "test": "qa",
    "split_qa": "qa",
}


class SplitAccessViolation(RuntimeError):
    """Raised when job code asks for a split it was not granted."""


class SplitContractMismatch(RuntimeError):
    """Raised when the materialized split no longer matches the contract."""


def normalize_split_name(name: str) -> str:
    key = str(name).strip().lower()
    try:
        return _ALIASES[key]
    except KeyError:
        raise SplitAccessViolation(f"unknown split name: {name!r}") from None


def parse_allowed_splits(value: str | Iterable[str] | None) -> set[str]:
    """Parse a ``--allowed-splits train,validation`` argument into a set."""
    if value is None:
        return set()
    if isinstance(value, str):
        parts = [p for p in value.replace(" ", "").split(",") if p]
    else:
        parts = [str(p) for p in value]
    return {normalize_split_name(p) for p in parts}


def split_ref(contract: Mapping[str, Any], split: str) -> dict[str, Any]:
    """Pull the ``SplitRef`` for ``split`` out of an EvaluationContract dict.

    Tolerates the two shapes the contract may take: a ``splits`` mapping, or
    top-level ``train_split`` / ``validation_split`` / ``qa_split`` keys.
    """
    name = normalize_split_name(split)
    splits = contract.get("splits")
    if isinstance(splits, Mapping):
        for key, ref in splits.items():
            if normalize_split_name(key) == name and isinstance(ref, Mapping):
                return dict(ref)
    ref = contract.get(f"{name}_split")
    if isinstance(ref, Mapping):
        return dict(ref)
    raise SplitContractMismatch(
        f"contract does not define a split reference for {name!r}"
    )


def read_local_table(path: str):
    """Read a local table in whichever of :data:`LOCAL_FORMATS` it uses."""
    import pandas as pd

    if os.path.isdir(path) or path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    if path.endswith(".json"):
        return pd.read_json(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise SplitContractMismatch(f"unsupported local table format: {path}")


def write_local_table(df, path: str) -> str:
    """Counterpart to :func:`read_local_table` used by local materialization."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, index=False)
    elif path.endswith(".jsonl"):
        # ISO dates so timestamps survive the round trip as timestamps.
        df.to_json(path, orient="records", lines=True, date_format="iso")
    elif path.endswith(".json"):
        df.to_json(path, orient="records", date_format="iso")
    elif path.endswith(".csv"):
        df.to_csv(path, index=False)
    else:
        raise SplitContractMismatch(f"unsupported local table format: {path}")
    return path


class SplitGuard:
    """Role-scoped, version-pinned access to the frozen split tables."""

    def __init__(
        self,
        allowed_splits: str | Iterable[str],
        contract: Mapping[str, Any] | None = None,
        local_dir: str | None = None,
    ) -> None:
        self.allowed_splits = parse_allowed_splits(allowed_splits)
        self.contract: Mapping[str, Any] = contract or {}
        # Local parquet root used when no Spark session is available.
        self.local_dir = local_dir or os.environ.get("AMT_LOCAL_DATA_DIR")
        self._reads: list[str] = []

    # -- guard ------------------------------------------------------------

    def is_allowed(self, split: str) -> bool:
        try:
            return normalize_split_name(split) in self.allowed_splits
        except SplitAccessViolation:
            return False

    def assert_allowed(self, split: str) -> str:
        name = normalize_split_name(split)
        if name not in self.allowed_splits:
            raise SplitAccessViolation(
                f"split {name!r} is not in allowed_splits "
                f"{sorted(self.allowed_splits)}; this job is not permitted to "
                "read it"
            )
        return name

    def require_subset_of(self, permitted: Iterable[str]) -> None:
        """Assert this job was granted no more than ``permitted``."""
        allowed = set(permitted)
        extra = self.allowed_splits - {normalize_split_name(p) for p in allowed}
        if extra:
            raise SplitAccessViolation(
                f"job granted forbidden splits {sorted(extra)}; "
                f"permitted: {sorted(allowed)}"
            )

    def require_exactly(self, permitted: Iterable[str]) -> None:
        wanted = {normalize_split_name(p) for p in permitted}
        if self.allowed_splits != wanted:
            raise SplitAccessViolation(
                f"job requires allowed_splits == {sorted(wanted)}, got "
                f"{sorted(self.allowed_splits)}"
            )

    @property
    def reads(self) -> list[str]:
        """Splits actually read, in order -- logged for the audit trail."""
        return list(self._reads)

    # -- reading ----------------------------------------------------------

    def ref(self, split: str) -> dict[str, Any]:
        return split_ref(self.contract, self.assert_allowed(split))

    def read(self, split_name: str, spark: Any | None = None):
        """Read a split at the exact Delta version recorded in the contract."""
        name = self.assert_allowed(split_name)
        ref = split_ref(self.contract, name)
        table = ref.get("table")
        version = ref.get("delta_version")

        if spark is not None:
            if not table:
                raise SplitContractMismatch(
                    f"split {name!r} has no table in the contract"
                )
            if version is None:
                query = f"SELECT * FROM {table}"  # noqa: S608 - contract-controlled
            else:
                query = (
                    f"SELECT * FROM {table} VERSION AS OF {int(version)}"
                )  # noqa: S608
            df = spark.sql(query)
            actual = df.count()
        else:
            df = self._read_local(name, ref)
            actual = len(df)

        self._verify_row_count(name, ref, actual)
        self._reads.append(name)
        return df

    def _read_local(self, name: str, ref: Mapping[str, Any]):
        path = ref.get("path")
        if not path:
            root = self.local_dir
            if not root:
                raise SplitContractMismatch(
                    "no Spark session and no local path/AMT_LOCAL_DATA_DIR for "
                    f"split {name!r}"
                )
            table = str(ref.get("table") or f"split_{name}")
            leaf = table.split(".")[-1]
            for ext in LOCAL_FORMATS:
                cand = os.path.join(root, f"{leaf}{ext}")
                if os.path.exists(cand):
                    path = cand
                    break
            else:
                if os.path.isdir(os.path.join(root, leaf)):
                    path = os.path.join(root, leaf)
                else:
                    raise SplitContractMismatch(
                        f"no local data file for split {name!r} under {root}"
                    )
        return read_local_table(path)

    @staticmethod
    def _verify_row_count(
        name: str, ref: Mapping[str, Any], actual: int
    ) -> None:
        expected = ref.get("row_count")
        if expected is None:
            return
        if int(expected) != int(actual):
            raise SplitContractMismatch(
                f"split {name!r} row count {actual} != contract row_count "
                f"{expected}; the frozen split has changed -- stop and return "
                "to the EvaluationContract stage"
            )
