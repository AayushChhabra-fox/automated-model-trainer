"""SplitGuard is the mechanism that makes "never train on QA" structural.

These tests assert the guard refuses, rather than merely discourages, the
accesses the SPEC forbids -- and that a split whose contents have drifted
from the contract is rejected instead of silently used.
"""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from amt.jobs.data.loaders import (
    SplitAccessViolation,
    SplitContractMismatch,
    SplitGuard,
    normalize_split_name,
    parse_allowed_splits,
    split_ref,
    write_local_table,
)
from amt.jobs.materialize_splits import (
    materialize,
    membership_hash,
    plan_from_contract,
    table_prefix_for,
)


def _interactions(n_users: int = 12, n_events: int = 10) -> pd.DataFrame:
    rows = []
    for u in range(n_users):
        for e in range(n_events):
            rows.append(
                {
                    "user_id": f"u{u}",
                    "item_id": f"i{(u + e) % 7}",
                    "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(days=e),
                }
            )
    return pd.DataFrame(rows)


def _interactions_between(start: str, end: str, n_users: int = 12) -> pd.DataFrame:
    """Weekly interactions spanning the mock contract's temporal range."""
    stamps = pd.date_range(start, end, freq="7D")
    return pd.DataFrame(
        [
            {
                "user_id": f"u{u}",
                "item_id": f"i{(u + e) % 7}",
                "ts": ts.isoformat(),
                "event_type": "start",
            }
            for u in range(n_users)
            for e, ts in enumerate(stamps)
        ]
    )


def _write(tmp_path, frame, name="interactions"):
    return write_local_table(frame, os.path.join(str(tmp_path), f"{name}.jsonl"))


def _contract(tmp_path):
    return {
        "contract_hash": "abc123",
        "dataset": {
            "source_table": "main.amt.interactions",
            "source_path": os.path.join(str(tmp_path), "interactions.jsonl"),
            "catalog": "main",
            "schema": "amt",
            "min_interactions_per_user": 3,
            "split": {
                "strategy": "temporal",
                "train_end": "2024-01-05",
                "val_end": "2024-01-08",
                "qa_end": "2024-01-11",
            },
        },
    }


# -- name handling ---------------------------------------------------------


def test_split_aliases_normalize():
    assert normalize_split_name("val") == "validation"
    assert normalize_split_name("TEST") == "qa"
    assert normalize_split_name("split_train") == "train"


def test_unknown_split_names_are_a_violation_not_a_typo():
    with pytest.raises(SplitAccessViolation):
        normalize_split_name("holdout_v2")


def test_parse_allowed_splits_handles_cli_form():
    assert parse_allowed_splits("train, validation") == {"train", "validation"}
    assert parse_allowed_splits("") == set()
    assert parse_allowed_splits(None) == set()


# -- the guard itself ------------------------------------------------------


def test_train_job_cannot_read_qa():
    guard = SplitGuard("train,validation", {"qa_split": {"table": "t", "row_count": 1}})
    assert guard.is_allowed("train")
    assert not guard.is_allowed("qa")
    with pytest.raises(SplitAccessViolation) as exc:
        guard.read("qa")
    assert "not in allowed_splits" in str(exc.value)


def test_require_subset_of_rejects_an_over_broad_grant():
    guard = SplitGuard("train,validation,qa", {})
    with pytest.raises(SplitAccessViolation):
        guard.require_subset_of(("train", "validation"))


def test_phase1_requires_exactly_qa():
    SplitGuard("qa", {}).require_exactly(("qa",))
    with pytest.raises(SplitAccessViolation):
        SplitGuard("qa,validation", {}).require_exactly(("qa",))
    with pytest.raises(SplitAccessViolation):
        SplitGuard("validation", {}).require_exactly(("qa",))


def test_split_ref_reads_both_contract_shapes():
    flat = {"train_split": {"table": "a", "delta_version": 3}}
    nested = {"splits": {"train": {"table": "a", "delta_version": 3}}}
    assert split_ref(flat, "train")["table"] == "a"
    assert split_ref(nested, "train")["delta_version"] == 3


def test_missing_split_ref_is_a_contract_mismatch():
    with pytest.raises(SplitContractMismatch):
        split_ref({}, "qa")


def test_version_as_of_is_used_for_spark_reads():
    class FakeDF:
        def count(self):
            return 5

    class FakeSpark:
        def __init__(self):
            self.queries = []

        def sql(self, query):
            self.queries.append(query)
            return FakeDF()

    contract = {
        "splits": {"train": {"table": "main.amt.split_train", "delta_version": 7,
                             "row_count": 5}}
    }
    spark = FakeSpark()
    SplitGuard("train", contract).read("train", spark)
    assert spark.queries == ["SELECT * FROM main.amt.split_train VERSION AS OF 7"]


def test_row_count_drift_is_rejected():
    class FakeDF:
        def count(self):
            return 99

    class FakeSpark:
        def sql(self, query):
            return FakeDF()

    contract = {
        "splits": {"train": {"table": "t", "delta_version": 1, "row_count": 5}}
    }
    with pytest.raises(SplitContractMismatch) as exc:
        SplitGuard("train", contract).read("train", FakeSpark())
    assert "row count" in str(exc.value)


def test_local_fallback_reads_and_records_the_access(tmp_path):
    frame = pd.DataFrame({"user_id": ["a", "b"], "item_id": ["x", "y"]})
    path = write_local_table(frame, os.path.join(str(tmp_path), "split_train.jsonl"))
    contract = {
        "splits": {
            "train": {"table": "main.amt.split_train", "path": path, "row_count": 2}
        }
    }
    guard = SplitGuard("train", contract, local_dir=str(tmp_path))
    got = guard.read("train")
    assert len(got) == 2
    assert guard.reads == ["train"]


def test_local_fallback_finds_the_table_by_name(tmp_path):
    frame = pd.DataFrame({"user_id": ["a"]})
    write_local_table(frame, os.path.join(str(tmp_path), "split_qa.jsonl"))
    contract = {"splits": {"qa": {"table": "main.amt.split_qa", "row_count": 1}}}
    guard = SplitGuard("qa", contract, local_dir=str(tmp_path))
    assert len(guard.read("qa")) == 1


# -- materialization -------------------------------------------------------


def test_membership_hash_is_order_independent_and_sensitive():
    assert membership_hash(["a", "b"]) == membership_hash(["b", "a"])
    assert membership_hash(["a", "b"]) != membership_hash(["a", "c"])


def test_plan_comes_from_the_frozen_contract_alone(evaluation_contract):
    """Materialization must be executable from the contract's own fields.

    The job runs on a cluster with no AMT settings and no access to the prose a
    human read at HC3, so anything it needs -- source table, column names,
    boundaries, eligibility -- has to be a field the contract carries.
    """
    plan = plan_from_contract(evaluation_contract.model_dump(mode="json"))
    assert plan.source_table == "main.amt.interactions"
    assert plan.strategy == "temporal"
    assert (plan.train_end, plan.val_end, plan.qa_end) == (
        "2026-07-10",
        "2026-07-24",
        "2026-08-01",
    )
    assert plan.min_interactions_per_user == 3
    assert plan.filters == ["event_type = 'start'"]


def test_plan_maps_the_contract_column_names(evaluation_contract):
    payload = evaluation_contract.model_dump(mode="json")
    payload["dataset"]["columns"] = {
        "user": "account_id",
        "item": "entity_id",
        "timestamp": "event_timestamp",
    }
    plan = plan_from_contract(payload)
    assert (plan.user_col, plan.item_col, plan.ts_col) == (
        "account_id",
        "entity_id",
        "event_timestamp",
    )


def test_plan_refuses_a_contract_with_nothing_to_split():
    with pytest.raises(ValueError, match="source_table"):
        plan_from_contract({"dataset": {}, "split_policy": {}})


def test_a_frozen_contract_materializes_end_to_end(tmp_path, evaluation_contract):
    frame = _interactions_between("2026-05-01", "2026-08-01")
    _write(tmp_path, frame, name="interactions")

    result = materialize(
        evaluation_contract.model_dump(mode="json"),
        output_dir=str(tmp_path),
        catalog="product_api_dev",
        schema="amt",
    )

    assert result["splits"]["train"]["table"] == "product_api_dev.amt.split_train"
    assert all(ref["row_count"] > 0 for ref in result["splits"].values())


def test_each_campaign_writes_its_own_tables(tmp_path, evaluation_contract):
    """Campaigns share a schema, so the name is all that keeps them apart."""
    _write(tmp_path, _interactions_between("2026-05-01", "2026-08-01"), name="interactions")

    result = materialize(
        evaluation_contract.model_dump(mode="json"),
        output_dir=str(tmp_path),
        catalog="product_api_dev",
        schema="amt",
        campaign_id="vod-nextitem-001",
    )

    assert [ref["table"] for ref in result["splits"].values()] == [
        "product_api_dev.amt.vod_nextitem_001_split_train",
        "product_api_dev.amt.vod_nextitem_001_split_validation",
        "product_api_dev.amt.vod_nextitem_001_split_qa",
    ]


@pytest.mark.parametrize(
    ("campaign_id", "expected"),
    [
        ("vod-nextitem-001", "vod_nextitem_001"),
        ("LFY Rerank v2", "lfy_rerank_v2"),
        ("2026-lfy", "c_2026_lfy"),  # an identifier cannot start with a digit
        ("--", ""),
    ],
)
def test_a_campaign_id_becomes_a_legal_identifier(campaign_id, expected):
    assert table_prefix_for(campaign_id) == expected


def test_plan_reads_the_dataset_block(tmp_path):
    plan = plan_from_contract(_contract(tmp_path))
    assert plan.strategy == "temporal"
    assert plan.min_interactions_per_user == 3
    assert plan.table("validation") == "main.amt.split_validation"


def test_temporal_materialization_produces_guarded_readable_splits(tmp_path):
    _write(tmp_path, _interactions())
    contract = _contract(tmp_path)
    result = materialize(contract, output_dir=str(tmp_path))

    assert set(result["splits"]) == {"train", "validation", "qa"}
    for ref in result["splits"].values():
        assert ref["row_count"] > 0
        assert len(ref["membership_hash"]) == 64

    # The contract now carries immutable refs; the guard can read through them.
    contract["splits"] = result["splits"]
    guard = SplitGuard("train,validation", contract, local_dir=str(tmp_path))
    train = guard.read("train")
    validation = guard.read("validation")
    assert {"history", "targets", "history_len"} <= set(validation.columns)
    # Train stops at the boundary: no leakage of post-boundary interactions.
    assert pd.to_datetime(train["ts"]).max() <= pd.Timestamp("2024-01-05")

    with pytest.raises(SplitAccessViolation):
        guard.read("qa")


def test_validation_history_precedes_targets(tmp_path):
    _write(tmp_path, _interactions())
    contract = _contract(tmp_path)
    result = materialize(contract, output_dir=str(tmp_path))
    contract["splits"] = result["splits"]
    validation = SplitGuard("validation", contract, local_dir=str(tmp_path)).read(
        "validation"
    )
    assert len(validation) > 0
    for _, row in validation.iterrows():
        # Every row must give the model something to predict.
        assert len(row["targets"]) > 0
        assert row["history_len"] == len(row["history"])


def test_leave_last_n_holds_out_the_tail_per_user(tmp_path):
    _write(tmp_path, _interactions(n_users=6, n_events=8))
    contract = _contract(tmp_path)
    contract["dataset"]["split"] = {"strategy": "leave_last_n", "n_val": 1, "n_qa": 1}
    result = materialize(contract, output_dir=str(tmp_path))
    contract["splits"] = result["splits"]

    qa = SplitGuard("qa", contract, local_dir=str(tmp_path)).read("qa")
    validation = SplitGuard("validation", contract, local_dir=str(tmp_path)).read(
        "validation"
    )
    assert all(len(t) == 1 for t in qa["targets"])
    assert all(len(t) == 1 for t in validation["targets"])
    # QA history is one event longer than validation history for the same user.
    val_len = dict(zip(validation["user_id"], validation["history_len"]))
    qa_len = dict(zip(qa["user_id"], qa["history_len"]))
    for user in val_len:
        assert qa_len[user] == val_len[user] + 1


def test_eligibility_filter_drops_short_users(tmp_path):
    frame = pd.concat(
        [
            _interactions(n_users=3, n_events=8),
            pd.DataFrame(
                [
                    {
                        "user_id": "shorty",
                        "item_id": "i1",
                        "ts": pd.Timestamp("2024-01-02"),
                    }
                ]
            ),
        ]
    )
    _write(tmp_path, frame)
    contract = _contract(tmp_path)
    result = materialize(contract, output_dir=str(tmp_path))
    contract["splits"] = result["splits"]
    train = SplitGuard("train", contract, local_dir=str(tmp_path)).read("train")
    assert "shorty" not in set(train["user_id"])


def test_result_payload_carries_contract_and_harness_stamps(tmp_path):
    _write(tmp_path, _interactions())
    result = materialize(_contract(tmp_path), output_dir=str(tmp_path))
    assert result["contract_hash"] == "abc123"
    assert result["harness_version"] == "1.0.0"
    assert result["status"] == "success"
    # Must round-trip as the result.json artifact.
    assert json.loads(json.dumps(result, default=str))["backend"] == "local"
