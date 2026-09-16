"""What `amt doctor` says about a pinned all-purpose cluster.

Pinning one is a trade an operator is entitled to make, so the check reports
what it reads off the cluster instead of refusing to proceed. These tests pin
the distinction the message turns on: a static `amt` tag does reach billing,
and the per-run dimensions still cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from amt.cli import _existing_cluster_report
from amt.settings import Settings

CLUSTER_ID = "0914-091216-h6cd8ahc"


@dataclass
class FakeCluster:
    cluster_name: str = "amt_cluster"
    node_type_id: str = "i3.2xlarge"
    spark_version: str = "15.4.x-cpu-ml-scala2.12"
    num_workers: int = 2
    custom_tags: dict[str, str] = field(default_factory=dict)


def _report(monkeypatch: pytest.MonkeyPatch, cluster: FakeCluster) -> str:
    class FakeClient:
        clusters = type("C", (), {"get": staticmethod(lambda cluster_id: cluster)})()

    monkeypatch.setattr("amt.databricks.client.get_workspace_client", lambda *_: FakeClient())
    settings = Settings(_env_file=None, AMT_CATALOG="main", AMT_SCHEMA="amt")
    return _existing_cluster_report(settings, CLUSTER_ID)


def test_an_amt_tag_is_credited_as_reaching_billing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator's tag works; saying otherwise teaches them to ignore doctor."""
    report = _report(monkeypatch, FakeCluster(custom_tags={"amt": "true"}))

    assert "IS attributable" in report
    assert "no amt tag" not in report


def test_a_missing_amt_tag_is_called_out(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report(monkeypatch, FakeCluster())

    assert "no amt tag" in report
    assert "amt=true" in report


def test_per_run_attribution_is_lost_even_when_tagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real cost of pinning, and the one a cluster tag cannot buy back."""
    report = _report(monkeypatch, FakeCluster(custom_tags={"amt": "true"}))

    assert "amt_campaign_id" in report
    assert "amt_hypothesis_id" in report
    assert "All-Purpose" in report


def test_a_plain_runtime_warns_that_boosters_install_per_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(monkeypatch, FakeCluster(spark_version="17.3.x-scala2.13"))

    assert "not an ML runtime" in report
    assert "lightgbm_*" in report


def test_an_ml_runtime_says_nothing_about_installs(monkeypatch: pytest.MonkeyPatch) -> None:
    report = _report(monkeypatch, FakeCluster())

    assert "ML runtime" not in report


def test_a_single_node_is_flagged_for_materialization_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Boosters fit on one machine; the Spark shuffle in materialization does not."""
    cluster = FakeCluster(num_workers=0, custom_tags={"ResourceClass": "SingleNode"})

    report = _report(monkeypatch, cluster)

    assert "single node" in report
    assert "materialization" in report


def test_a_multi_worker_cluster_is_not_flagged_as_single_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(monkeypatch, FakeCluster())

    assert "single node" not in report
    assert "2 worker(s)" in report
