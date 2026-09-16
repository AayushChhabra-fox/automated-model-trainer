"""Which experiment the orchestrator logs into, and where it looks for runs."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from amt.databricks import mlflow_client
from amt.executor.databricks import DatabricksExecutor

EXPERIMENT_ID = "3680987379186329"
JOB_RUN_ID = "6ae2a5c2b2684429b21f1c949f3c1543"


class _FakeMlflow:
    """Stands in for the module so the test never touches global MLflow state."""

    def __init__(self) -> None:
        self.uri = ""

    def set_tracking_uri(self, uri: str) -> None:
        self.uri = uri

    def get_tracking_uri(self) -> str:
        return self.uri


def _executor(tracking_uri: str = "databricks") -> DatabricksExecutor:
    settings = SimpleNamespace(
        mlflow_tracking_uri=tracking_uri,
        mlflow_experiment_id=EXPERIMENT_ID,
        databricks_host="https://example.cloud.databricks.com",
        databricks_token="dapi-xxx",
    )
    return DatabricksExecutor(settings=settings, client=object())


@pytest.fixture
def tracking(monkeypatch: pytest.MonkeyPatch) -> _FakeMlflow:
    monkeypatch.setenv("DATABRICKS_AUTH_TYPE", "unset")
    fake = _FakeMlflow()
    monkeypatch.setattr(mlflow_client, "_mlflow", lambda: fake)
    return fake


def test_the_search_runs_against_databricks_not_a_local_store(
    monkeypatch: pytest.MonkeyPatch, tracking: _FakeMlflow
) -> None:
    """MLflow defaults to a local store, where the cluster's run will never be."""
    seen: dict[str, Any] = {}

    class _Client:
        def search_runs(self, experiment_ids: list[str], **kwargs: Any) -> list[Any]:
            seen["uri"] = tracking.uri
            seen["experiment_ids"] = experiment_ids
            return [SimpleNamespace(info=SimpleNamespace(run_id="r1"))]

    monkeypatch.setattr("mlflow.tracking.MlflowClient", _Client)

    run = _executor().find_mlflow_run(JOB_RUN_ID)

    assert run.info.run_id == "r1"
    assert seen == {"uri": "databricks", "experiment_ids": [EXPERIMENT_ID]}


def test_a_run_the_search_cannot_find_names_the_experiment(
    monkeypatch: pytest.MonkeyPatch, tracking: _FakeMlflow
) -> None:
    from amt.executor.base import ExecutorError

    class _Client:
        def search_runs(self, experiment_ids: list[str], **kwargs: Any) -> list[Any]:
            return []

    monkeypatch.setattr("mlflow.tracking.MlflowClient", _Client)

    with pytest.raises(ExecutorError, match=EXPERIMENT_ID):
        _executor().find_mlflow_run(JOB_RUN_ID)


# --- which experiment the campaign logs into -------------------------------------


def _settings(experiment_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        mlflow_tracking_uri="databricks",
        mlflow_experiment_id=experiment_id,
        mlflow_experiment_root="/Users/me/amt-exp",
        experiment_name=lambda campaign: f"/Users/me/amt-exp/{campaign}",
        databricks_host="https://example.cloud.databricks.com",
        databricks_token="dapi-xxx",
    )


def test_a_pinned_experiment_wins_over_the_per_campaign_default(
    tracking: _FakeMlflow,
) -> None:
    """Jobs are already told to log into it; resolving another would split the campaign."""

    def _fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a pinned experiment must not be looked up or created")

    tracking.get_experiment_by_name = _fail
    tracking.create_experiment = _fail

    assert mlflow_client.get_or_create_experiment("c1", _settings(EXPERIMENT_ID)) == EXPERIMENT_ID


def test_a_root_that_is_already_an_experiment_says_so(tracking: _FakeMlflow) -> None:
    tracking.get_experiment_by_name = lambda name: None
    tracking.create_experiment = _raise(
        "INVALID_PARAMETER_VALUE: Path component 'amt-exp' exists under parent 28 "
        "but is not a directory"
    )

    with pytest.raises(RuntimeError, match="AMT_MLFLOW_EXPERIMENT_ROOT"):
        mlflow_client.get_or_create_experiment("c1", _settings())


def test_an_unrelated_creation_failure_is_not_reinterpreted(tracking: _FakeMlflow) -> None:
    tracking.get_experiment_by_name = lambda name: None
    tracking.create_experiment = _raise("PERMISSION_DENIED: no CREATE on the workspace")

    with pytest.raises(Exception, match="PERMISSION_DENIED"):
        mlflow_client.get_or_create_experiment("c1", _settings())


def _raise(message: str) -> Any:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise ValueError(message)

    return _boom
