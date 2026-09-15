"""Which Databricks identity the whole process uses, AMT's clients or not."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from amt.databricks import mlflow_client
from amt.databricks.client import pin_auth_env

PAT = SimpleNamespace(
    databricks_token="dapi-xxx",
    databricks_client_id="sp-id",
    databricks_client_secret="sp-secret",
    mlflow_tracking_uri="databricks",
)
SERVICE_PRINCIPAL = SimpleNamespace(
    databricks_token=None,
    databricks_client_id="sp-id",
    databricks_client_secret="sp-secret",
    mlflow_tracking_uri="databricks",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv, not delenv: monkeypatch then also undoes what the code under test writes.
    monkeypatch.setenv("DATABRICKS_AUTH_TYPE", "unset")


def test_a_pat_and_a_service_principal_together_resolve_to_the_pat() -> None:
    """The SDK refuses to start when the environment describes both."""
    assert pin_auth_env(PAT) == "pat"
    assert os.environ["DATABRICKS_AUTH_TYPE"] == "pat"


def test_a_service_principal_alone_is_still_named() -> None:
    assert pin_auth_env(SERVICE_PRINCIPAL) == "oauth-m2m"
    assert os.environ["DATABRICKS_AUTH_TYPE"] == "oauth-m2m"


def test_mlflow_is_pointed_at_databricks_as_that_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MLflow builds its own workspace client and reads only the environment."""

    class _FakeMlflow:
        uri = ""

        def set_tracking_uri(self, uri: str) -> None:
            type(self).uri = uri

        def get_tracking_uri(self) -> str:
            return self.uri

    monkeypatch.setattr(mlflow_client, "_mlflow", _FakeMlflow)

    assert mlflow_client.set_tracking_uri(PAT) == "databricks"
    assert os.environ["DATABRICKS_AUTH_TYPE"] == "pat"
