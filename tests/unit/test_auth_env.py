"""Which Databricks identity the whole process uses, AMT's clients or not."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from amt.databricks import mlflow_client
from amt.databricks.client import pin_auth_env

HOST = "https://example.cloud.databricks.com"

PAT = SimpleNamespace(
    databricks_host=HOST,
    databricks_token="dapi-xxx",
    databricks_client_id="sp-id",
    databricks_client_secret="sp-secret",
    mlflow_tracking_uri="databricks",
)
SERVICE_PRINCIPAL = SimpleNamespace(
    databricks_host=HOST,
    databricks_token=None,
    databricks_client_id="sp-id",
    databricks_client_secret="sp-secret",
    mlflow_tracking_uri="databricks",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv, not delenv: monkeypatch then also undoes what the code under test writes.
    for name in (
        "DATABRICKS_AUTH_TYPE",
        "DATABRICKS_HOST",
        "DATABRICKS_TOKEN",
        "DATABRICKS_CLIENT_ID",
        "DATABRICKS_CLIENT_SECRET",
    ):
        monkeypatch.setenv(name, "unset")


def test_a_pat_and_a_service_principal_together_resolve_to_the_pat() -> None:
    """The SDK refuses to start when the environment describes both."""
    assert pin_auth_env(PAT) == "pat"
    assert os.environ["DATABRICKS_AUTH_TYPE"] == "pat"


def test_a_service_principal_alone_is_still_named() -> None:
    assert pin_auth_env(SERVICE_PRINCIPAL) == "oauth-m2m"
    assert os.environ["DATABRICKS_AUTH_TYPE"] == "oauth-m2m"


def test_the_credential_itself_is_exported_not_just_the_auth_type() -> None:
    """`.env` is invisible to MLflow, which then falls through to ~/.databrickscfg.

    A `[DEFAULT]` profile there holding a username and password answers every
    MLflow call with `403 should_change_password`, and nothing in the campaign
    says why.
    """
    pin_auth_env(PAT)

    assert os.environ["DATABRICKS_HOST"] == HOST
    assert os.environ["DATABRICKS_TOKEN"] == "dapi-xxx"


def test_a_service_principal_exports_its_pair_and_no_token() -> None:
    pin_auth_env(SERVICE_PRINCIPAL)

    assert os.environ["DATABRICKS_CLIENT_ID"] == "sp-id"
    assert os.environ["DATABRICKS_CLIENT_SECRET"] == "sp-secret"
    assert os.environ["DATABRICKS_TOKEN"] == "unset", "a stale PAT must not linger"


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
