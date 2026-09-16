"""WorkspaceClient factory using service-principal OAuth (M2M) from the environment."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from amt.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk import WorkspaceClient


class DatabricksConfigError(RuntimeError):
    """Raised when required Databricks configuration is missing."""


def _require(settings: Settings) -> None:
    if not settings.databricks_host:
        raise DatabricksConfigError(
            "Missing Databricks configuration: DATABRICKS_HOST. See .env.example."
        )
    if settings.databricks_token:
        return
    missing = [
        name
        for name, value in (
            ("DATABRICKS_CLIENT_ID", settings.databricks_client_id),
            ("DATABRICKS_CLIENT_SECRET", settings.databricks_client_secret),
        )
        if not value
    ]
    if missing:
        raise DatabricksConfigError(
            "Missing Databricks configuration: DATABRICKS_TOKEN, or "
            + " and ".join(missing)
            + ". See .env.example."
        )


def pin_auth_env(settings: Settings | None = None) -> str:
    """Name AMT's chosen identity in the environment; return its SDK auth type.

    The SDK refuses to start when the environment describes two identities at
    once, and AMT is not the only thing that builds a client from it: MLflow's
    artifact repository constructs its own ``WorkspaceClient()``, which reads
    the ambient environment and knows nothing of the precedence below.

    The credential itself has to be written out too, not just the auth type.
    ``.env`` is read by pydantic-settings into :class:`Settings`, which MLflow
    cannot see; with only ``DATABRICKS_AUTH_TYPE`` set, MLflow's client falls
    through to ``~/.databrickscfg``, and a ``[DEFAULT]`` profile holding a
    username and password there answers every call with
    ``403 should_change_password``.
    """
    settings = settings or get_settings()
    auth_type = "pat" if settings.databricks_token else "oauth-m2m"
    os.environ["DATABRICKS_AUTH_TYPE"] = auth_type
    if settings.databricks_host:
        os.environ["DATABRICKS_HOST"] = settings.databricks_host
    if auth_type == "pat":
        os.environ["DATABRICKS_TOKEN"] = str(settings.databricks_token)
    else:
        os.environ["DATABRICKS_CLIENT_ID"] = str(settings.databricks_client_id or "")
        os.environ["DATABRICKS_CLIENT_SECRET"] = str(settings.databricks_client_secret or "")
    return auth_type


def get_workspace_client(settings: Settings | None = None) -> WorkspaceClient:
    """Build a ``WorkspaceClient`` for the configured identity.

    A PAT takes precedence over the service-principal pair: everything AMT submits
    then runs as that user, under their Unity Catalog grants.
    """
    settings = settings or get_settings()
    _require(settings)
    pin_auth_env(settings)
    from databricks.sdk import WorkspaceClient  # imported lazily: keeps unit tests offline

    if settings.databricks_token:
        # Pin auth_type so an ambient service-principal env or ~/.databrickscfg profile
        # cannot make the SDK see two credential sources and refuse to authenticate.
        return WorkspaceClient(
            host=settings.databricks_host,
            token=settings.databricks_token,
            auth_type="pat",
        )
    return WorkspaceClient(
        host=settings.databricks_host,
        client_id=settings.databricks_client_id,
        client_secret=settings.databricks_client_secret,
    )


@lru_cache(maxsize=1)
def default_client() -> Any:
    """Process-wide cached client for the default settings."""
    return get_workspace_client()
