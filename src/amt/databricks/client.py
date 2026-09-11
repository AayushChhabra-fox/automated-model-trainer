"""WorkspaceClient factory using service-principal OAuth (M2M) from the environment."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

from amt.settings import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - typing only
    from databricks.sdk import WorkspaceClient


class DatabricksConfigError(RuntimeError):
    """Raised when required Databricks configuration is missing."""


def _require(settings: Settings) -> None:
    missing = [
        name
        for name, value in (
            ("DATABRICKS_HOST", settings.databricks_host),
            ("DATABRICKS_CLIENT_ID", settings.databricks_client_id),
            ("DATABRICKS_CLIENT_SECRET", settings.databricks_client_secret),
        )
        if not value
    ]
    if missing:
        raise DatabricksConfigError(
            "Missing Databricks configuration: " + ", ".join(missing) + ". See .env.example."
        )


def get_workspace_client(settings: Settings | None = None) -> WorkspaceClient:
    """Build a ``WorkspaceClient`` authenticated as the configured service principal."""
    settings = settings or get_settings()
    _require(settings)
    from databricks.sdk import WorkspaceClient  # imported lazily: keeps unit tests offline

    return WorkspaceClient(
        host=settings.databricks_host,
        client_id=settings.databricks_client_id,
        client_secret=settings.databricks_client_secret,
    )


@lru_cache(maxsize=1)
def default_client() -> Any:
    """Process-wide cached client for the default settings."""
    return get_workspace_client()
