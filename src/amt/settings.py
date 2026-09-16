"""Environment-driven configuration.

Secrets are read from the environment only (or a local ``.env``); nothing is ever
persisted to the repo. See ``.env.example`` for the full set of variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All configuration for a local AMT orchestrator process."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Databricks workspace / auth (OAuth M2M, service principal) -------------
    databricks_host: str | None = Field(
        default=None, validation_alias=AliasChoices("DATABRICKS_HOST", "databricks_host")
    )
    databricks_client_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABRICKS_CLIENT_ID", "databricks_client_id"),
    )
    databricks_client_secret: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABRICKS_CLIENT_SECRET", "databricks_client_secret"),
    )
    # A PAT authenticates as a human, so jobs run under that user's Unity Catalog grants.
    databricks_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABRICKS_TOKEN", "databricks_token"),
    )
    warehouse_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DATABRICKS_WAREHOUSE_ID", "AMT_WAREHOUSE_ID"),
    )

    # --- Unity Catalog destinations ---------------------------------------------
    catalog: str = Field(default="main", validation_alias=AliasChoices("AMT_CATALOG", "catalog"))
    schema_name: str = Field(
        default="amt", validation_alias=AliasChoices("AMT_SCHEMA", "schema_name")
    )
    volume_path: str | None = Field(
        default=None, validation_alias=AliasChoices("AMT_VOLUME_PATH", "volume_path")
    )

    # --- Compute: job cluster spec, or an existing cluster -----------------------
    spark_version: str = Field(
        default="15.4.x-cpu-ml-scala2.12", validation_alias=AliasChoices("AMT_SPARK_VERSION")
    )
    node_type_id: str = Field(
        default="Standard_DS3_v2", validation_alias=AliasChoices("AMT_NODE_TYPE_ID")
    )
    num_workers: int = Field(default=2, validation_alias=AliasChoices("AMT_NUM_WORKERS"))
    # A job cluster with no access mode is a *non*-Unity-Catalog cluster, and a
    # wheel on a UC Volume cannot be installed on one ("Libraries from Volumes
    # are not allowed on non Unity Catalog cluster"). AMT also needs the
    # caller's UC grants to read the source table, which is single-user access.
    data_security_mode: str = Field(
        default="SINGLE_USER", validation_alias=AliasChoices("AMT_DATA_SECURITY_MODE")
    )
    existing_cluster_id: str | None = Field(
        default=None, validation_alias=AliasChoices("AMT_EXISTING_CLUSTER_ID")
    )

    # --- MLflow -------------------------------------------------------------------
    mlflow_experiment_root: str = Field(
        default="/Shared/amt", validation_alias=AliasChoices("AMT_MLFLOW_EXPERIMENT_ROOT")
    )
    mlflow_tracking_uri: str = Field(
        default="databricks", validation_alias=AliasChoices("MLFLOW_TRACKING_URI")
    )
    # Pin every job to one existing experiment instead of the per-campaign default.
    mlflow_experiment_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("AMT_MLFLOW_EXPERIMENT_ID", "MLFLOW_EXPERIMENT_ID"),
    )

    # --- Agent models: ``provider:model_id`` (see ``amt.llm``) ----------------------
    planner_model: str = Field(
        default="anthropic:claude-opus-5", validation_alias=AliasChoices("AMT_PLANNER_MODEL")
    )
    trainer_model: str = Field(
        default="anthropic:claude-opus-5", validation_alias=AliasChoices("AMT_TRAINER_MODEL")
    )
    qa_model: str = Field(
        default="anthropic:claude-opus-5", validation_alias=AliasChoices("AMT_QA_MODEL")
    )
    problem_spec_model: str = Field(
        default="anthropic:claude-sonnet-5",
        validation_alias=AliasChoices("AMT_PROBLEM_SPEC_MODEL"),
    )
    # One credential pair for every provider; vendor names are accepted so an existing
    # single-provider ``.env`` keeps working.
    llm_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "AMT_LLM_API_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "HF_TOKEN",
            "HUGGINGFACE_API_KEY",
        ),
    )
    llm_base_url: str | None = Field(
        default=None, validation_alias=AliasChoices("AMT_LLM_BASE_URL", "OPENAI_BASE_URL")
    )
    llm_max_retries: int = Field(default=3, validation_alias=AliasChoices("AMT_LLM_MAX_RETRIES"))
    llm_temperature: float = Field(
        default=0.0, validation_alias=AliasChoices("AMT_LLM_TEMPERATURE")
    )

    # --- Orchestration ---------------------------------------------------------------
    checkpoint_db: Path = Field(
        default=Path("~/.amt/checkpoints.sqlite"),
        validation_alias=AliasChoices("AMT_CHECKPOINT_DB"),
    )
    max_probe_steps: int = Field(default=8, validation_alias=AliasChoices("AMT_MAX_PROBE_STEPS"))
    phase0_max_attempts: int = Field(
        default=3, validation_alias=AliasChoices("AMT_PHASE0_MAX_ATTEMPTS")
    )
    # `n` candidates per validation row is the one recipe number that can make a
    # job unrunnable, and it looks innocuous in the proposal.
    max_candidate_rows: int = Field(
        default=50_000_000, validation_alias=AliasChoices("AMT_MAX_CANDIDATE_ROWS")
    )

    @field_validator("checkpoint_db", mode="before")
    @classmethod
    def _expand(cls, v: object) -> object:
        if isinstance(v, str):
            return Path(v).expanduser()
        if isinstance(v, Path):
            return v.expanduser()
        return v

    # --- Derived ----------------------------------------------------------------------
    @property
    def artifact_volume(self) -> str:
        """UC Volume path used for the wheel and campaign artifacts."""
        if self.volume_path:
            return self.volume_path
        return f"/Volumes/{self.catalog}/{self.schema_name}/amt_artifacts"

    def experiment_name(self, campaign_id: str) -> str:
        return f"{self.mlflow_experiment_root.rstrip('/')}/{campaign_id}"

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema_name}.{name}"

    def model_for(self, agent: str) -> str:
        return {
            "planner": self.planner_model,
            "trainer": self.trainer_model,
            "qa": self.qa_model,
            "problem_spec": self.problem_spec_model,
        }.get(agent, self.planner_model)

    def cluster_spec(self) -> dict:
        """Job-cluster spec, or ``{"existing_cluster_id": ...}`` when one is configured."""
        if self.existing_cluster_id:
            return {"existing_cluster_id": self.existing_cluster_id}
        return {
            "spark_version": self.spark_version,
            "node_type_id": self.node_type_id,
            "num_workers": self.num_workers,
            "data_security_mode": self.data_security_mode,
        }

    def resolved_checkpoint_db(self) -> Path:
        p = Path(self.checkpoint_db).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def reports_dir(self) -> Path:
        """Where rendered reports land: beside the checkpoint database.

        Derived rather than configured, so a test that isolates the campaign
        database also isolates its reports, and so the two never disagree
        about which machine a campaign lives on.
        """
        directory = Path(self.checkpoint_db).expanduser().parent / "reports"
        directory.mkdir(parents=True, exist_ok=True)
        return directory


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton."""
    return Settings()
