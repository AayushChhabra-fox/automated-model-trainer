"""MLflow conventions (PLAN §7). MLflow is the main persistence layer for AMT."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from amt.settings import Settings, get_settings

# --- tag constants (PLAN §7) ---------------------------------------------------
TAG_CAMPAIGN_ID = "amt.campaign_id"
TAG_STAGE = "amt.stage"
TAG_ROLE = "amt.role"
TAG_HYPOTHESIS_ID = "amt.hypothesis_id"
TAG_CONTRACT_HASH = "amt.contract_hash"
TAG_SPEC_HASH = "amt.spec_hash"
TAG_HARNESS_VERSION = "amt.harness_version"
TAG_CONFIG_HASH = "amt.config_hash"
TAG_CODE_VERSION = "amt.code_version"

ALL_TAGS = (
    TAG_CAMPAIGN_ID,
    TAG_STAGE,
    TAG_ROLE,
    TAG_HYPOTHESIS_ID,
    TAG_CONTRACT_HASH,
    TAG_SPEC_HASH,
    TAG_HARNESS_VERSION,
    TAG_CONFIG_HASH,
    TAG_CODE_VERSION,
)


def build_tags(
    campaign_id: str,
    stage: str,
    role: str,
    *,
    hypothesis_id: str = "",
    contract_hash: str = "",
    spec_hash: str = "",
    harness_version: str = "",
    config_hash: str = "",
    code_version: str = "",
) -> dict[str, str]:
    """Standard tag set applied to every AMT run."""
    return {
        TAG_CAMPAIGN_ID: campaign_id,
        TAG_STAGE: stage,
        TAG_ROLE: role,
        TAG_HYPOTHESIS_ID: hypothesis_id,
        TAG_CONTRACT_HASH: contract_hash,
        TAG_SPEC_HASH: spec_hash,
        TAG_HARNESS_VERSION: harness_version,
        TAG_CONFIG_HASH: config_hash,
        TAG_CODE_VERSION: code_version,
    }


def _mlflow() -> Any:
    import mlflow  # lazy: keeps unit tests offline and imports cheap

    return mlflow


def set_tracking_uri(settings: Settings | None = None) -> str:
    """Point MLflow at Databricks-managed tracking."""
    settings = settings or get_settings()
    mlflow = _mlflow()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri or "databricks")
    return mlflow.get_tracking_uri()


def get_or_create_experiment(campaign_id: str, settings: Settings | None = None) -> str:
    """One MLflow experiment per campaign: ``{root}/{campaign_id}``. Returns its id."""
    settings = settings or get_settings()
    set_tracking_uri(settings)
    mlflow = _mlflow()
    name = settings.experiment_name(campaign_id)
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return str(existing.experiment_id)
    return str(mlflow.create_experiment(name))


def log_final_run(
    *,
    campaign_id: str,
    experiment_id: str,
    frozen: Any,
    conclusion: Any,
    problem_spec: Any,
    evaluation_contract: Any,
    phase1: Any | None = None,
    code_version: str = "",
    settings: Settings | None = None,
) -> str:
    """Stage 12: log the final campaign run (lineage, QA metrics, report). Returns run id."""
    settings = settings or get_settings()
    set_tracking_uri(settings)
    mlflow = _mlflow()
    tags = build_tags(
        campaign_id,
        stage="conclusion",
        role="final",
        hypothesis_id=getattr(frozen, "hypothesis_id", ""),
        contract_hash=getattr(frozen, "contract_hash", ""),
        spec_hash=getattr(frozen, "spec_hash", ""),
        harness_version=getattr(evaluation_contract, "harness_version", ""),
        config_hash=getattr(frozen, "config_hash", ""),
        code_version=code_version,
    )
    with mlflow.start_run(experiment_id=experiment_id, run_name=f"{campaign_id}-final") as run:
        mlflow.set_tags({k: v for k, v in tags.items() if v})
        mlflow.set_tag("amt.champion_run_id", getattr(frozen, "mlflow_run_id", ""))
        mlflow.set_tag("amt.qa_decision", str(getattr(conclusion, "qa_decision", "")))
        if phase1 is not None:
            for key, value in (getattr(phase1, "qa_metrics", None) or {}).items():
                mlflow.log_metric(f"qa/{key}", float(value))
            for key, value in (getattr(phase1, "baseline_metrics", None) or {}).items():
                mlflow.log_metric(f"qa/baseline/{key}", float(value))
            for delta in getattr(phase1, "paired_deltas", None) or []:
                mlflow.log_metric(f"qa/delta/{delta.metric}", float(delta.mean_delta))
                mlflow.log_metric(f"qa/delta/{delta.metric}/ci_low", float(delta.ci_low))
                mlflow.log_metric(f"qa/delta/{delta.metric}/ci_high", float(delta.ci_high))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _dump(root / "problem_spec.json", problem_spec)
            _dump(root / "evaluation_contract.json", evaluation_contract)
            _dump(root / "frozen_champion.json", frozen)
            (root / "conclusion.md").write_text(
                getattr(conclusion, "report_markdown", "") or "", encoding="utf-8"
            )
            mlflow.log_artifacts(str(root))
        return run.info.run_id


def register_candidate(
    model_uri: str,
    name: str,
    alias: str = "candidate",
    settings: Settings | None = None,
) -> Any:
    """Register the frozen model in UC and apply an alias (Stage 12, QA=PASS only)."""
    settings = settings or get_settings()
    set_tracking_uri(settings)
    mlflow = _mlflow()
    mlflow.set_registry_uri("databricks-uc")
    version = mlflow.register_model(model_uri=model_uri, name=name)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(name=name, alias=alias, version=version.version)
    return version


def _dump(path: Path, model: Any) -> None:
    if model is None:
        return
    if hasattr(model, "model_dump_json"):
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    else:
        path.write_text(json.dumps(model, indent=2, default=str), encoding="utf-8")
