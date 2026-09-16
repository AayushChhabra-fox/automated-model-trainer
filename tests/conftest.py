"""Shared fixtures. Nothing here touches the network: fake LLM + mock executor only."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from langgraph.checkpoint.memory import MemorySaver

from amt.agents.problem_spec_agent import draft_to_contract
from amt.executor.local_mock import LocalMockExecutor
from amt.graph.build import build_graph
from amt.schemas.champion import FrozenChampion
from amt.schemas.evaluation_contract import EvaluationContract
from amt.schemas.experiment_config import ExperimentConfig
from amt.schemas.problem_spec import ProblemSpec
from amt.settings import Settings
from amt.testing import (
    CannedResponder,
    FakeLLM,
    fake_llm_factory,
    mock_stack,
)

__all__ = ["FakeLLM"]


class _StubBackend:
    """Stands in for xgboost/lightgbm so the recipe path is tested without them."""

    name = "stub"
    artifact = "booster.json"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def fit(self, train, validation, params, rounds, early_stopping_rounds, ranker):
        self.calls.append(
            {
                "params": dict(params),
                "ranker": ranker,
                "rounds": rounds,
                "early_stopping_rounds": early_stopping_rounds,
                "n_rows": len(train["X"]),
                "positives": float(np.sum(train["y"])),
                "group_sizes": list(train["group_sizes"]),
                "n_val_rows": None if validation is None else len(validation["X"]),
            }
        )
        metric = params.get("eval_metric") or params.get("metric")
        curve = {"train": {metric: [0.5, 0.4]}, "validation": {metric: [0.6, 0.55]}}
        return {"weights": [1.0] * train["X"].shape[1]}, curve, 2

    def predict(self, booster, frame):
        return frame.to_numpy(dtype=float) @ np.asarray(booster["weights"])

    def save(self, booster, path):
        import os

        with open(os.path.join(path, self.artifact), "w", encoding="utf-8") as fh:
            json.dump(booster, fh)

    def load(self, path):
        import os

        with open(os.path.join(path, self.artifact), encoding="utf-8") as fh:
            return json.load(fh)


@pytest.fixture
def stub_backend(monkeypatch):
    from amt.jobs.models import tabular_common

    stub = _StubBackend()
    monkeypatch.setitem(tabular_common.BACKENDS, "xgboost", stub)
    monkeypatch.setitem(tabular_common.BACKENDS, "lightgbm", stub)
    return stub


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Settings isolated from the developer's environment and home directory."""
    return Settings(
        _env_file=None,
        AMT_CHECKPOINT_DB=str(tmp_path / "checkpoints.sqlite"),
        AMT_CATALOG="main",
        AMT_SCHEMA="amt",
    )


@pytest.fixture
def responder() -> CannedResponder:
    """The canned-schema brain behind :class:`FakeLLM`."""
    return CannedResponder()


@pytest.fixture
def llm_factory(responder: CannedResponder):
    """``llm_factory`` returning a FakeLLM for any model name."""
    return fake_llm_factory(responder)


@pytest.fixture
def mock_executor() -> LocalMockExecutor:
    return LocalMockExecutor()


@pytest.fixture
def problem_spec(responder: CannedResponder) -> ProblemSpec:
    return responder(ProblemSpec, "")


@pytest.fixture
def evaluation_contract(
    responder: CannedResponder, problem_spec: ProblemSpec
) -> EvaluationContract:
    from amt.agents.problem_spec_agent import EvaluationContractDraft

    draft = responder(EvaluationContractDraft, "")
    return draft_to_contract(draft, spec_hash=problem_spec.spec_hash)


@pytest.fixture
def experiment_config(responder: CannedResponder) -> ExperimentConfig:
    return responder(ExperimentConfig, "")


@pytest.fixture
def frozen_champion(
    evaluation_contract: EvaluationContract, experiment_config: ExperimentConfig
) -> FrozenChampion:
    return FrozenChampion(
        hypothesis_id="hyp-1",
        run_id="run-1",
        mlflow_run_id="mlflow-1",
        config=experiment_config,
        config_hash=experiment_config.config_hash,
        family=experiment_config.family,
        contract_hash=evaluation_contract.contract_hash,
        harness_version=evaluation_contract.harness_version,
        validation_metrics={f"val/{evaluation_contract.primary_metric}": 0.36},
    )


@pytest.fixture
def campaign(settings: Settings):
    """Factory building a compiled graph plus its mock stack.

    Returns ``(graph, config, executor, responder)``.
    """

    def _factory(
        campaign_id: str = "test-campaign",
        qa_outcome: str = "PASS",
        phase0_invalid_first_attempt: bool = False,
        proxy_correlation: float | None = None,
        converge_after: int = 3,
        n_hypotheses: int = 1,
        executor: Any | None = None,
        family: str = "sasrec",
        structural_drift_on_iteration: int = 0,
        evaluation_mode: str = "next_item",
        gate_champions: bool = True,
    ):
        mock_executor, llm_factory, canned = mock_stack(
            qa_outcome=qa_outcome,  # type: ignore[arg-type]
            phase0_invalid_first_attempt=phase0_invalid_first_attempt,
            proxy_correlation=proxy_correlation,
            converge_after=converge_after,
            n_hypotheses=n_hypotheses,
            family=family,
            structural_drift_on_iteration=structural_drift_on_iteration,
            evaluation_mode=evaluation_mode,
        )
        executor = executor or mock_executor
        graph = build_graph(
            executor=executor,
            llm_factory=llm_factory,
            settings=settings,
            checkpointer=MemorySaver(),
            use_mlflow=False,
            gate_champions=gate_champions,
        )
        config = {"configurable": {"thread_id": campaign_id}, "recursion_limit": 150}
        return graph, config, executor, canned

    return _factory


def pending_interrupt(graph: Any, config: dict) -> dict | None:
    """Payload of the interrupt the graph is currently suspended on (or None)."""
    snapshot = graph.get_state(config)
    for task in getattr(snapshot, "tasks", ()) or ():
        for interrupt in getattr(task, "interrupts", ()) or ():
            value = getattr(interrupt, "value", None)
            if isinstance(value, dict):
                return value
    return None


def graph_state(graph: Any, config: dict) -> dict:
    return dict(graph.get_state(config).values or {})
