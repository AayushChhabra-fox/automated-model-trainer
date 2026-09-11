"""Executor protocol (PLAN §5).

Every implementation must be deterministic from the orchestrator's point of view: it
receives frozen contracts and returns compact Pydantic results. Implementations are
responsible for asserting ``contract_hash`` and ``harness_version`` on the compute side.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from amt.schemas.champion import FrozenChampion
from amt.schemas.evaluation_contract import EvaluationContract, SplitMaterialization
from amt.schemas.experiment_config import ExperimentConfig
from amt.schemas.hypothesis import ExperimentPlan
from amt.schemas.problem_spec import ProblemSpec
from amt.schemas.qa import Phase0Result, Phase1Result
from amt.schemas.run_result import TrainingRunResult


class ExecutorError(RuntimeError):
    """Raised when a compute job cannot be submitted, or fails irrecoverably."""


@runtime_checkable
class Executor(Protocol):
    """Compute backend contract. Implemented by ``DatabricksExecutor`` and ``LocalMockExecutor``."""

    def materialize_splits(self, contract: EvaluationContract) -> SplitMaterialization:
        """Materialize the frozen train/validation/qa splits as immutable Delta tables."""
        ...

    def run_baseline(
        self,
        spec: ProblemSpec,
        contract: EvaluationContract,
        split: Literal["validation", "qa"],
    ) -> TrainingRunResult:
        """Evaluate the baseline defined by the spec under the same EvaluationContract."""
        ...

    def run_training(
        self,
        config: ExperimentConfig,
        contract: EvaluationContract,
        plan: ExperimentPlan,
    ) -> TrainingRunResult:
        """Train one configuration on `train`, evaluate on `validation`. Never reads `qa`."""
        ...

    def run_qa_phase0(
        self,
        frozen: FrozenChampion,
        contract: EvaluationContract,
    ) -> Phase0Result:
        """QA Phase 0: integrity & reproducibility. Does not observe QA-split performance."""
        ...

    def run_qa_phase1(
        self,
        frozen: FrozenChampion,
        baseline_ref: str,
        contract: EvaluationContract,
    ) -> Phase1Result:
        """QA Phase 1: holdout evaluation of the frozen champion and the baseline."""
        ...
