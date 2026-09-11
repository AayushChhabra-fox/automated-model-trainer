"""Stage 4/6/7a agent: proposes configurations, interprets results, picks the champion.

The Trainer sees train and validation results only. It never receives QA data.
"""

from __future__ import annotations

from pydantic import Field

from amt.agents.base import AgentBase
from amt.schemas.common import AmtModel
from amt.schemas.evaluation_contract import EvaluationContract
from amt.schemas.experiment_config import ExperimentConfig
from amt.schemas.hypothesis import ExperimentPlan, Hypothesis
from amt.schemas.problem_spec import ProblemSpec
from amt.schemas.run_result import RunSummary, TrainingRunResult, Verdict


class TrainerInterpretation(AmtModel):
    """Observation → Interpretation → Proposed Change → Expected Effect (+ convergence)."""

    observation: str
    interpretation: str
    proposed_change: str = ""
    expected_effect: str = ""
    verdict: Verdict = "no_change"
    generalization_gap_notes: str = ""
    converged: bool = False
    convergence_reason: str = ""


class ChampionSelection(AmtModel):
    """Trainer's choice of the single best run for this hypothesis."""

    run_id: str
    selection_rationale: str
    guardrails_passed: bool = True
    cohort_regressions: list[str] = Field(default_factory=list)
    complexity_notes: str = ""
    operational_notes: str = ""


class TrainerAgent(AgentBase):
    agent_key = "trainer"

    def propose_config(
        self,
        hypothesis: Hypothesis,
        plan: ExperimentPlan,
        spec: ProblemSpec,
        contract: EvaluationContract,
        history: list[RunSummary] | None = None,
        last_result: TrainingRunResult | None = None,
        last_interpretation: TrainerInterpretation | None = None,
    ) -> ExperimentConfig:
        agent = self.agent("trainer.propose", "trainer_propose", ExperimentConfig)
        config = agent.invoke(
            {
                "hypothesis": hypothesis,
                "experiment_plan": plan,
                "problem_spec": spec,
                "primary_metric": contract.primary_metric,
                "experiment_history": [h.model_dump() for h in history or []],
                "last_result": _trainer_view(last_result),
                "last_interpretation": last_interpretation,
            }
        )
        return config.model_copy(
            update={
                "hypothesis_id": hypothesis.hypothesis_id,
                "contract_hash": contract.contract_hash,
            }
        )

    def interpret(
        self,
        result: TrainingRunResult,
        hypothesis: Hypothesis,
        plan: ExperimentPlan,
        history: list[RunSummary] | None = None,
        iteration: int = 1,
    ) -> TrainerInterpretation:
        agent = self.agent("trainer.interpret", "trainer_interpret", TrainerInterpretation)
        return agent.invoke(
            {
                "result": _trainer_view(result),
                "hypothesis": hypothesis,
                "experiment_plan": plan,
                "experiment_history": [h.model_dump() for h in history or []],
                "iteration": iteration,
                "max_iterations": plan.max_iterations,
            }
        )

    def select_champion(
        self,
        hypothesis: Hypothesis,
        plan: ExperimentPlan,
        history: list[RunSummary],
        contract: EvaluationContract,
    ) -> ChampionSelection:
        agent = self.agent("trainer.champion", "trainer_champion", ChampionSelection)
        return agent.invoke(
            {
                "hypothesis": hypothesis,
                "experiment_plan": plan,
                "runs": [h.model_dump() for h in history],
                "acceptance_criteria": contract.acceptance_criteria,
                "cohorts": [c.model_dump() for c in contract.cohorts],
            }
        )


def _trainer_view(result: TrainingRunResult | None) -> dict | None:
    """Train/validation view of a run. Structurally excludes anything QA-related."""
    if result is None:
        return None
    return {
        "run_id": result.run_id,
        "mlflow_run_id": result.mlflow_run_id,
        "config_hash": result.config_hash,
        "family": result.family,
        "status": result.status,
        "train_metrics": result.train_metrics,
        "validation_metrics": result.validation_metrics,
        "cohort_metrics": result.cohort_metrics,
        "training_curve": result.training_curve,
        "runtime_seconds": result.runtime_seconds,
        "error": result.error,
    }
