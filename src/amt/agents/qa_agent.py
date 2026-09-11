"""Stage 9/10 agent: Phase 0 triage and the QA decision. Never optimizes."""

from __future__ import annotations

from amt.agents.base import AgentBase
from amt.schemas.champion import FrozenChampion
from amt.schemas.evaluation_contract import EvaluationContract
from amt.schemas.qa import Phase0Result, Phase0Triage, Phase1Result, QAAssessment


class QAAgent(AgentBase):
    agent_key = "qa"

    def triage_phase0(
        self,
        result: Phase0Result,
        frozen: FrozenChampion,
        contract: EvaluationContract,
    ) -> Phase0Triage:
        """Decide whether a Phase-0 defect is repairable without changing the experiment."""
        agent = self.agent("qa.phase0", "qa_phase0_triage", Phase0Triage)
        return agent.invoke(
            {
                "phase0_result": result,
                "frozen_champion_summary": {
                    "champion_id": frozen.champion_id,
                    "config_hash": frozen.config_hash,
                    "family": frozen.family,
                    "contract_hash": frozen.contract_hash,
                },
                "expected_harness_version": contract.harness_version,
                "attempt": result.attempt,
            }
        )

    def decide(
        self,
        phase0: Phase0Result,
        phase1: Phase1Result,
        contract: EvaluationContract,
        frozen: FrozenChampion,
    ) -> QAAssessment:
        """Stage 10 decision against the pre-registered acceptance criteria."""
        agent = self.agent("qa.decision", "qa_decision", QAAssessment)
        return agent.invoke(
            {
                "acceptance_criteria": contract.acceptance_criteria,
                "primary_metric": contract.primary_metric,
                "cohorts": [c.model_dump() for c in contract.cohorts],
                "phase0_summary": {
                    "passed": phase0.passed,
                    "attempt": phase0.attempt,
                    "reproduction_within_tolerance": phase0.reproduction_within_tolerance,
                },
                "phase1_result": phase1,
                "validation_metrics": frozen.validation_metrics,
            }
        )
