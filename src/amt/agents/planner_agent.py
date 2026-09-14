"""Stage 3 / 7b / 11 agent: hypotheses, plans, final champion, scientific conclusion."""

from __future__ import annotations

from amt.agents.base import AgentBase
from amt.schemas.champion import FinalChampionSelection, HypothesisChampion
from amt.schemas.conclusion import ScientificConclusion
from amt.schemas.evaluation_contract import EvaluationContract
from amt.schemas.feature_recipe import FeatureFitProposal
from amt.schemas.hypothesis import ExperimentPlan, Hypothesis, HypothesisBacklog
from amt.schemas.problem_spec import ProblemSpec
from amt.schemas.qa import Phase0Result, Phase1Result, QAAssessment
from amt.schemas.run_result import RunSummary


class PlannerAgent(AgentBase):
    """Generates and prioritizes falsifiable hypotheses. Never trains."""

    agent_key = "planner"

    def propose_hypotheses(
        self,
        spec: ProblemSpec,
        contract: EvaluationContract,
        history: list[RunSummary] | None = None,
        existing: list[Hypothesis] | None = None,
    ) -> HypothesisBacklog:
        agent = self.agent("planner.hypotheses", "planner_hypotheses", HypothesisBacklog)
        return agent.invoke(
            {
                "problem_spec": spec,
                "evaluation_contract": contract,
                "experiment_history": [h.model_dump() for h in history or []],
                "existing_hypotheses": [h.model_dump() for h in existing or []],
            }
        )

    def build_plan(
        self,
        hypothesis: Hypothesis,
        spec: ProblemSpec,
        contract: EvaluationContract,
        baseline_reference: str = "",
        history: list[RunSummary] | None = None,
    ) -> ExperimentPlan:
        agent = self.agent("planner.plan", "planner_plan", ExperimentPlan)
        plan = agent.invoke(
            {
                "hypothesis": hypothesis,
                "problem_spec": spec,
                "evaluation_contract": contract,
                "baseline_reference": baseline_reference,
                "experiment_history": [h.model_dump() for h in history or []],
            }
        )
        return plan.model_copy(
            update={
                "hypothesis_id": hypothesis.hypothesis_id,
                "primary_metric": contract.primary_metric,
                "contract_hash": contract.contract_hash,
                "baseline_reference": plan.baseline_reference or baseline_reference,
            }
        )

    def propose_feature_fit(
        self,
        hypothesis: Hypothesis,
        plan: ExperimentPlan,
        spec: ProblemSpec,
        contract: EvaluationContract,
        family: str = "",
        previous: FeatureFitProposal | None = None,
        human_feedback: str | None = None,
    ) -> FeatureFitProposal:
        """Declare what goes into a tabular model. Presented for human approval."""
        agent = self.agent("planner.feature_fit", "planner_feature_fit", FeatureFitProposal)
        proposal = agent.invoke(
            {
                "hypothesis": hypothesis,
                "experiment_plan": plan,
                "problem_spec": spec,
                "data_sources": [d.model_dump() for d in spec.data_sources],
                "train_split": contract.train_split,
                "validation_split": contract.validation_split,
                "label_definition": contract.label_definition,
                "candidate_generation_rules": contract.candidate_generation_rules,
                "target_family": family,
                "previous_proposal": previous,
                "human_feedback": human_feedback,
            }
        )
        return proposal.model_copy(update={"family": proposal.family or family})

    def select_final_champion(
        self,
        champions: list[HypothesisChampion],
        spec: ProblemSpec,
        contract: EvaluationContract,
    ) -> FinalChampionSelection:
        agent = self.agent(
            "planner.final_champion", "planner_final_champion", FinalChampionSelection
        )
        return agent.invoke(
            {
                "hypothesis_champions": [c.model_dump() for c in champions],
                "acceptance_criteria": contract.acceptance_criteria,
                "primary_metric": contract.primary_metric,
                "product_constraints": spec.problem_description.product_constraints,
            }
        )

    def write_conclusion(
        self,
        campaign_id: str,
        spec: ProblemSpec,
        contract: EvaluationContract,
        history: list[RunSummary],
        champions: list[HypothesisChampion],
        assessment: QAAssessment | None,
        phase0: Phase0Result | None,
        phase1: Phase1Result | None,
    ) -> ScientificConclusion:
        agent = self.agent("planner.conclusion", "planner_conclusion", ScientificConclusion)
        conclusion = agent.invoke(
            {
                "campaign_id": campaign_id,
                "problem_spec": spec,
                "evaluation_contract": contract,
                "experiment_history": [h.model_dump() for h in history],
                "hypothesis_champions": [c.model_dump() for c in champions],
                "qa_assessment": assessment,
                "qa_phase0": phase0,
                "qa_phase1": phase1,
            }
        )
        return conclusion.model_copy(
            update={
                "campaign_id": campaign_id,
                "contract_hash": contract.contract_hash,
                "qa_decision": assessment.decision if assessment else conclusion.qa_decision,
            }
        )
