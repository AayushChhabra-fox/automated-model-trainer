"""Stage 3 / 7b / 11 agent: hypotheses, plans, final champion, scientific conclusion."""

from __future__ import annotations

from amt.agents.base import AgentBase
from amt.schemas.champion import FinalChampionSelection, HypothesisChampion
from amt.schemas.conclusion import ScientificConclusion
from amt.schemas.evaluation_contract import EvaluationContract
from amt.schemas.feature_recipe import (
    ApprovedShape,
    FeatureFitProposal,
    runtime_vocabulary,
)
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
        human_feedback: str | None = None,
        baseline_metrics: dict[str, float] | None = None,
    ) -> HypothesisBacklog:
        """The campaign backlog. Presented for human approval before anything is planned.

        The runtime's vocabulary and the baseline's numbers go in as context:
        the previous campaign's backlog proposed transforms that do not exist
        and effect sizes untethered to what the baseline actually scored.
        """
        agent = self.agent("planner.hypotheses", "planner_hypotheses", HypothesisBacklog)
        return agent.invoke(
            {
                "problem_spec": spec,
                "evaluation_contract": contract,
                "runtime_vocabulary": runtime_vocabulary(),
                "baseline_validation_metrics": baseline_metrics or {},
                # Summaries only: `rows` holds up to 200 sampled rows per probe,
                # which would be most of the prompt and none of the signal.
                "data_investigation": [
                    {"action": o.action, "target": o.target, "summary": o.summary}
                    for o in spec.investigation_notes
                ],
                "experiment_history": [h.model_dump() for h in history or []],
                "existing_hypotheses": [h.model_dump() for h in existing or []],
                "human_feedback": human_feedback,
            }
        )

    def build_plan(
        self,
        hypothesis: Hypothesis,
        spec: ProblemSpec,
        contract: EvaluationContract,
        baseline_reference: str = "",
        history: list[RunSummary] | None = None,
        shape: ApprovedShape | None = None,
    ) -> ExperimentPlan:
        """Bound the Trainer for one hypothesis, under an already-approved shape.

        ``shape`` is an input, not an output: HC_FEATURE runs before selection,
        so the plan is written knowing what the model is allowed to see.
        """
        agent = self.agent("planner.plan", "planner_plan", ExperimentPlan)
        plan = agent.invoke(
            {
                "hypothesis": hypothesis,
                "problem_spec": spec,
                "evaluation_contract": contract,
                "baseline_reference": baseline_reference,
                "experiment_history": [h.model_dump() for h in history or []],
                "approved_shape": shape,
            }
        )
        stamped = {
            "hypothesis_id": hypothesis.hypothesis_id,
            "primary_metric": contract.primary_metric,
            "contract_hash": contract.contract_hash,
            "baseline_reference": plan.baseline_reference or baseline_reference,
            "family": hypothesis.family or plan.family,
        }
        if shape is not None:
            stamped |= {
                "family": shape.family,
                "feature_recipe": shape.feature_recipe,
                "fit_spec": shape.fit_spec,
                "feature_fit_hash": shape.feature_fit_hash,
            }
        return plan.model_copy(update=stamped)

    def propose_feature_fit(
        self,
        hypothesis: Hypothesis,
        spec: ProblemSpec,
        contract: EvaluationContract,
        family: str = "",
        previous: FeatureFitProposal | None = None,
        human_feedback: str | None = None,
    ) -> FeatureFitProposal:
        """Declare what goes into a tabular model. Presented for human approval.

        Takes no ``ExperimentPlan``: this runs before any hypothesis is planned,
        and the plan never carried anything the recipe needs that the
        hypothesis and the contract do not.
        """
        agent = self.agent("planner.feature_fit", "planner_feature_fit", FeatureFitProposal)
        proposal = agent.invoke(
            {
                "hypothesis": hypothesis,
                "problem_spec": spec,
                "data_sources": [d.model_dump() for d in spec.data_sources],
                "train_split": contract.train_split,
                "validation_split": contract.validation_split,
                "label_definition": contract.label_definition,
                "candidate_generation_rules": contract.candidate_generation_rules,
                "runtime_vocabulary": runtime_vocabulary(),
                "max_candidate_rows": self.settings.max_candidate_rows,
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
