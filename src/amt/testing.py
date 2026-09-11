"""Deterministic fakes: a canned-schema LLM and the mock executor pairing.

Used by the unit/integration tests and by ``amt ... --mock`` for offline dry runs.
Nothing here touches the network.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from amt.agents.problem_spec_agent import (
    EvaluationContractDraft,
    ProbeRequest,
    SplitSpec,
)
from amt.agents.trainer_agent import ChampionSelection, TrainerInterpretation
from amt.executor.local_mock import LocalMockExecutor, QaOutcome
from amt.schemas.champion import FinalChampionSelection
from amt.schemas.common import CohortDef, MetricDef
from amt.schemas.conclusion import ProductInterpretation, ScientificConclusion
from amt.schemas.evaluation_contract import AcceptanceCriteria, DatasetDefinition, TemporalRange
from amt.schemas.experiment_config import ExperimentConfig
from amt.schemas.hypothesis import ExperimentPlan, Hypothesis, HypothesisBacklog
from amt.schemas.problem_spec import (
    BaselineDefinition,
    DataSource,
    ObjectiveFunction,
    ProblemDescription,
    ProblemSpec,
    ProductMetric,
)
from amt.schemas.qa import Phase0Triage, QAAssessment, QADecision

PRIMARY_METRIC = "ndcg@10"
INTERACTIONS_TABLE = "main.amt.interactions"


def _text(messages: Any) -> str:
    if isinstance(messages, str):
        return messages
    parts = []
    for message in messages or []:
        content = getattr(message, "content", message)
        parts.append(content if isinstance(content, str) else str(content))
    return "\n".join(parts)


def _find_all(pattern: str, text: str) -> list[str]:
    return re.findall(pattern, text)


class FakeLLM:
    """Stands in for ``ChatAnthropic``; returns canned instances of the requested schema."""

    def __init__(self, responder: CannedResponder, model: str = "fake-model") -> None:
        self.responder = responder
        self.model = model

    def with_structured_output(self, schema: type[BaseModel], **_: Any) -> _FakeStructured:
        return _FakeStructured(self, schema)


class _FakeStructured:
    def __init__(self, llm: FakeLLM, schema: type[BaseModel]) -> None:
        self.llm = llm
        self.schema = schema

    def invoke(self, messages: Any, **_: Any) -> BaseModel:
        return self.llm.responder(self.schema, _text(messages))


class CannedResponder:
    """Produces a plausible instance of each schema the agents ask for.

    Stateful on purpose: configs differ between iterations so ``config_hash`` changes,
    and the trainer reports convergence after ``converge_after`` interpretations.
    """

    def __init__(
        self,
        qa_outcome: QaOutcome = "PASS",
        converge_after: int = 3,
        n_hypotheses: int = 1,
        probe_steps: int = 2,
    ) -> None:
        self.qa_outcome: QaOutcome = qa_outcome
        self.converge_after = converge_after
        self.n_hypotheses = n_hypotheses
        self.probe_steps = probe_steps
        self.calls: list[str] = []
        self.counts: dict[str, int] = {}

    def _bump(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def __call__(self, schema: type[BaseModel], text: str) -> BaseModel:
        self.calls.append(schema.__name__)
        handler = {
            ProbeRequest: self._probe,
            ProblemSpec: self._problem_spec,
            EvaluationContractDraft: self._contract,
            HypothesisBacklog: self._backlog,
            ExperimentPlan: self._plan,
            ExperimentConfig: self._config,
            TrainerInterpretation: self._interpretation,
            ChampionSelection: self._champion,
            FinalChampionSelection: self._final_champion,
            Phase0Triage: self._triage,
            QAAssessment: self._assessment,
            ScientificConclusion: self._conclusion,
        }.get(schema)
        if handler is None:
            raise AssertionError(f"FakeLLM has no canned response for {schema.__name__}")
        return handler(text)

    # --- Stage 1/2 --------------------------------------------------------------
    def _probe(self, text: str) -> ProbeRequest:
        step = self._bump("probe")
        if step > self.probe_steps:
            return ProbeRequest(action="done", reason="enough context to specify the problem")
        action = "describe_table" if step == 1 else "profile"
        return ProbeRequest(
            action=action,  # type: ignore[arg-type]
            target=INTERACTIONS_TABLE,
            reason="need schema, grain and temporal coverage",
        )

    def _problem_spec(self, text: str) -> ProblemSpec:
        revision = self._bump("spec")
        feedback = _extract_section(text, "human_feedback")
        assumptions = ["Implicit feedback only; no explicit ratings available."]
        if feedback and feedback != "null":
            assumptions.append(f"Incorporated human feedback: {feedback}")
        return ProblemSpec(
            problem_description=ProblemDescription(
                product_behavior_to_improve="Homepage row ranking engagement.",
                decision="Which items to rank into the top positions of each homepage row.",
                prediction_target="Probability the user starts the item in this session.",
                serving_context="Batch-scored candidates re-ranked online at request time.",
                product_constraints=["p95 latency < 50ms", "no cold-start regression"],
            ),
            data_sources=[
                DataSource(
                    name="interactions",
                    table=INTERACTIONS_TABLE,
                    relevant_columns=["user_id", "item_id", "event_ts", "event_type"],
                    grain="one row per user-item interaction event",
                    join_keys=["user_id", "item_id"],
                    timestamp_columns=["event_ts"],
                    filters=["event_type IN ('start','complete')"],
                    label_construction="positive when a start occurs within the session window",
                )
            ],
            product_metrics=[
                ProductMetric(
                    name="ctr", definition="starts / impressions", measurability="offline_proxy"
                ),
                ProductMetric(
                    name="retention_d7",
                    definition="7-day return rate",
                    measurability="online_only",
                ),
            ],
            offline_metrics=[
                MetricDef(
                    name=PRIMARY_METRIC,
                    definition="Normalized discounted cumulative gain at rank 10.",
                    k=10,
                    implementation="harness:ndcg_at_k",
                    is_primary=True,
                    tolerance=1e-3,
                ),
                MetricDef(
                    name="recall@10",
                    definition="Fraction of held-out positives in the top 10.",
                    k=10,
                    implementation="harness:recall_at_k",
                ),
            ],
            primary_offline_metric=PRIMARY_METRIC,
            objective=ObjectiveFunction(
                name="BCE(next_item_prediction)",
                description="Binary cross-entropy over sampled negatives.",
                relation_to_offline_metric="Higher likelihood of the true next item lifts NDCG@10.",
                expected_product_outcome="More relevant top-of-row items, higher start rate.",
            ),
            baseline=BaselineDefinition(
                kind="popularity",
                name="trailing-7d popularity",
                description="Rank by trailing 7-day start counts.",
                evaluated_with_same_contract=True,
            ),
            assumptions=assumptions,
            open_questions=["Is the candidate set identical online and offline?"],
            revision=revision - 1,
        )

    def _contract(self, text: str) -> EvaluationContractDraft:
        self._bump("contract")
        return EvaluationContractDraft(
            dataset=DatasetDefinition(
                source_tables=[INTERACTIONS_TABLE],
                snapshot_id="snap-2026-08-01",
                dataset_version="1",
                population_eligibility="users with >=3 interactions in the training window",
                temporal_range=TemporalRange(start="2026-05-01", end="2026-08-01"),
                exclusions=["internal test accounts"],
                label_construction="start within the session window",
            ),
            train=SplitSpec(
                table="main.amt.split_train",
                description="first 10 weeks",
                temporal_range=("2026-05-01", "2026-07-10"),
            ),
            validation=SplitSpec(
                table="main.amt.split_validation",
                description="following 2 weeks",
                temporal_range=("2026-07-11", "2026-07-24"),
            ),
            qa=SplitSpec(
                table="main.amt.split_qa",
                description="final 1 week holdout",
                temporal_range=("2026-07-25", "2026-08-01"),
            ),
            split_strategy="temporal",
            user_population="eligible users active in the evaluation window",
            item_population="catalog items available for the full window",
            label_definition="start within the session window",
            candidate_generation_rules="top 500 trailing-popularity candidates per request",
            cohorts=[
                CohortDef(name="cold_start", description="users with <5 prior interactions"),
                CohortDef(name="tail_items", description="items outside the top 20% by volume"),
            ],
            metrics=[
                MetricDef(
                    name=PRIMARY_METRIC,
                    definition="NDCG at rank 10 over the fixed candidate set.",
                    k=10,
                    implementation="harness:ndcg_at_k",
                    is_primary=True,
                    tolerance=1e-3,
                ),
                MetricDef(
                    name="recall@10",
                    definition="Recall at rank 10.",
                    k=10,
                    implementation="harness:recall_at_k",
                ),
            ],
            primary_metric=PRIMARY_METRIC,
            acceptance_criteria=AcceptanceCriteria(
                primary_metric=PRIMARY_METRIC,
                min_absolute_improvement=0.005,
                min_relative_improvement=0.02,
                require_ci_excludes_zero=True,
                max_cohort_regression=0.02,
                max_latency_ms_p95=50.0,
            ),
        )

    # --- Stage 3 -----------------------------------------------------------------
    def _backlog(self, text: str) -> HypothesisBacklog:
        self._bump("backlog")
        hypotheses = [
            Hypothesis(
                hypothesis_id=f"hyp-{i + 1}",
                title=f"Sequential context improves ranking ({i + 1})",
                proposed_change="Replace the popularity baseline with a SASRec sequence model.",
                why_it_should_improve="Recent watch order carries intent the counts discard.",
                motivating_evidence="Sessions show strong temporal adjacency between titles.",
                experiment_design="Train SASRec on the frozen train split; compare on validation.",
                metric_expected_to_improve=PRIMARY_METRIC,
                falsification_criteria=(
                    f"No meaningful {PRIMARY_METRIC} gain over the popularity baseline, "
                    "or a cold-start regression."
                ),
                expected_effect_size="+0.02 to +0.05 NDCG@10",
                confidence=0.6 - 0.1 * i,
                priority=i + 1,
            )
            for i in range(self.n_hypotheses)
        ]
        return HypothesisBacklog(
            hypotheses=hypotheses,
            prioritization_rationale="Highest expected value per compute minute first.",
        )

    def _plan(self, text: str) -> ExperimentPlan:
        self._bump("plan")
        hypothesis_ids = _find_all(r'"hypothesis_id":\s*"([^"]+)"', text)
        return ExperimentPlan(
            hypothesis_id=hypothesis_ids[0] if hypothesis_ids else "hyp-1",
            hypothesis_summary="Sequential modeling beats popularity for top-of-row ranking.",
            rationale="Tests whether order-aware context adds signal beyond frequency.",
            expected_effect="+0.02 to +0.05 NDCG@10 on validation.",
            primary_metric=PRIMARY_METRIC,
            secondary_metrics=["recall@10", "coverage"],
            permitted_model_changes=["hidden_dim", "num_layers", "heads", "dropout"],
            permitted_feature_changes=["recency", "metadata"],
            search_space={
                "hidden_dim": [64, 128, 256],
                "num_layers": [2, 3],
                "learning_rate": [1e-4, 3e-4],
            },
            success_criteria=f"+0.01 absolute {PRIMARY_METRIC} over baseline on validation.",
            falsification_criteria="No improvement after 3 informed iterations.",
            max_iterations=3,
            compute_budget_minutes=60.0,
        )

    # --- Stage 4/6/7 ----------------------------------------------------------------
    def _config(self, text: str) -> ExperimentConfig:
        i = self._bump("config")
        return ExperimentConfig(
            family="sasrec",
            model={
                "hidden_dim": 64 * i,
                "num_layers": 2,
                "heads": 4,
                "dropout": 0.2,
                "max_seq_len": 50,
            },
            training={"learning_rate": 3e-4 / i, "batch_size": 1024, "epochs": 10},
            loss={"type": "bce", "negatives": 100},
            features={"metadata": True, "recency": i > 1},
            rationale=(
                "Start from a small sequence model, then widen capacity only while validation "
                "keeps improving."
            ),
            changes_from_previous=[] if i == 1 else ["wider hidden_dim", "lower learning rate"],
        )

    def _interpretation(self, text: str) -> TrainerInterpretation:
        i = self._bump("interpret")
        converged = i >= self.converge_after
        return TrainerInterpretation(
            observation="Validation NDCG@10 improved, with a widening train/validation gap.",
            interpretation=(
                "Capacity is no longer the binding constraint; further widening mostly adds "
                "overfitting."
                if converged
                else "Additional capacity is still converting into validation gains."
            ),
            proposed_change="" if converged else "Widen hidden_dim and lower the learning rate.",
            expected_effect="" if converged else "Small further NDCG@10 gain.",
            verdict="improved",
            generalization_gap_notes="Gap grows ~0.004 per iteration.",
            converged=converged,
            convergence_reason=(
                "Validation improvement fell below the meaningful threshold." if converged else ""
            ),
        )

    def _champion(self, text: str) -> ChampionSelection:
        self._bump("champion")
        run_ids = _find_all(r'"run_id":\s*"([^"]+)"', text)
        return ChampionSelection(
            run_id=run_ids[-1] if run_ids else "",
            selection_rationale=(
                "Best validation NDCG@10 with no cohort regression and acceptable complexity."
            ),
            guardrails_passed=True,
            cohort_regressions=[],
            complexity_notes="128-dim, 2-layer model; inference well inside the latency budget.",
            operational_notes="Artifact ~48MB; p95 latency 24ms in the harness.",
        )

    def _final_champion(self, text: str) -> FinalChampionSelection:
        self._bump("final_champion")
        ids = _find_all(r'"hypothesis_id":\s*"([^"]+)"', text)
        return FinalChampionSelection(
            hypothesis_id=ids[0] if ids else "hyp-1",
            rationale="Strongest validation evidence without guardrail violations.",
        )

    # --- Stage 9/10/11 ------------------------------------------------------------
    def _triage(self, text: str) -> Phase0Triage:
        self._bump("triage")
        return Phase0Triage(
            fixable=True,
            defects=["preprocessing artifact missing from the logged model"],
            fix_plan=(
                "Re-log the pyfunc wrapper including the preprocessing step and rerun Phase 0. "
                "No change to architecture, features, hyperparameters or selection."
            ),
            rationale="A packaging defect, not a modeling defect.",
        )

    def _assessment(self, text: str) -> QAAssessment:
        self._bump("assessment")
        decision = QADecision(self.qa_outcome)
        rationale = {
            QADecision.PASS: (
                "Paired holdout improvement exceeds the pre-registered threshold and the 95% CI "
                "excludes zero, with no cohort regression."
            ),
            QADecision.FAIL_HYPOTHESIS: (
                "The candidate validated well but showed no significant holdout separation from "
                "the baseline; the CI includes zero."
            ),
            QADecision.SPEC_REVIEW: (
                "Candidate and baseline both degrade on the holdout and the validation-to-QA "
                "divergence far exceeds sampling noise: evidence of an evaluation-design or "
                "label problem rather than a modeling failure."
            ),
        }[decision]
        return QAAssessment(
            decision=decision,
            rationale=rationale,
            acceptance_criteria_met={
                "min_absolute_improvement": decision == QADecision.PASS,
                "ci_excludes_zero": decision != QADecision.FAIL_HYPOTHESIS,
                "no_cohort_regression": decision == QADecision.PASS,
            },
            evidence=["paired bootstrap 95% CI", "cohort deltas", "operational measurements"],
            spec_review_reasons=(
                ["label timing relative to the serving decision point"]
                if decision == QADecision.SPEC_REVIEW
                else []
            ),
            generalization_gap_notes="Validation-to-QA gap consistent with the training gap."
            if decision == QADecision.PASS
            else "Validation-to-QA gap larger than the training gap.",
        )

    def _conclusion(self, text: str) -> ScientificConclusion:
        self._bump("conclusion")
        decision = QADecision(self.qa_outcome)
        recommendation = {
            QADecision.PASS: "proceed_to_ab_test",
            QADecision.FAIL_HYPOTHESIS: "hypothesis_rejected",
            QADecision.SPEC_REVIEW: "evaluation_contract_requires_revision",
        }[decision]
        return ScientificConclusion(
            campaign_id="",
            problem="Improve homepage row ranking engagement.",
            hypothesis="Sequential context improves ranking over a popularity baseline.",
            experiments_summary=[
                "SASRec 64d: +0.02 NDCG@10 over baseline on validation.",
                "SASRec 128d: further +0.01; generalization gap widening.",
            ],
            best_model="sasrec (128d, 2 layers)",
            why_selected="Best validation NDCG@10 without cohort regressions.",
            validation_evidence="Consistent improvement over the popularity baseline.",
            qa_evidence="Paired holdout comparison with bootstrap confidence intervals.",
            cohort_effects="Cold-start cohort moves less than the overall population.",
            operational_considerations="48MB artifact; p95 latency 24ms; well inside budget.",
            product_interpretation=ProductInterpretation(
                expected_offline_impact="Modest offline ranking-quality improvement.",
                assumptions_and_risks=["Offline candidate set matches the online one."],
            ),
            qa_decision=decision,
            recommendation=recommendation,  # type: ignore[arg-type]
            recommendation_rationale="Follows directly from the QA evidence.",
            negative_result=decision != QADecision.PASS,
            report_markdown="# Scientific Conclusion\n\nSee structured fields.\n",
        )


def _extract_section(text: str, name: str) -> str:
    match = re.search(rf"^## {re.escape(name)}\n\n(.*?)(?=\n## |\Z)", text, re.S | re.M)
    return match.group(1).strip() if match else ""


def fake_llm_factory(responder: CannedResponder) -> Callable[[str], FakeLLM]:
    """Build an ``llm_factory`` usable anywhere a real one is accepted."""

    def factory(model: str) -> FakeLLM:
        return FakeLLM(responder, model)

    return factory


def mock_stack(
    qa_outcome: QaOutcome = "PASS",
    phase0_invalid_first_attempt: bool = False,
    converge_after: int = 3,
    n_hypotheses: int = 1,
) -> tuple[LocalMockExecutor, Callable[[str], FakeLLM], CannedResponder]:
    """``(executor, llm_factory, responder)`` configured consistently for a dry run."""
    responder = CannedResponder(
        qa_outcome=qa_outcome, converge_after=converge_after, n_hypotheses=n_hypotheses
    )
    executor = LocalMockExecutor(
        qa_outcome=qa_outcome,
        phase0_invalid_first_attempt=phase0_invalid_first_attempt,
        converge_after=converge_after,
    )
    return executor, fake_llm_factory(responder), responder
