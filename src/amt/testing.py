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
from amt.schemas.evaluation_contract import (
    AcceptanceCriteria,
    DatasetDefinition,
    SplitPolicy,
    TemporalRange,
)
from amt.schemas.experiment_config import ExperimentConfig
from amt.schemas.feature_recipe import (
    CandidateGeneration,
    FeatureFitProposal,
    FeatureRecipe,
    FeatureTransform,
    FitSpec,
    LabelSpec,
    RecipeSource,
)
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

# `grouped_rerank` fixtures: a container rail with four visible tiles, scored
# on the conversion label against the order the incumbent rendered.
GROUPED_METRIC = "ndcg@4"
IMPRESSIONS_TABLE = "main.amt.impressions"
GROUP_KEY = ["session_id", "page_id", "parent_id"]


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
    """Stands in for any provider's chat model; returns canned instances of the schema."""

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
        family: str = "sasrec",
        structural_drift_on_iteration: int = 0,
        evaluation_mode: str = "next_item",
    ) -> None:
        self.qa_outcome: QaOutcome = qa_outcome
        self.converge_after = converge_after
        self.n_hypotheses = n_hypotheses
        self.probe_steps = probe_steps
        # Which family the canned planner/trainer aim at; a tabular one exercises
        # the feature/fit checkpoint.
        self.family = family
        # On this config call the Trainer asks for a wider recipe and a different
        # objective -- the ask the clamp exists to refuse.
        self.structural_drift_on_iteration = structural_drift_on_iteration
        # `grouped_rerank` swaps the whole framing: one render per row, the
        # logged order as the baseline, and position-bias weighting.
        self.evaluation_mode = evaluation_mode
        self.calls: list[str] = []
        self.counts: dict[str, int] = {}

    def _bump(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    @property
    def grouped(self) -> bool:
        return self.evaluation_mode == "grouped_rerank"

    @property
    def metric(self) -> str:
        return GROUPED_METRIC if self.grouped else PRIMARY_METRIC

    @property
    def table(self) -> str:
        return IMPRESSIONS_TABLE if self.grouped else INTERACTIONS_TABLE

    def __call__(self, schema: type[BaseModel], text: str) -> BaseModel:
        self.calls.append(schema.__name__)
        handler = {
            ProbeRequest: self._probe,
            ProblemSpec: self._problem_spec,
            EvaluationContractDraft: self._contract,
            HypothesisBacklog: self._backlog,
            ExperimentPlan: self._plan,
            FeatureFitProposal: self._feature_fit,
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
                    table=self.table,
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
                    name=self.metric,
                    definition="Normalized discounted cumulative gain.",
                    k=int(self.metric.split("@")[1]),
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
            primary_offline_metric=self.metric,
            evaluation_mode=self.evaluation_mode,  # type: ignore[arg-type]
            objective=ObjectiveFunction(
                name="BCE(next_item_prediction)",
                description="Binary cross-entropy over sampled negatives.",
                relation_to_offline_metric="Higher likelihood of the true next item lifts NDCG@10.",
                expected_product_outcome="More relevant top-of-row items, higher start rate.",
            ),
            baseline=BaselineDefinition(
                kind="logged_order" if self.grouped else "popularity",
                name="the incumbent's own order" if self.grouped else "trailing-7d popularity",
                description="Rank by the slot the container rendered."
                if self.grouped
                else "Rank by trailing 7-day start counts.",
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
                source_table=self.table,
                source_tables=[self.table],
                group_key=GROUP_KEY if self.grouped else [],
                position_column="item_index" if self.grouped else "",
                label_column="conversion_flag" if self.grouped else "",
                label_positive_values=[True] if self.grouped else [],
                filters=["event_type = 'start'"],
                min_interactions_per_user=3,
                snapshot_id="snap-2026-08-01",
                dataset_version="1",
                population_eligibility="users with >=3 interactions in the training window",
                temporal_range=TemporalRange(start="2026-05-01", end="2026-08-01"),
                exclusions=["internal test accounts"],
                label_construction="start within the session window",
            ),
            train=SplitSpec(
                description="first 10 weeks",
                temporal_range=("2026-05-01", "2026-07-10"),
            ),
            validation=SplitSpec(
                description="following 2 weeks",
                temporal_range=("2026-07-11", "2026-07-24"),
            ),
            qa=SplitSpec(
                description="final 1 week holdout",
                temporal_range=("2026-07-25", "2026-08-01"),
            ),
            split_policy=SplitPolicy(
                strategy="temporal",
                train_end="2026-07-10",
                validation_end="2026-07-24",
                qa_end="2026-08-01",
                rationale="Respects the deployment's forward-in-time prediction.",
            ),
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
                    name=self.metric,
                    definition="NDCG over the fixed candidate set.",
                    k=int(self.metric.split("@")[1]),
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
            primary_metric=self.metric,
            evaluation_mode=self.evaluation_mode,  # type: ignore[arg-type]
            relevance_weighting="pbm" if self.grouped else "none",  # type: ignore[arg-type]
            acceptance_criteria=AcceptanceCriteria(
                primary_metric=self.metric,
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
        rationale = "Highest expected value per compute minute first."
        feedback = _extract_section(text, "human_feedback")
        if feedback and feedback != "null":
            rationale = f"{rationale} Revised after human feedback: {feedback}"
        hypotheses = [
            Hypothesis(
                hypothesis_id=f"hyp-{i + 1}",
                title=f"Sequential context improves ranking ({i + 1})",
                proposed_change="Replace the popularity baseline with a SASRec sequence model.",
                why_it_should_improve="Recent watch order carries intent the counts discard.",
                motivating_evidence="Sessions show strong temporal adjacency between titles.",
                experiment_design="Train SASRec on the frozen train split; compare on validation.",
                metric_expected_to_improve=self.metric,
                falsification_criteria=(
                    f"No meaningful {self.metric} gain over the incumbent, "
                    "or a cold-start regression."
                ),
                family=self.family,
                expected_effect_size="+0.02 to +0.05 NDCG@10",
                confidence=0.6 - 0.1 * i,
                priority=i + 1,
            )
            for i in range(self.n_hypotheses)
        ]
        return HypothesisBacklog(hypotheses=hypotheses, prioritization_rationale=rationale)

    def _plan(self, text: str) -> ExperimentPlan:
        self._bump("plan")
        hypothesis_ids = _find_all(r'"hypothesis_id":\s*"([^"]+)"', text)
        return ExperimentPlan(
            hypothesis_id=hypothesis_ids[0] if hypothesis_ids else "hyp-1",
            hypothesis_summary="Sequential modeling beats popularity for top-of-row ranking.",
            rationale="Tests whether order-aware context adds signal beyond frequency.",
            expected_effect="+0.02 to +0.05 NDCG@10 on validation.",
            primary_metric=self.metric,
            secondary_metrics=["recall@10", "coverage"],
            permitted_model_changes=["hidden_dim", "num_layers", "heads", "dropout"],
            permitted_feature_changes=["recency", "metadata"],
            search_space={
                "hidden_dim": [64, 128, 256],
                "num_layers": [2, 3],
                "learning_rate": [1e-4, 3e-4],
            },
            success_criteria=f"+0.01 absolute {self.metric} over baseline on validation.",
            falsification_criteria="No improvement after 3 informed iterations.",
            max_iterations=3,
            compute_budget_minutes=60.0,
            family=self.family,
        )

    def _recipe(self, revision: int = 1, extra: list[FeatureTransform] | None = None):
        if self.grouped:
            # Impression grain: the candidates and the label are on the render
            # row, so there is nothing to generate and nothing to sample.
            return FeatureRecipe(
                sources=[RecipeSource(name="impressions", role="interactions", split="train")],
                feature_columns=[
                    FeatureTransform(
                        name="item_popularity", kind="count", of="item_id", over="train"
                    ),
                    FeatureTransform(name="item_rank", kind="column", source="item_train_rank"),
                    *(extra or []),
                ],
                label=LabelSpec(
                    strategy="label_column",
                    source_column="conversion_flag",
                    positive_values=[True],
                    definition="playback started within 90s of the click",
                ),
                group_key="group_id",
                candidate_generation=CandidateGeneration(strategy="logged_impressions", n=1),
                leakage_notes=(
                    "The logged slot is never an inference-time input; only train-window "
                    "item statistics and the candidate's own attributes are."
                ),
            )
        return FeatureRecipe(
            sources=[
                RecipeSource(name="interactions", role="interactions", split="train"),
                RecipeSource(
                    name="item_attrs",
                    role="item_attributes",
                    split="train",
                    columns=["event_type"],
                ),
            ],
            feature_columns=[
                FeatureTransform(name="item_popularity", kind="count", of="item_id", over="train"),
                FeatureTransform(name="history_length", kind="column", source="history_length"),
                FeatureTransform(
                    name="type_affinity", kind="count", of="event_type", over="history"
                ),
            ]
            + (
                [FeatureTransform(name="item_rank", kind="column", source="item_train_rank")]
                if revision > 1
                else []
            )
            + list(extra or []),
            label=LabelSpec(strategy="holdout_next_item", definition="last train start"),
            group_key="user_id",
            candidate_generation=CandidateGeneration(strategy="popularity_topn", n=200),
            leakage_notes="Every feature is computable from history plus the candidate.",
        )

    def _feature_fit(self, text: str) -> FeatureFitProposal:
        revision = self._bump("feature_fit")
        return FeatureFitProposal(
            family=self.family,
            feature_recipe=self._recipe(revision),
            fit_spec=FitSpec(
                num_boost_round=200,
                early_stopping_rounds=20,
                negative_samples_per_positive=4,
                booster_params={"learning_rate": 0.1, "max_depth": 6},
            ),
            rationale="Catalog frequency plus per-user type affinity tests the hypothesis.",
            leakage_argument="No feature reads the labelled event or anything after it.",
        )

    def _drifted_recipe(self) -> FeatureRecipe:
        """One feature wider than the shape the human approved."""
        return self._recipe(
            extra=[FeatureTransform(name="smuggled_bias", kind="literal", value=1.0)]
        )

    # --- Stage 4/6/7 ----------------------------------------------------------------
    def _config(self, text: str) -> ExperimentConfig:
        i = self._bump("config")
        if self.family != "sasrec":
            drifting = i == self.structural_drift_on_iteration
            return ExperimentConfig(
                family=self.family,
                model={"max_depth": 4 + i},
                training={"learning_rate": 0.1 / i},
                feature_recipe=self._drifted_recipe() if drifting else None,
                fit_spec=FitSpec(
                    num_boost_round=900 if drifting else 200,
                    early_stopping_rounds=20,
                    negative_samples_per_positive=4,
                    objective="rank:pairwise" if drifting else "",
                    booster_params={"learning_rate": 0.1 / i, "max_depth": 4 + i},
                ),
                rationale="Retune depth and learning rate inside the approved shape.",
                changes_from_previous=[] if i == 1 else ["deeper trees", "lower learning rate"],
            )
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
    proxy_correlation: float | None = None,
    converge_after: int = 3,
    n_hypotheses: int = 1,
    family: str = "sasrec",
    structural_drift_on_iteration: int = 0,
    evaluation_mode: str = "next_item",
) -> tuple[LocalMockExecutor, Callable[[str], FakeLLM], CannedResponder]:
    """``(executor, llm_factory, responder)`` configured consistently for a dry run."""
    responder = CannedResponder(
        qa_outcome=qa_outcome,
        converge_after=converge_after,
        n_hypotheses=n_hypotheses,
        family=family,
        structural_drift_on_iteration=structural_drift_on_iteration,
        evaluation_mode=evaluation_mode,
    )
    executor = LocalMockExecutor(
        qa_outcome=qa_outcome,
        phase0_invalid_first_attempt=phase0_invalid_first_attempt,
        proxy_correlation=proxy_correlation,
        converge_after=converge_after,
    )
    return executor, fake_llm_factory(responder), responder
