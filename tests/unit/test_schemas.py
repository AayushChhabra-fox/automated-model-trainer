"""Schema invariants that the whole system leans on."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from amt.schemas.common import ExperimentBudget, MetricDef, SplitRef
from amt.schemas.evaluation_contract import EvaluationContract, SplitPolicy
from amt.schemas.hypothesis import Hypothesis
from amt.schemas.qa import PairedDelta, Phase0Result, QADecision
from amt.schemas.run_result import TrainingRunResult


def test_problem_spec_has_every_required_section(problem_spec):
    assert problem_spec.problem_description.decision
    assert problem_spec.problem_description.prediction_target
    assert problem_spec.problem_description.serving_context
    assert problem_spec.data_sources and problem_spec.data_sources[0].grain
    assert problem_spec.data_sources[0].label_construction
    assert {m.measurability for m in problem_spec.product_metrics} <= {
        "offline",
        "offline_proxy",
        "simulated",
        "online_only",
    }
    assert problem_spec.primary_offline_metric in {m.name for m in problem_spec.offline_metrics}
    assert problem_spec.objective.relation_to_offline_metric
    assert problem_spec.baseline.evaluated_with_same_contract is True


def test_contract_carries_immutable_split_references(evaluation_contract):
    for name in ("train", "validation", "qa"):
        split = evaluation_contract.split(name)
        assert isinstance(split, SplitRef)
        # The table, version and membership hash are placeholders until
        # materialization returns the real references.
        assert split.delta_version >= 0
        assert split.row_count >= 0
        assert split.membership_hash is not None
    assert evaluation_contract.harness_version
    assert evaluation_contract.cohorts
    assert evaluation_contract.metric(evaluation_contract.primary_metric) is not None


def test_contract_rejects_primary_metric_outside_metric_definitions(evaluation_contract):
    payload = evaluation_contract.model_dump()
    payload["primary_metric"] = "not_defined"
    payload["acceptance_criteria"]["primary_metric"] = "not_defined"
    with pytest.raises(ValidationError):
        EvaluationContract.model_validate(payload)


def test_contract_rejects_acceptance_criteria_on_a_different_metric(evaluation_contract):
    payload = evaluation_contract.model_dump()
    payload["acceptance_criteria"]["primary_metric"] = "recall@10"
    with pytest.raises(ValidationError):
        EvaluationContract.model_validate(payload)


def test_contract_rejects_metrics_the_harness_cannot_compute(evaluation_contract):
    # A campaign whose primary metric nothing computes cannot select a champion,
    # and the failure would only surface after the first training job.
    payload = evaluation_contract.model_dump()
    payload["primary_metric"] = "IPS-NDCG@10"
    payload["acceptance_criteria"]["primary_metric"] = "IPS-NDCG@10"
    payload["metrics"].append({"name": "IPS-NDCG@10", "definition": "Position-corrected NDCG."})
    with pytest.raises(ValidationError, match="cannot compute"):
        EvaluationContract.model_validate(payload)


def test_temporal_split_policy_without_boundaries_is_rejected():
    # Boundaries are what materialization executes; prose about "the last 35
    # days" leaves the job with nothing to cut on.
    with pytest.raises(ValidationError, match="train_end"):
        SplitPolicy(strategy="temporal")
    assert SplitPolicy(strategy="leave_last_n", n_validation=2, n_qa=1).n_validation == 2


def test_split_boundaries_must_be_instants_not_sql():
    # The job quotes the boundary into SQL, so `TIMESTAMP '...'` would be
    # nested inside its own quotes and compare as a string.
    with pytest.raises(ValidationError, match="not a timestamp"):
        SplitPolicy(
            strategy="temporal",
            train_end="TIMESTAMP '2026-09-07 00:00:00 UTC'",
            validation_end="2026-09-10",
        )
    # Every form Spark casts to an instant is allowed, including a named zone.
    for train_end in (
        "2026-09-07",
        "2026-09-07 00:00:00",
        "2026-09-07 00:00:00 UTC",
        "2026-09-07T00:00:00+00:00",
        "1757203200",
    ):
        assert (
            SplitPolicy(
                strategy="temporal", train_end=train_end, validation_end="2026-09-10"
            ).train_end
            == train_end
        )


def test_qa_split_use_before_freeze_is_structurally_disallowed(evaluation_contract):
    assert evaluation_contract.split_policy.qa_use_allowed_before_freeze is False
    with pytest.raises(ValidationError):
        evaluation_contract.split_policy.qa_use_allowed_before_freeze = True


def test_hypothesis_answers_all_six_questions():
    hypothesis = Hypothesis(
        title="t",
        proposed_change="c",
        why_it_should_improve="w",
        motivating_evidence="e",
        experiment_design="d",
        metric_expected_to_improve="ndcg@10",
        falsification_criteria="f",
    )
    for field in (
        "proposed_change",
        "why_it_should_improve",
        "motivating_evidence",
        "experiment_design",
        "metric_expected_to_improve",
        "falsification_criteria",
    ):
        assert getattr(hypothesis, field)


def test_frozen_champion_is_immutable(frozen_champion):
    with pytest.raises(ValidationError):
        frozen_champion.config_hash = "tampered"
    with pytest.raises(ValidationError):
        frozen_champion.validation_metrics = {}


def test_qa_decision_values():
    assert {d.value for d in QADecision} == {
        "PASS",
        "FAIL_HYPOTHESIS",
        "SPEC_REVIEW",
        "INVALID_EXPERIMENT",
    }


def test_phase0_is_the_only_source_of_invalid_experiment():
    result = Phase0Result(decision=QADecision.INVALID_EXPERIMENT, fixable=True)
    assert result.decision is QADecision.INVALID_EXPERIMENT
    assert result.passed is False


def test_paired_delta_significance_follows_the_confidence_interval():
    assert PairedDelta(
        metric="ndcg@10", baseline_value=0.3, candidate_value=0.32, mean_delta=0.02,
        ci_low=0.01, ci_high=0.03,
    ).significant
    assert not PairedDelta(
        metric="ndcg@10", baseline_value=0.3, candidate_value=0.30, mean_delta=0.001,
        ci_low=-0.004, ci_high=0.006,
    ).significant


def test_training_run_result_metric_lookup_accepts_prefixed_keys():
    result = TrainingRunResult(
        mlflow_run_id="m", validation_metrics={"val/ndcg@10": 0.34}
    )
    assert result.primary("ndcg@10") == 0.34
    assert result.primary("missing") is None
    assert result.succeeded


def test_budget_accounting():
    budget = ExperimentBudget(max_iterations=2, max_hypotheses=1, max_compute_minutes=10)
    assert budget.iterations_remaining == 2
    budget = budget.record_iteration(4.0)
    assert budget.iterations_used == 1 and budget.compute_minutes_used == 4.0
    budget = budget.record_iteration(7.0)
    assert budget.exhausted is True
    started = budget.record_hypothesis()
    assert started.hypotheses_used == 1 and started.iterations_used == 0


def test_metric_def_defaults_to_maximize():
    assert MetricDef(name="ndcg@10", definition="d").direction == "maximize"
