"""Full-graph campaign: HC2 reject→approve, HC3 approve, trainer loop, freeze, every QA path."""

from __future__ import annotations

import pytest
from langgraph.types import Command

from amt.executor.local_mock import LocalMockExecutor
from amt.graph.state import new_campaign_state
from amt.schemas.champion import FrozenChampion
from amt.schemas.qa import Phase0Result, QADecision
from tests.conftest import graph_state, pending_interrupt

PROBLEM = "Improve homepage row ranking so members start something faster."


def start(graph, config, campaign_id="test-campaign"):
    graph.invoke(new_campaign_state(PROBLEM, campaign_id), config)
    return pending_interrupt(graph, config)


def resume(graph, config, **payload):
    graph.invoke(Command(resume=payload), config)
    return pending_interrupt(graph, config)


def run_to_hc4(graph, config, campaign_id="test-campaign"):
    """HC1 → HC2 (reject, then approve) → HC3 → HC_BACKLOG → autonomous → HC4."""
    hc2 = start(graph, config, campaign_id)
    assert hc2 and hc2["checkpoint"] == "HC2"

    rejected = resume(graph, config, approved=False, feedback="use a 30-day label window")
    assert rejected and rejected["checkpoint"] == "HC2", "rejection must re-run Stage 1"

    hc3 = resume(graph, config, approved=True)
    assert hc3 and hc3["checkpoint"] == "HC3"

    backlog = resume(graph, config, approved=True)
    assert backlog and backlog["checkpoint"] == "HC_BACKLOG"

    hc4 = resume(graph, config, approved=True)
    return hc4


# --- happy path ----------------------------------------------------------------------
def test_campaign_reaches_hc4_and_passes_qa(campaign):
    graph, config, executor, responder = campaign(qa_outcome="PASS")
    hc4 = run_to_hc4(graph, config)

    assert hc4 and hc4["checkpoint"] == "HC4"
    assert hc4["qa_decision"] == "PASS"

    state = graph_state(graph, config)
    assert state["qa_decision"] is QADecision.PASS
    assert state["conclusion"].recommendation == "proceed_to_ab_test"
    assert state["conclusion"].negative_result is False

    # HC4 decision closes the campaign.
    resume(graph, config, approved=True, decision="promote")
    final = graph_state(graph, config)
    assert final["status"] == "done"
    assert final["promotion_decision"] == "promote"


def test_rejection_feedback_reaches_the_redrafted_spec(campaign):
    graph, config, executor, responder = campaign()
    start(graph, config)
    resume(graph, config, approved=False, feedback="use a 30-day label window")
    spec = graph_state(graph, config)["problem_spec"]
    assert spec.revision >= 1
    assert any("30-day label window" in a for a in spec.assumptions)


# --- trainer loop -----------------------------------------------------------------------
def test_trainer_loop_iterates_then_converges(campaign):
    graph, config, executor, responder = campaign(converge_after=3)
    run_to_hc4(graph, config)

    # Plan allows 3 iterations and the agent converges on the 3rd interpretation.
    assert len(executor.training_calls) == 3
    assert len(set(executor.training_calls)) == 3, "each iteration must be a different config"

    state = graph_state(graph, config)
    history = state["experiment_history"]
    assert [h.iteration for h in history] == [1, 2, 3]
    assert all(h.primary_metric == "ndcg@10" for h in history)
    values = [h.primary_metric_value for h in history]
    assert values == sorted(values), "mock metrics improve then plateau"
    assert state["trainer_converged"] is True
    assert state["convergence_reason"]


def test_iteration_cap_is_enforced_even_if_the_agent_never_converges(campaign):
    graph, config, executor, responder = campaign(converge_after=99)
    run_to_hc4(graph, config)
    assert len(executor.training_calls) == 3  # plan.max_iterations
    state = graph_state(graph, config)
    assert "max iterations" in state["convergence_reason"]


# --- freeze + QA independence ---------------------------------------------------------------
def test_exactly_one_frozen_champion_and_qa_never_runs_before_freeze(campaign):
    graph, config, executor, responder = campaign()
    run_to_hc4(graph, config)
    state = graph_state(graph, config)

    frozen = state["frozen_champion"]
    assert isinstance(frozen, FrozenChampion)
    assert len(state["hypothesis_champions"]) == 1
    assert frozen.hypothesis_id == state["hypothesis_champions"][0].hypothesis_id

    calls = executor.calls
    last_training = max(i for i, c in enumerate(calls) if c == "run_training")
    qa_calls = [i for i, c in enumerate(calls) if c.startswith("run_qa")]
    assert qa_calls, "QA must have run"
    assert min(qa_calls) > last_training, "QA ran before the champion was frozen"

    # The QA split is read exactly once — by Phase 1, after freeze.
    assert executor.qa_split_reads == 1
    assert "run_baseline:qa" not in calls


def test_contract_hash_is_constant_for_the_whole_campaign(campaign):
    graph, config, executor, responder = campaign()
    start(graph, config)
    resume(graph, config, approved=False, feedback="tighten the population")
    resume(graph, config, approved=True)

    stamped_at_hc3 = graph_state(graph, config)["evaluation_contract"].contract_hash
    resume(graph, config, approved=True)  # HC_BACKLOG
    resume(graph, config, approved=True)

    state = graph_state(graph, config)
    assert state["contract_hash"] == stamped_at_hc3
    assert state["evaluation_contract"].contract_hash == stamped_at_hc3
    assert state["frozen_champion"].contract_hash == stamped_at_hc3
    assert state["qa_phase0"].contract_hash == stamped_at_hc3
    assert state["qa_phase1"].contract_hash == stamped_at_hc3
    assert state["conclusion"].contract_hash == stamped_at_hc3
    assert executor.expected_contract_hash == stamped_at_hc3


def test_splits_are_materialized_once_with_immutable_references(campaign):
    graph, config, executor, responder = campaign()
    run_to_hc4(graph, config)
    state = graph_state(graph, config)
    materialization = state["split_materialization"]
    contract = state["evaluation_contract"]
    for name in ("train", "validation", "qa"):
        split = contract.split(name)
        assert split.membership_hash, "split membership must be pinned"
        assert split.row_count > 0
        assert materialization.splits()[name].membership_hash == split.membership_hash


# --- QA decisions -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("outcome", "recommendation", "status"),
    [
        ("PASS", "proceed_to_ab_test", "done"),
        ("FAIL_HYPOTHESIS", "hypothesis_rejected", "done"),
        ("SPEC_REVIEW", "evaluation_contract_requires_revision", "spec_review"),
    ],
)
def test_each_qa_decision_routes_to_the_conclusion(campaign, outcome, recommendation, status):
    graph, config, executor, responder = campaign(qa_outcome=outcome)
    hc4 = run_to_hc4(graph, config)
    assert hc4 and hc4["checkpoint"] == "HC4"
    assert hc4["qa_decision"] == outcome

    state = graph_state(graph, config)
    assert state["qa_decision"] is QADecision(outcome)
    assert state["conclusion"].recommendation == recommendation
    assert state["conclusion"].qa_decision is QADecision(outcome)
    assert (state["conclusion"].negative_result) is (outcome != "PASS")
    if outcome == "SPEC_REVIEW":
        assert state["status"] == "spec_review"
        assert state["qa_assessment"].spec_review_reasons

    resume(graph, config, approved=outcome == "PASS")
    assert graph_state(graph, config)["status"] == status
    # No retraining happened after QA under any outcome.
    assert executor.calls[-1] == "run_qa_phase1"


def test_fixable_phase0_defect_is_repaired_and_qa_continues(campaign):
    graph, config, executor, responder = campaign(
        qa_outcome="PASS", phase0_invalid_first_attempt=True
    )
    hc4 = run_to_hc4(graph, config)
    assert hc4 and hc4["qa_decision"] == "PASS"

    state = graph_state(graph, config)
    assert executor.phase0_attempts == 2
    assert state["phase0_attempts"] == 2
    assert state["qa_phase0"].passed is True
    assert state["qa_phase0"].attempt == 2
    assert state["qa_phase1"] is not None
    # Repairing a Phase-0 defect must not retrain or reselect.
    assert len(executor.training_calls) == 3
    assert state["frozen_champion"].config_hash in executor.training_calls


class AlwaysBrokenExecutor(LocalMockExecutor):
    """Phase 0 never passes: exercises the retry cap."""

    def run_qa_phase0(self, frozen, contract) -> Phase0Result:
        result = super().run_qa_phase0(frozen, contract)
        return result.model_copy(
            update={
                "passed": False,
                "fixable": True,
                "decision": QADecision.INVALID_EXPERIMENT,
                "defects": ["artifact cannot be deserialized"],
            }
        )


def test_phase0_retry_cap_aborts_with_invalid_experiment(campaign):
    graph, config, executor, responder = campaign(executor=AlwaysBrokenExecutor())
    hc4 = run_to_hc4(graph, config)

    assert executor.phase0_attempts == 3
    assert hc4 and hc4["checkpoint"] == "HC4"
    state = graph_state(graph, config)
    assert state["qa_decision"] is QADecision.INVALID_EXPERIMENT
    assert state["qa_phase1"] is None, "Phase 1 must never run after a failed Phase 0"
    assert executor.qa_split_reads == 0
    assert "run_qa_phase1" not in executor.calls


# --- QA independence invariants ---------------------------------------------------------------
def test_qa_agent_never_sees_training_configuration_choices(campaign):
    graph, config, executor, responder = campaign()
    run_to_hc4(graph, config)
    # The QA agent is only ever asked for a triage or an assessment — never a config.
    assert "ExperimentConfig" not in responder.calls[responder.calls.index("QAAssessment") :]
    assert responder.calls.index("QAAssessment") > responder.calls.index("ChampionSelection")


# --- backlog checkpoint ------------------------------------------------------------------
def test_the_whole_backlog_is_presented_before_any_hypothesis_is_planned(campaign):
    graph, config, executor, responder = campaign(n_hypotheses=2)
    start(graph, config)
    resume(graph, config, approved=True)  # HC2
    backlog = resume(graph, config, approved=True)  # HC3

    assert backlog and backlog["checkpoint"] == "HC_BACKLOG"
    assert len(backlog["hypotheses"]) == 2
    assert all(h["status"] == "pending" for h in backlog["hypotheses"])
    # The baseline is known by then, so the human judges the backlog against it.
    assert backlog["baseline_run_id"]
    assert "run_baseline:validation" in executor.calls
    assert "run_training" not in executor.calls
    assert graph_state(graph, config)["experiment_plan"] is None


def test_rejecting_the_backlog_revises_it_and_still_trains_nothing(campaign):
    graph, config, executor, responder = campaign()
    start(graph, config)
    resume(graph, config, approved=True)
    resume(graph, config, approved=True)

    again = resume(graph, config, approved=False, feedback="drop anything needing item metadata")
    assert again and again["checkpoint"] == "HC_BACKLOG"
    assert responder.counts["backlog"] == 2
    assert "item metadata" in again["prioritization_rationale"], "feedback must reach the revision"
    assert "run_training" not in executor.calls
    assert graph_state(graph, config)["hypotheses_approved"] is False

    resume(graph, config, approved=True)
    state = graph_state(graph, config)
    assert state["hypotheses_approved"] is True
    assert executor.training_calls


def test_the_backlog_is_approved_once_not_once_per_hypothesis(campaign):
    graph, config, executor, responder = campaign(n_hypotheses=2)
    hc4 = run_to_hc4(graph, config)
    assert hc4 and hc4["checkpoint"] == "HC4"
    assert responder.counts["backlog"] == 1
    assert len(graph_state(graph, config)["hypothesis_champions"]) == 2


def test_multiple_hypotheses_produce_one_final_champion(campaign):
    graph, config, executor, responder = campaign(n_hypotheses=2)
    run_to_hc4(graph, config)
    state = graph_state(graph, config)
    assert len(state["hypothesis_champions"]) == 2
    assert isinstance(state["frozen_champion"], FrozenChampion)
    assert state["frozen_champion"].hypothesis_id in {
        c.hypothesis_id for c in state["hypothesis_champions"]
    }


# --- feature/fit checkpoint (booster families) ------------------------------------------------
def run_tabular_to_hc4(graph, config, campaign_id="test-campaign", reject_first_shape=False):
    """Same path as :func:`run_to_hc4`, plus the HC_FEATURE gate before any training."""
    hc2 = start(graph, config, campaign_id)
    assert hc2 and hc2["checkpoint"] == "HC2"
    hc3 = resume(graph, config, approved=True)
    assert hc3 and hc3["checkpoint"] == "HC3"
    backlog = resume(graph, config, approved=True)
    assert backlog and backlog["checkpoint"] == "HC_BACKLOG"

    hc_feature = resume(graph, config, approved=True)
    assert hc_feature and hc_feature["checkpoint"] == "HC_FEATURE"
    if reject_first_shape:
        hc_feature = resume(graph, config, approved=False, feedback="drop the leaky feature")
        assert hc_feature and hc_feature["checkpoint"] == "HC_FEATURE"
    return resume(graph, config, approved=True), hc_feature


def test_a_booster_campaign_must_pass_the_feature_checkpoint_before_training(campaign):
    graph, config, executor, responder = campaign(family="lightgbm_reranker")
    hc4, hc_feature = run_tabular_to_hc4(graph, config)

    payload = hc_feature["feature_fit"]
    assert payload["family"] == "lightgbm_reranker"
    assert payload["feature_recipe"]["feature_columns"]
    assert hc_feature["feature_fit_hash"]

    assert hc4 and hc4["checkpoint"] == "HC4"
    state = graph_state(graph, config)
    assert state["feature_fit_approved"] is True
    assert state["feature_fit_hash"] == hc_feature["feature_fit_hash"]
    # Every run went out under the approved shape, and the recipe reached the executor.
    assert len(executor.training_calls) == 3
    champion_config = state["frozen_champion"].config
    assert champion_config.family == "lightgbm_reranker"
    assert champion_config.feature_recipe is not None
    assert champion_config.feature_fit_hash == hc_feature["feature_fit_hash"]


def test_rejecting_a_shape_redrafts_it_and_blocks_training(campaign):
    graph, config, executor, responder = campaign(family="xgboost_classifier")
    hc4, _shape = run_tabular_to_hc4(graph, config, reject_first_shape=True)

    assert hc4 and hc4["checkpoint"] == "HC4"
    # Two proposals, and nothing was trained until one was approved.
    assert responder.counts["feature_fit"] == 2
    first_training = executor.calls.index("run_training")
    assert executor.calls[:first_training].count("run_training") == 0
    state = graph_state(graph, config)
    assert state["feature_fit_approved"] is True


def test_hyperparameter_only_iterations_do_not_reopen_the_checkpoint(campaign):
    graph, config, executor, responder = campaign(family="xgboost_reranker")
    hc4, _shape = run_tabular_to_hc4(graph, config)
    assert hc4 and hc4["checkpoint"] == "HC4"

    # One approval covered all three iterations; the recipe never changed.
    assert responder.counts["feature_fit"] == 1
    assert len(executor.training_calls) == 3
    assert len(set(executor.training_calls)) == 3, "each iteration must be a different config"


def test_the_fixed_shape_families_never_reach_the_feature_checkpoint(campaign):
    graph, config, executor, responder = campaign(family="sasrec")
    hc4 = run_to_hc4(graph, config)
    assert hc4 and hc4["checkpoint"] == "HC4"
    assert "feature_fit" not in responder.counts
    assert graph_state(graph, config)["feature_fit_hash"] == ""
