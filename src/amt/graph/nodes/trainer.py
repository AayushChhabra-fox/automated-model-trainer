"""Stage 4/5/6 — propose a configuration, execute it on Databricks, interpret the result."""

from __future__ import annotations

from collections.abc import Callable

from amt.graph.deps import NodeDeps
from amt.graph.routing import (
    ContractViolation,
    assert_contract_hash,
    assert_feature_fit_approved,
    assert_not_frozen,
    trainer_should_continue,
)
from amt.graph.state import CampaignState, Stage, current_hypothesis
from amt.schemas.common import ExperimentBudget
from amt.schemas.experiment_config import ExperimentConfig
from amt.schemas.feature_recipe import FitSpec, feature_fit_hash, is_tabular_family
from amt.schemas.run_result import RunSummary


def _bind_feature_fit(state: CampaignState, config: ExperimentConfig) -> ExperimentConfig:
    """Pin a config to the approved feature/fit shape.

    Non-tabular families carry no recipe at all. For a tabular family the approved
    recipe and the structural part of the approved FitSpec always win; the Trainer's
    contribution is ``booster_params``. When the Trainer asks for a structurally
    different shape the config keeps that ask, its hash stops matching the approved
    one, and routing sends the campaign back to the feature/fit checkpoint.
    """
    if not is_tabular_family(config.family):
        return config.model_copy(
            update={"feature_recipe": None, "fit_spec": None, "feature_fit_hash": ""}
        )

    recipe = state.get("approved_feature_recipe")
    approved_fit = state.get("approved_fit_spec") or FitSpec()
    if recipe is None or not state.get("feature_fit_approved"):
        # Nothing approved yet: record what was asked for and let routing gate it.
        return config.model_copy(
            update={"feature_fit_hash": feature_fit_hash(config.feature_recipe, config.fit_spec)}
        )

    proposed = config.fit_spec
    requested_recipe = config.feature_recipe
    structural_change = (
        requested_recipe is not None
        and requested_recipe.structural_payload != recipe.structural_payload
    ) or (
        proposed is not None
        and proposed.structural_payload != approved_fit.structural_payload
    )
    if structural_change:
        return config.model_copy(
            update={
                "feature_recipe": requested_recipe or recipe,
                "fit_spec": proposed or approved_fit,
                "feature_fit_hash": feature_fit_hash(
                    requested_recipe or recipe, proposed or approved_fit
                ),
            }
        )

    fit = approved_fit
    if proposed is not None:
        fit = approved_fit.model_copy(
            update={"booster_params": dict(proposed.booster_params)}
        )
    return config.model_copy(
        update={
            "feature_recipe": recipe,
            "fit_spec": fit,
            "feature_fit_hash": feature_fit_hash(recipe, fit),
        }
    )


def make_trainer_propose_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    def trainer_propose_node(state: CampaignState) -> dict:
        assert_not_frozen(state)
        contract = state.get("evaluation_contract")
        assert_contract_hash(state, contract)
        plan = state.get("experiment_plan")
        hypothesis = current_hypothesis(state)
        if plan is None or hypothesis is None:
            raise ContractViolation("Trainer has no ExperimentPlan / hypothesis to work on.")
        config = deps.trainer_agent.propose_config(
            hypothesis=hypothesis,
            plan=plan,
            spec=state.get("problem_spec"),
            contract=contract,
            history=list(state.get("experiment_history") or []),
            last_result=state.get("last_result"),
            last_interpretation=state.get("last_interpretation"),
        )
        config = _bind_feature_fit(state, config)
        update: dict = {
            "stage": Stage.TRAINING.value,
            "current_config": config,
        }
        if is_tabular_family(config.family) and config.feature_fit_hash != (
            state.get("feature_fit_hash") or ""
        ):
            # The approved shape no longer covers this config.
            update["feature_fit_approved"] = False
        return update

    return trainer_propose_node


def make_execute_training_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Side-effecting node: submits the training job and records a compact summary."""

    def execute_training_node(state: CampaignState) -> dict:
        assert_not_frozen(state)
        contract = state.get("evaluation_contract")
        assert_contract_hash(state, contract)
        plan = state.get("experiment_plan")
        config = state.get("current_config")
        if config is None or plan is None:
            raise ContractViolation("No ExperimentConfig to execute.")
        if config.contract_hash and config.contract_hash != state.get("contract_hash"):
            raise ContractViolation("ExperimentConfig references a stale EvaluationContract.")
        assert_feature_fit_approved(state)

        iteration = int(state.get("iteration", 0) or 0) + 1
        result = deps.executor.run_training(config, contract, plan)
        if result.contract_hash and result.contract_hash != state.get("contract_hash"):
            raise ContractViolation("Training job ran against a different EvaluationContract.")

        budget: ExperimentBudget = state.get("budget") or ExperimentBudget()
        summary = RunSummary(
            run_id=result.run_id,
            mlflow_run_id=result.mlflow_run_id,
            hypothesis_id=result.hypothesis_id or (state.get("current_hypothesis_id") or ""),
            config_hash=result.config_hash,
            iteration=iteration,
            primary_metric=contract.primary_metric,
            primary_metric_value=result.primary(contract.primary_metric),
            verdict="failed" if not result.succeeded else "no_change",
            runtime_seconds=result.runtime_seconds,
        )
        results = dict(state.get("run_results") or {})
        results[result.run_id] = result
        return {
            "iteration": iteration,
            "last_result": result,
            "run_results": results,
            "experiment_history": [*(state.get("experiment_history") or []), summary],
            "budget": budget.record_iteration(result.runtime_seconds / 60.0),
        }

    return execute_training_node


def make_trainer_interpret_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Observation → Interpretation → Proposed Change → Expected Effect, plus convergence."""

    def trainer_interpret_node(state: CampaignState) -> dict:
        result = state.get("last_result")
        plan = state.get("experiment_plan")
        hypothesis = current_hypothesis(state)
        if result is None or plan is None or hypothesis is None:
            raise ContractViolation("Nothing to interpret.")
        history = list(state.get("experiment_history") or [])
        interpretation = deps.trainer_agent.interpret(
            result=result,
            hypothesis=hypothesis,
            plan=plan,
            history=history,
            iteration=int(state.get("iteration", 0) or 0),
        )
        if history:
            history[-1] = history[-1].model_copy(
                update={
                    "verdict": interpretation.verdict,
                    "notes": interpretation.interpretation[:500],
                }
            )
        update: dict = {
            "experiment_history": history,
            "last_interpretation": interpretation,
            "trainer_converged": interpretation.converged,
            "convergence_reason": interpretation.convergence_reason,
        }
        # Hard stops beat the agent's opinion.
        should_continue, reason = trainer_should_continue({**state, **update})
        if not should_continue:
            update["trainer_converged"] = True
            update["convergence_reason"] = update["convergence_reason"] or reason
        return update

    return trainer_interpret_node
