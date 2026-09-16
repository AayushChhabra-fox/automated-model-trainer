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


def _structural_asks(approved: FitSpec, proposed: FitSpec | None) -> list[str]:
    """FitSpec fields where the Trainer asked for something it will not get.

    A blank is not an ask. The Trainer routinely omits ``objective`` and
    ``eval_metric``, and those inherit the approved values rather than being
    reported as drift — which is the whole of B2.
    """
    if proposed is None:
        return []
    approved_payload = approved.structural_payload
    return [
        f"fit_spec.{field}={value!r} (kept {approved_payload.get(field)!r})"
        for field, value in proposed.structural_payload.items()
        if value not in ("", None) and value != approved_payload.get(field)
    ]


def _bind_feature_fit(
    state: CampaignState, config: ExperimentConfig
) -> tuple[ExperimentConfig, list[str]]:
    """Pin a config to the approved feature/fit shape, and say what was refused.

    Non-tabular families carry no recipe at all. For a tabular family the shape
    a human approved *always* wins and the Trainer's only contribution is
    ``booster_params``: a structural ask is recorded and discarded, not routed
    back to the checkpoint. The bounce loop that used to live here cost eight
    review rounds and no compute, and the approved shape won every time anyway.

    Returns the bound config and the asks that were clamped.
    """
    if not is_tabular_family(config.family):
        return (
            config.model_copy(
                update={"feature_recipe": None, "fit_spec": None, "feature_fit_hash": ""}
            ),
            [],
        )

    recipe = state.get("approved_feature_recipe")
    if recipe is None or not state.get("feature_fit_approved"):
        raise ContractViolation(
            f"family {config.family!r} needs an approved feature/fit shape and "
            "state carries none; HC_FEATURE runs before selection, so the "
            "Trainer can never legitimately be the first to ask for one."
        )
    approved_fit = state.get("approved_fit_spec") or FitSpec()

    clamped = _structural_asks(approved_fit, config.fit_spec)
    if (
        config.feature_recipe is not None
        and config.feature_recipe.structural_payload != recipe.structural_payload
    ):
        clamped.insert(0, "feature_recipe (kept the approved recipe)")

    fit = approved_fit
    if config.fit_spec is not None:
        fit = approved_fit.model_copy(
            update={"booster_params": dict(config.fit_spec.booster_params)}
        )
    return (
        config.model_copy(
            update={
                "feature_recipe": recipe,
                "fit_spec": fit,
                "feature_fit_hash": feature_fit_hash(recipe, fit),
            }
        ),
        clamped,
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
        config, clamped = _bind_feature_fit(state, config)
        update: dict = {
            "stage": Stage.TRAINING.value,
            "current_config": config,
        }
        if clamped:
            iteration = int(state.get("iteration", 0) or 0) + 1
            update["structural_asks_clamped"] = [
                *(state.get("structural_asks_clamped") or []),
                *(f"{hypothesis.hypothesis_id} it{iteration}: {ask}" for ask in clamped),
            ]
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
            feature_fit_hash=config.feature_fit_hash,
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


def _repeated_configs(history: list[RunSummary]) -> list[str]:
    """Iterations whose configuration is bit-identical to an earlier one.

    A retune that changes nothing is not an iteration's worth of evidence, and
    the Trainer has no other way to notice: it reads metrics, and identical
    configurations produce identical metrics that look like a plateau.
    """
    seen: dict[str, int] = {}
    repeats = []
    for summary in history:
        if not summary.config_hash:
            continue
        first = seen.setdefault(summary.config_hash, summary.iteration)
        if first != summary.iteration:
            repeats.append(
                f"iteration {summary.iteration} repeats iteration {first} exactly "
                f"(config_hash {summary.config_hash[:12]})"
            )
    return repeats


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
            repeats=_repeated_configs(history),
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
