"""CampaignState — deliberately small (PLAN §3).

Detailed training information lives in MLflow. State holds only what is needed to
resume: stage, current hypothesis, budget, run ids, current champion and contract refs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, TypedDict

from amt.schemas.champion import FrozenChampion, HypothesisChampion
from amt.schemas.common import ExperimentBudget, new_id, utcnow
from amt.schemas.conclusion import ExperimentationReport, ScientificConclusion
from amt.schemas.evaluation_contract import EvaluationContract, SplitMaterialization
from amt.schemas.experiment_config import ExperimentConfig
from amt.schemas.feature_recipe import (
    ApprovedShape,
    FeatureFitProposal,
    FeatureRecipe,
    FitSpec,
)
from amt.schemas.hypothesis import ExperimentPlan, Hypothesis
from amt.schemas.problem_spec import ProblemSpec
from amt.schemas.qa import Phase0Result, Phase1Result, QAAssessment, QADecision
from amt.schemas.run_result import RunSummary

CampaignStatus = Literal["running", "awaiting_human", "done", "aborted", "spec_review"]


class Stage(str, Enum):
    """Mirrors the SPEC.md workflow."""

    PROBLEM_SPEC = "problem_spec"
    EVALUATION_CONTRACT = "evaluation_contract"
    BASELINE = "baseline"
    HYPOTHESES = "hypotheses"
    PLANNING = "planning"
    FEATURE_FIT = "feature_fit"
    TRAINING = "training"
    HYPOTHESIS_CHAMPION = "hypothesis_champion"
    FINAL_CHAMPION = "final_champion"
    FREEZE = "freeze"
    QA_PHASE0 = "qa_phase0"
    QA_PHASE1 = "qa_phase1"
    QA_DECISION = "qa_decision"
    CONCLUSION = "conclusion"
    DONE = "done"


class CampaignState(TypedDict, total=False):
    """LangGraph state for one campaign (thread_id == campaign_id)."""

    campaign_id: str
    created_at: str
    code_version: str
    stage: str
    status: CampaignStatus

    # Stage 1/2 — ground truth
    problem_text: str
    problem_spec: ProblemSpec | None
    problem_spec_approved: bool
    spec_hash: str
    evaluation_contract: EvaluationContract | None
    contract_approved: bool
    contract_hash: str
    split_materialization: SplitMaterialization | None

    # Stage 3-7 — experimentation
    baseline_run_id: str | None
    mlflow_experiment_id: str
    # The backlog is written once and approved once, before any hypothesis is planned.
    hypotheses: list[Hypothesis]
    backlog_rationale: str
    hypotheses_approved: bool
    current_hypothesis_id: str | None
    experiment_plan: ExperimentPlan | None
    current_config: ExperimentConfig | None
    # Feature/fit shapes for the tabular families: the Planner proposes one per
    # tabular hypothesis and a human approves the whole set at HC_FEATURE, once,
    # before any hypothesis is planned. Keyed by hypothesis id.
    feature_fit_proposals: dict[str, FeatureFitProposal]
    shapes_approved: bool
    approved_shapes: dict[str, ApprovedShape]
    # The selected hypothesis's shape, copied out of `approved_shapes` so the
    # Trainer, `assert_feature_fit_approved` and the executor read one shape.
    approved_feature_recipe: FeatureRecipe | None
    approved_fit_spec: FitSpec | None
    feature_fit_hash: str
    feature_fit_approved: bool
    # Structural changes the Trainer asked for and did not get, for `status` to
    # report: a clamped ask is invisible otherwise.
    structural_asks_clamped: list[str]
    # Results for the *current* hypothesis only; cleared when its champion is selected.
    run_results: dict[str, Any]
    last_result: Any | None
    last_interpretation: Any | None
    final_selection_rationale: str
    experiment_history: list[RunSummary]
    iteration: int
    trainer_converged: bool
    convergence_reason: str
    budget: ExperimentBudget
    hypothesis_champions: list[HypothesisChampion]
    baseline_metrics: dict[str, float]
    # HC_CHAMPION: the operator read this hypothesis's champion, and whether
    # they asked to stop exploring and take it to QA.
    champion_reviewed: bool
    freeze_requested: bool

    # Stage 8-10 — freeze and QA
    frozen_champion: FrozenChampion | None
    qa_phase0: Phase0Result | None
    phase0_attempts: int
    # Stamped by `qa_phase0` from settings so routing never reads settings itself.
    phase0_max_attempts: int
    qa_phase1: Phase1Result | None
    qa_assessment: QAAssessment | None
    qa_decision: QADecision | None

    # Stage 11/12
    conclusion: ScientificConclusion | None
    # The decision-ready evidence behind the conclusion's narrative.
    experimentation_report: ExperimentationReport | None
    report_path: str
    mlflow_final_run_id: str | None
    promotion_decision: str | None

    # Human interaction
    human_feedback: str | None
    awaiting: str | None
    # Every HC_* resume with approved=False. The previous campaign burned eight
    # of these and `status` never moved, because a rejection costs no compute.
    gate_cycles: int
    errors: list[str]


def new_campaign_state(
    problem_text: str,
    campaign_id: str | None = None,
    budget: ExperimentBudget | None = None,
    code_version: str = "",
) -> CampaignState:
    """Initial state for a fresh campaign (HC1: the human's problem statement)."""
    return CampaignState(
        campaign_id=campaign_id or new_id("campaign"),
        created_at=utcnow().isoformat(),
        code_version=code_version,
        stage=Stage.PROBLEM_SPEC.value,
        status="running",
        problem_text=problem_text,
        problem_spec=None,
        problem_spec_approved=False,
        spec_hash="",
        evaluation_contract=None,
        contract_approved=False,
        contract_hash="",
        split_materialization=None,
        baseline_run_id=None,
        mlflow_experiment_id="",
        hypotheses=[],
        backlog_rationale="",
        hypotheses_approved=False,
        current_hypothesis_id=None,
        experiment_plan=None,
        current_config=None,
        feature_fit_proposals={},
        shapes_approved=False,
        approved_shapes={},
        approved_feature_recipe=None,
        approved_fit_spec=None,
        feature_fit_hash="",
        feature_fit_approved=False,
        structural_asks_clamped=[],
        run_results={},
        last_result=None,
        last_interpretation=None,
        final_selection_rationale="",
        experiment_history=[],
        iteration=0,
        trainer_converged=False,
        convergence_reason="",
        budget=budget or ExperimentBudget(),
        hypothesis_champions=[],
        baseline_metrics={},
        champion_reviewed=False,
        freeze_requested=False,
        frozen_champion=None,
        qa_phase0=None,
        phase0_attempts=0,
        phase0_max_attempts=0,
        qa_phase1=None,
        qa_assessment=None,
        qa_decision=None,
        conclusion=None,
        experimentation_report=None,
        report_path="",
        mlflow_final_run_id=None,
        promotion_decision=None,
        human_feedback=None,
        awaiting=None,
        gate_cycles=0,
        errors=[],
    )


def current_hypothesis(state: CampaignState) -> Hypothesis | None:
    hid = state.get("current_hypothesis_id")
    if not hid:
        return None
    return next((h for h in state.get("hypotheses") or [] if h.hypothesis_id == hid), None)


def describe(state: CampaignState) -> dict[str, Any]:
    """Compact, printable status view used by ``amt status``."""
    plan = state.get("experiment_plan")
    budget = state.get("budget") or ExperimentBudget()
    feature_fit = (state.get("feature_fit_hash") or "")[:12] or "n/a"
    if state.get("feature_fit_approved"):
        feature_fit = f"{feature_fit} (approved)"
    hypotheses = state.get("hypotheses") or []
    by_status: dict[str, int] = {}
    for hypothesis in hypotheses:
        by_status[hypothesis.status] = by_status.get(hypothesis.status, 0) + 1
    backlog = f"{len(hypotheses)}"
    if by_status:
        backlog += " (" + ", ".join(f"{n} {s}" for s, n in sorted(by_status.items())) + ")"
    if state.get("hypotheses_approved"):
        backlog += " approved"

    cycles = int(state.get("gate_cycles", 0) or 0)
    clamped = list(state.get("structural_asks_clamped") or [])
    described = {
        "campaign_id": state.get("campaign_id"),
        "stage": state.get("stage"),
        "status": state.get("status"),
        "awaiting": state.get("awaiting"),
        "contract_hash": (state.get("contract_hash") or "")[:12],
        "hypotheses": backlog,
        "current_hypothesis": state.get("current_hypothesis_id"),
        "feature_fit": feature_fit,
        "iteration": state.get("iteration", 0),
        "max_iterations": plan.max_iterations if plan else budget.max_iterations,
        "experiments": len(state.get("experiment_history") or []),
        "champions": len(state.get("hypothesis_champions") or []),
        "frozen": state.get("frozen_champion") is not None,
        "qa_decision": state.get("qa_decision").value if state.get("qa_decision") else None,
        # Rejections cost no compute, so without this line a campaign that has
        # spent a day arguing with its own planner looks the same as a fresh one.
        "gate_cycles": f"{cycles} (high)" if cycles > 2 else str(cycles),
    }
    if clamped:
        described["trainer_asks_refused"] = (
            f"{len(clamped)} — trainer requested a structural change; kept the approved shape"
        )
    return described
