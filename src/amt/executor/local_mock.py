"""Deterministic in-process executor for tests and ``--mock`` dry runs.

No network, no Databricks, no MLflow. Metrics improve and then plateau so the trainer
loop converges naturally, and QA evidence can be shaped to exercise every decision path.
"""

from __future__ import annotations

import hashlib
import math
from typing import Literal

from amt.schemas.champion import FrozenChampion
from amt.schemas.common import SplitRef
from amt.schemas.evaluation_contract import EvaluationContract, SplitMaterialization
from amt.schemas.experiment_config import ExperimentConfig
from amt.schemas.hypothesis import ExperimentPlan
from amt.schemas.problem_spec import ProblemSpec
from amt.schemas.qa import (
    IntegrityCheck,
    PairedDelta,
    Phase0Result,
    Phase1Result,
    QADecision,
)
from amt.schemas.run_result import TrainingRunResult

QaOutcome = Literal["PASS", "FAIL_HYPOTHESIS", "SPEC_REVIEW"]

BASELINE_VALUE = 0.3000
MAX_GAIN = 0.0600


def _jitter(seed: str, scale: float = 1e-4) -> float:
    """Deterministic, tiny, sign-stable perturbation derived from a string."""
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0 * scale


class ContractMismatchError(RuntimeError):
    """Raised when a job is handed a contract that is not the frozen one."""


class LocalMockExecutor:
    """Implements :class:`amt.executor.base.Executor` deterministically.

    Parameters
    ----------
    qa_outcome:
        Shape of the Phase-1 evidence: ``PASS`` (significant improvement),
        ``FAIL_HYPOTHESIS`` (no significant holdout improvement) or ``SPEC_REVIEW``
        (evidence of a fundamental problem: leakage-like validation/QA divergence).
    phase0_invalid_first_attempt:
        When True the first Phase-0 attempt reports a *fixable* defect
        (``INVALID_EXPERIMENT``); the next attempt passes.
    """

    def __init__(
        self,
        qa_outcome: QaOutcome = "PASS",
        phase0_invalid_first_attempt: bool = False,
        converge_after: int = 3,
        expected_contract_hash: str | None = None,
    ) -> None:
        self.qa_outcome: QaOutcome = qa_outcome
        self.phase0_invalid_first_attempt = phase0_invalid_first_attempt
        self.converge_after = converge_after
        self.expected_contract_hash = expected_contract_hash
        self.calls: list[str] = []
        self.training_calls: list[str] = []
        #: Every config that reached compute, so a test can assert what the
        #: clamp let through rather than only what the orchestrator kept.
        self.training_configs: list[ExperimentConfig] = []
        self.phase0_attempts = 0
        self.qa_split_reads = 0
        self._iteration: dict[str, int] = {}

    # --- guards -----------------------------------------------------------------
    def _check(self, contract: EvaluationContract) -> None:
        if self.expected_contract_hash is None:
            self.expected_contract_hash = contract.contract_hash
        elif contract.contract_hash != self.expected_contract_hash:
            raise ContractMismatchError(
                "EvaluationContract changed mid-campaign: "
                f"{contract.contract_hash[:12]} != {self.expected_contract_hash[:12]}"
            )

    # --- Executor protocol --------------------------------------------------------
    def materialize_splits(self, contract: EvaluationContract) -> SplitMaterialization:
        # Materialization happens *before* HC3: the split references it returns are folded
        # back into the contract, which changes its hash. The campaign's ground truth is
        # pinned by the first call made under the frozen contract (run_baseline), not here.
        self.calls.append("materialize_splits")

        def ref(base: SplitRef, rows: int) -> SplitRef:
            return base.model_copy(
                update={
                    # The real job names the output table; the draft leaves it empty.
                    "table": base.table or f"main.amt.split_{base.name}",
                    "delta_version": 1,
                    "row_count": base.row_count or rows,
                    "membership_hash": hashlib.sha256(
                        f"{contract.contract_id}:{base.name}".encode()
                    ).hexdigest(),
                }
            )

        return SplitMaterialization(
            contract_id=contract.contract_id,
            contract_hash=contract.contract_hash,
            train=ref(contract.train_split, 80_000),
            validation=ref(contract.validation_split, 10_000),
            qa=ref(contract.qa_split, 10_000),
            # Shaped like the real estimator's output: slot 0 is the reference,
            # and the deeper slots are examined far less often.
            propensities=(
                {"ROKU|0": 1.0, "ROKU|1": 0.12, "ROKU|2": 0.05, "ROKU|3": 0.03}
                if contract.relevance_weighting == "pbm"
                else {}
            ),
            job_run_id="mock-materialize-1",
        )

    def run_baseline(
        self,
        spec: ProblemSpec,
        contract: EvaluationContract,
        split: Literal["validation", "qa"] = "validation",
    ) -> TrainingRunResult:
        self._check(contract)
        self.calls.append(f"run_baseline:{split}")
        if split == "qa":
            self.qa_split_reads += 1
        metric = contract.primary_metric
        value = BASELINE_VALUE
        return TrainingRunResult(
            run_id="run-baseline",
            mlflow_run_id="mlflow-baseline",
            mlflow_experiment_id="mock-exp",
            role="baseline",
            family=spec.baseline.kind,
            config_hash="baseline",
            train_metrics={},
            validation_metrics={f"val/{metric}": value, "val/recall@10": value * 1.4},
            cohort_metrics={
                c.name: {f"val/{metric}": value * (0.9 + 0.05 * i)}
                for i, c in enumerate(contract.cohorts)
            },
            runtime_seconds=12.0,
            model_artifact_uri="mock://baseline/model",
            status="success",
            contract_hash=contract.contract_hash,
            harness_version=contract.harness_version,
        )

    def run_training(
        self,
        config: ExperimentConfig,
        contract: EvaluationContract,
        plan: ExperimentPlan,
    ) -> TrainingRunResult:
        self._check(contract)
        self.calls.append("run_training")
        self.training_calls.append(config.config_hash)
        self.training_configs.append(config)
        key = plan.hypothesis_id or config.hypothesis_id or "default"
        i = self._iteration.get(key, 0)
        self._iteration[key] = i + 1

        metric = contract.primary_metric
        # Improving then plateauing: saturating exponential in the iteration index.
        gain = MAX_GAIN * (1.0 - math.exp(-1.1 * i))
        value = BASELINE_VALUE + gain + _jitter(config.config_hash)
        train_value = value + 0.05 + 0.004 * i  # widening generalization gap
        return TrainingRunResult(
            run_id=f"run-{key}-{i}",
            mlflow_run_id=f"mlflow-{key}-{i}",
            mlflow_experiment_id="mock-exp",
            role="train",
            hypothesis_id=key,
            config=config,
            config_hash=config.config_hash,
            family=config.family,
            train_metrics={f"train/{metric}": train_value, "train/loss": 0.60 / (1 + i)},
            validation_metrics={
                f"val/{metric}": value,
                "val/recall@10": value * 1.4,
                "val/coverage": 0.42 + 0.01 * i,
            },
            cohort_metrics={
                c.name: {f"val/{metric}": value * (0.92 + 0.04 * idx)}
                for idx, c in enumerate(contract.cohorts)
            },
            training_curve=[
                {"epoch": e, "train/loss": 0.6 / (1 + i) * math.exp(-0.2 * e)} for e in range(5)
            ],
            runtime_seconds=90.0 + 5 * i,
            model_artifact_uri=f"mock://models/{config.config_hash[:12]}",
            status="success",
            contract_hash=contract.contract_hash,
            harness_version=contract.harness_version,
        )

    def run_qa_phase0(self, frozen: FrozenChampion, contract: EvaluationContract) -> Phase0Result:
        self._check(contract)
        self.calls.append("run_qa_phase0")
        self.phase0_attempts += 1
        attempt = self.phase0_attempts
        broken = self.phase0_invalid_first_attempt and attempt == 1

        checks = [
            IntegrityCheck(
                name="artifact_loads",
                passed=not broken,
                detail=(
                    "preprocessing artifact missing from the logged model"
                    if broken
                    else "pyfunc load"
                ),
            ),
            IntegrityCheck(name="serialization_valid", passed=True),
            IntegrityCheck(name="model_version_expected", passed=True),
            IntegrityCheck(name="schema_compatible", passed=True),
            IntegrityCheck(
                name="contract_hash_matches",
                passed=frozen.contract_hash == contract.contract_hash,
            ),
            IntegrityCheck(
                name="harness_version_matches",
                passed=frozen.harness_version in ("", contract.harness_version),
            ),
            IntegrityCheck(name="inference_executes", passed=not broken),
        ]
        metric = contract.primary_metric
        logged = frozen.validation_metrics.get(f"val/{metric}", BASELINE_VALUE + MAX_GAIN)
        recomputed = logged if not broken else logged * 0.5
        delta = abs(recomputed - logged)
        within = delta <= 1e-3
        passed = all(c.passed for c in checks) and within
        return Phase0Result(
            attempt=attempt,
            passed=passed,
            checks=checks,
            artifact_loaded=not broken,
            expected_model_version=frozen.config_hash[:12],
            schema_compatible=True,
            contract_hash_matches=frozen.contract_hash == contract.contract_hash,
            harness_version_matches=True,
            inference_executed=not broken,
            reproduction_metrics={f"val/{metric}": recomputed},
            reproduction_deltas={f"val/{metric}": delta},
            reproduction_tolerance=1e-3,
            reproduction_within_tolerance=within,
            defects=[] if passed else ["preprocessing artifact missing from the logged model"],
            fixable=not passed,
            decision=None if passed else QADecision.INVALID_EXPERIMENT,
            mlflow_run_id=f"mlflow-qa0-{attempt}",
            contract_hash=contract.contract_hash,
            harness_version=contract.harness_version,
            runtime_seconds=20.0,
        )

    def run_qa_phase1(
        self,
        frozen: FrozenChampion,
        baseline_ref: str,
        contract: EvaluationContract,
    ) -> Phase1Result:
        self._check(contract)
        self.calls.append("run_qa_phase1")
        self.qa_split_reads += 1
        metric = contract.primary_metric
        baseline = BASELINE_VALUE
        val = frozen.validation_metrics.get(f"val/{metric}", BASELINE_VALUE + MAX_GAIN)

        if self.qa_outcome == "PASS":
            candidate = baseline + 0.75 * (val - baseline)
            ci = (0.012, 0.030)
            frac = 0.61
            failures: list[str] = []
        elif self.qa_outcome == "FAIL_HYPOTHESIS":
            candidate = baseline + 0.05 * (val - baseline)
            ci = (-0.004, 0.007)
            frac = 0.50
            failures = ["No holdout separation from baseline on head or tail users."]
        else:  # SPEC_REVIEW
            candidate = baseline - 0.04
            ci = (-0.061, -0.019)
            frac = 0.21
            failures = [
                "Validation-to-QA divergence far exceeds sampling noise.",
                "Label positives appear time-shifted relative to the serving decision point.",
                "Baseline also degrades on the QA split, suggesting an evaluation design problem.",
            ]

        mean_delta = candidate - baseline
        primary = PairedDelta(
            metric=metric,
            baseline_value=baseline,
            candidate_value=candidate,
            mean_delta=mean_delta,
            relative_delta=mean_delta / baseline if baseline else 0.0,
            ci_low=ci[0],
            ci_high=ci[1],
            fraction_improved=frac,
            n=10_000,
        )
        secondary = PairedDelta(
            metric="recall@10",
            baseline_value=baseline * 1.4,
            candidate_value=candidate * 1.4,
            mean_delta=mean_delta * 1.4,
            relative_delta=(mean_delta / baseline if baseline else 0.0),
            ci_low=ci[0] * 1.4,
            ci_high=ci[1] * 1.4,
            fraction_improved=frac,
            n=10_000,
        )
        cohort_deltas = {
            c.name: primary.model_copy(
                update={
                    "metric": f"{metric}[{c.name}]",
                    "mean_delta": mean_delta * (0.8 + 0.2 * idx),
                }
            )
            for idx, c in enumerate(contract.cohorts)
        }
        return Phase1Result(
            mlflow_run_id="mlflow-qa1",
            primary_metric=metric,
            qa_metrics={f"qa/{metric}": candidate, "qa/recall@10": candidate * 1.4},
            baseline_metrics={f"qa/{metric}": baseline, "qa/recall@10": baseline * 1.4},
            paired_deltas=[primary, secondary],
            cohort_metrics={
                c.name: {f"qa/{metric}": candidate * (0.93 + 0.03 * idx)}
                for idx, c in enumerate(contract.cohorts)
            },
            cohort_deltas=cohort_deltas,
            cold_start=primary.model_copy(
                update={"metric": f"{metric}[cold_start]", "mean_delta": mean_delta * 0.4}
            ),
            coverage={"item_coverage": 0.47},
            diversity={"intra_list_diversity": 0.63},
            calibration={"ece": 0.031},
            operational={
                "latency_ms_p50": 11.0,
                "latency_ms_p95": 24.0,
                "throughput_qps": 850.0,
                "model_size_mb": 48.0,
                "cost_usd_per_1k": 0.004,
            },
            stability_notes="Metrics stable across 5 bootstrap seeds.",
            failure_cases=failures,
            qa_population_size=10_000,
            contract_hash=contract.contract_hash,
            harness_version=contract.harness_version,
            runtime_seconds=45.0,
        )
