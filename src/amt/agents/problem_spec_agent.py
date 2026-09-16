"""Stage 1/2 agent: investigates the data, drafts the ProblemSpec and EvaluationContract."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from amt.agents.base import AgentBase, LLMFactory
from amt.databricks.sql import SqlProbe
from amt.schemas.common import AmtModel, CohortDef, MetricDef, SplitRef
from amt.schemas.evaluation_contract import (
    HARNESS_VERSION,
    AcceptanceCriteria,
    DatasetDefinition,
    EvaluationContract,
    ProductMetricSource,
    RelevanceWeighting,
    SplitPolicy,
)
from amt.schemas.problem_spec import EvaluationMode, ProbeObservation, ProblemSpec
from amt.settings import Settings

ProbeAction = Literal["describe_table", "profile", "sample", "done"]


class ProbeRequest(AmtModel):
    """One step of the read-only data investigation (a tool call in schema form)."""

    action: ProbeAction
    target: str = Field(default="", description="Fully-qualified table name.")
    columns: list[str] = Field(default_factory=list)
    limit: int = Field(default=10, ge=1, le=100)
    reason: str = ""


class SplitSpec(AmtModel):
    """Split definition before materialization fills in the immutable reference."""

    table: str = Field(
        default="",
        description="Leave empty: materialization names the output table.",
    )
    description: str = ""
    temporal_range: tuple[str, str] | None = None


class EvaluationContractDraft(AmtModel):
    """LLM-authored part of the contract; immutable references are filled in by execution."""

    dataset: DatasetDefinition
    train: SplitSpec
    validation: SplitSpec
    qa: SplitSpec
    split_policy: SplitPolicy
    user_population: str = ""
    item_population: str = ""
    label_definition: str = ""
    candidate_generation_rules: str = ""
    cohorts: list[CohortDef] = Field(default_factory=list)
    metrics: list[MetricDef] = Field(default_factory=list)
    primary_metric: str
    # Carried from the ProblemSpec rather than re-decided: the mode and the
    # weighting are the framing a human approved at HC2, and a contract that
    # quietly changed either would be scoring a different question.
    evaluation_mode: EvaluationMode = "next_item"
    relevance_weighting: RelevanceWeighting = "none"
    product_metric_source: ProductMetricSource | None = None
    acceptance_criteria: AcceptanceCriteria


def draft_to_contract(
    draft: EvaluationContractDraft,
    spec_hash: str,
    harness_version: str = HARNESS_VERSION,
) -> EvaluationContract:
    """Assemble a full :class:`EvaluationContract` from an LLM draft.

    Split row counts, Delta versions and membership hashes are placeholders until
    ``Executor.materialize_splits`` returns immutable references.
    """

    def ref(name: str, spec: SplitSpec) -> SplitRef:
        return SplitRef(
            name=name,  # type: ignore[arg-type]
            table=spec.table,
            delta_version=0,
            row_count=0,
            membership_hash="",
            description=spec.description,
            temporal_range=spec.temporal_range,
        )

    criteria = draft.acceptance_criteria
    if criteria.primary_metric != draft.primary_metric:
        criteria = criteria.model_copy(update={"primary_metric": draft.primary_metric})
    metrics = list(draft.metrics)
    if draft.primary_metric not in {m.name for m in metrics}:
        metrics.append(
            MetricDef(
                name=draft.primary_metric,
                definition="Primary offline metric carried over from the ProblemSpec.",
                is_primary=True,
            )
        )
    return EvaluationContract(
        spec_hash=spec_hash,
        dataset=draft.dataset,
        train_split=ref("train", draft.train),
        validation_split=ref("validation", draft.validation),
        qa_split=ref("qa", draft.qa),
        split_policy=draft.split_policy,
        user_population=draft.user_population,
        item_population=draft.item_population,
        label_definition=draft.label_definition,
        candidate_generation_rules=draft.candidate_generation_rules,
        cohorts=draft.cohorts,
        metrics=metrics,
        primary_metric=draft.primary_metric,
        evaluation_mode=draft.evaluation_mode,
        relevance_weighting=draft.relevance_weighting,
        product_metric_source=draft.product_metric_source,
        harness_version=harness_version,
        acceptance_criteria=criteria,
    )


class ProblemSpecAgent(AgentBase):
    """Stage 1 (+ Stage 2 drafting). Investigates before it specifies."""

    agent_key = "problem_spec"

    def __init__(
        self,
        llm_factory: LLMFactory | None = None,
        settings: Settings | None = None,
        sql_probe: SqlProbe | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(llm_factory=llm_factory, settings=settings, model=model)
        self.sql_probe = sql_probe

    # --- investigation -------------------------------------------------------
    def investigate(
        self,
        problem_text: str,
        known_tables: list[str] | None = None,
        max_steps: int | None = None,
    ) -> list[ProbeObservation]:
        """Tool-calling loop over the read-only SQL probe. Returns what was learned."""
        if self.sql_probe is None:
            return []
        max_steps = max_steps or self.settings.max_probe_steps
        prober = self.agent("problem_spec.probe", "problem_spec_probe", ProbeRequest)
        observations: list[ProbeObservation] = []
        seen: set[tuple[str, str]] = set()

        for step in range(max_steps):
            request = prober.invoke(
                {
                    "problem": problem_text,
                    "known_tables": known_tables or [],
                    "observations_so_far": [o.model_dump() for o in observations],
                    "steps_remaining": max_steps - step,
                }
            )
            if request.action == "done":
                break
            key = (request.action, request.target)
            if key in seen:
                break
            seen.add(key)
            observations.append(self._run_probe(request))
        return observations

    def _run_probe(self, request: ProbeRequest) -> ProbeObservation:
        probe = self.sql_probe
        assert probe is not None
        try:
            if request.action == "describe_table":
                rows = probe.describe_table(request.target)
                return ProbeObservation(
                    action="describe_table",
                    target=request.target,
                    summary=f"{len(rows)} columns",
                    rows=list(rows)[:50],
                )
            if request.action == "profile":
                profile = probe.profile(request.target, request.columns or None)
                return ProbeObservation(
                    action="profile",
                    target=request.target,
                    summary=f"row_count={profile.get('row_count')}",
                    rows=[profile],
                )
            rows = probe.sample(request.target, request.limit)
            return ProbeObservation(
                action="sample",
                target=request.target,
                summary=f"{len(rows)} sample rows",
                rows=list(rows)[:20],
            )
        except Exception as exc:  # a failed probe is itself an observation
            return ProbeObservation(
                action=request.action,
                target=request.target,
                summary=f"probe failed: {exc}",
            )

    # --- drafting -------------------------------------------------------------
    def draft_spec(
        self,
        problem_text: str,
        observations: list[ProbeObservation] | None = None,
        previous: ProblemSpec | None = None,
        human_feedback: str | None = None,
    ) -> ProblemSpec:
        """Stage 1 output, presented to the human at HC2."""
        agent = self.agent("problem_spec", "problem_spec", ProblemSpec)
        spec = agent.invoke(
            {
                "product_problem": problem_text,
                "data_investigation": [o.model_dump() for o in observations or []],
                "previous_draft": previous,
                "human_feedback": human_feedback,
            }
        )
        update: dict[str, Any] = {"problem_text": problem_text}
        if observations:
            update["investigation_notes"] = observations
        if previous is not None:
            update["revision"] = previous.revision + 1
        return spec.model_copy(update=update)

    def draft_contract(
        self,
        spec: ProblemSpec,
        previous: EvaluationContract | None = None,
        human_feedback: str | None = None,
        harness_version: str = HARNESS_VERSION,
    ) -> EvaluationContract:
        """Stage 2 output, presented to the human at HC3 (after split materialization)."""
        agent = self.agent("evaluation_contract", "evaluation_contract", EvaluationContractDraft)
        draft = agent.invoke(
            {
                "problem_spec": spec,
                "harness_version": harness_version,
                "previous_draft": previous,
                "human_feedback": human_feedback,
            }
        )
        contract = draft_to_contract(
            draft, spec_hash=spec.spec_hash, harness_version=harness_version
        )
        if previous is not None:
            contract = contract.model_copy(update={"revision": previous.revision + 1})
        return contract
