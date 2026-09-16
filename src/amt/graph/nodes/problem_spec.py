"""Stage 1 node + HC2 (ProblemSpec approval)."""

from __future__ import annotations

from collections.abc import Callable

from langgraph.types import interrupt

from amt.graph.deps import NodeDeps
from amt.graph.human import gate_update, parse_resume
from amt.graph.state import CampaignState, Stage


def make_problem_spec_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """Investigate the data, then draft (or redraft) the ProblemSpec."""

    def problem_spec_node(state: CampaignState) -> dict:
        agent = deps.problem_spec_agent
        previous = state.get("problem_spec")
        feedback = state.get("human_feedback")
        observations = list(previous.investigation_notes) if previous else []
        if not observations:
            observations = agent.investigate(state.get("problem_text", ""))
        spec = agent.draft_spec(
            problem_text=state.get("problem_text", ""),
            observations=observations,
            previous=previous,
            human_feedback=feedback,
        )
        return {
            "stage": Stage.PROBLEM_SPEC.value,
            "status": "running",
            "problem_spec": spec,
            "spec_hash": spec.spec_hash,
            "problem_spec_approved": False,
            "human_feedback": None,
        }

    return problem_spec_node


def make_hc_problem_spec_node(deps: NodeDeps) -> Callable[[CampaignState], dict]:
    """HC2 — present the ProblemSpec and wait for approval."""

    def hc_problem_spec_node(state: CampaignState) -> dict:
        spec = state.get("problem_spec")
        payload = interrupt(
            {
                "checkpoint": "HC2",
                "title": "ProblemSpec approval",
                "campaign_id": state.get("campaign_id"),
                "problem_spec": spec.model_dump(mode="json") if spec else None,
                "instructions": "amt approve <campaign> | amt reject <campaign> -m '<feedback>'",
            }
        )
        decision = parse_resume(payload)
        return {
            "problem_spec_approved": decision.approved,
            **gate_update(state, decision),
        }

    return hc_problem_spec_node
