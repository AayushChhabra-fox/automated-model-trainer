"""Assemble the campaign graph (PLAN §4)."""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from amt.agents.base import LLMFactory
from amt.graph.deps import NodeDeps
from amt.graph.nodes.champion import (
    make_freeze_champion_node,
    make_select_hypothesis_champion_node,
)
from amt.graph.nodes.conclusion import (
    make_conclusion_node,
    make_hc_conclusion_node,
    make_log_final_mlflow_node,
)
from amt.graph.nodes.evaluation_contract import (
    make_evaluation_contract_node,
    make_hc_contract_node,
    make_materialize_splits_node,
    make_run_baseline_node,
)
from amt.graph.nodes.planner import (
    make_hc_feature_fit_node,
    make_hc_hypotheses_node,
    make_planner_feature_fit_node,
    make_planner_final_champion_node,
    make_planner_hypotheses_node,
    make_planner_select_hypothesis_node,
)
from amt.graph.nodes.problem_spec import make_hc_problem_spec_node, make_problem_spec_node
from amt.graph.nodes.qa import make_qa_decision_node, make_qa_phase0_node, make_qa_phase1_node
from amt.graph.nodes.trainer import (
    make_execute_training_node,
    make_trainer_interpret_node,
    make_trainer_propose_node,
)
from amt.graph.routing import (
    route_after_hc_contract,
    route_after_hc_feature_fit,
    route_after_hc_hypotheses,
    route_after_hc_problem_spec,
    route_after_planner_select,
    route_after_qa_phase0,
    route_after_trainer_interpret,
    route_after_trainer_propose,
)
from amt.graph.state import CampaignState
from amt.settings import Settings, get_settings

#: Packages whose own classes may appear in a ``CampaignState`` value.
CHECKPOINT_PACKAGES: tuple[str, ...] = ("amt.schemas", "amt.agents", "amt.graph")


@lru_cache(maxsize=1)
def checkpoint_types() -> tuple[type, ...]:
    """The types the checkpointer is allowed to reconstruct from a checkpoint.

    Without an allowlist LangGraph warns once per type per load ("Deserializing
    unregistered type"), which is seventeen lines in front of every command,
    and a later version will refuse outright. With one, a type that is *not*
    named comes back as a plain dict rather than an error -- so the list is
    enumerated from the packages instead of written out, because a schema added
    next month and forgotten here would corrupt a resumed campaign quietly.
    Only these packages' own classes are taken, so it cannot widen to whatever
    they happen to import.
    """
    import enum
    import importlib
    import pkgutil

    from amt.schemas.common import AmtModel

    types: set[type] = set()
    for name in CHECKPOINT_PACKAGES:
        package = importlib.import_module(name)
        for info in pkgutil.iter_modules(package.__path__):
            module = importlib.import_module(f"{name}.{info.name}")
            types.update(
                obj
                for obj in vars(module).values()
                if isinstance(obj, type)
                and obj.__module__ == module.__name__
                and issubclass(obj, (AmtModel, enum.Enum))
            )
    return tuple(sorted(types, key=lambda t: (t.__module__, t.__name__)))


def sqlite_checkpointer(path: str | Path | None = None, settings: Settings | None = None) -> Any:
    """``SqliteSaver`` at ``~/.amt/checkpoints.sqlite`` (thread_id == campaign_id)."""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = settings or get_settings()
    db = Path(path).expanduser() if path else settings.resolved_checkpoint_db()
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    # A reader (`amt status`, and the web surface Phase 7 adds) opens this file
    # while a runner writes it; the default is to fail immediately on the lock.
    conn.execute("PRAGMA busy_timeout=5000")
    return SqliteSaver(conn, serde=JsonPlusSerializer(allowed_msgpack_modules=checkpoint_types()))


def build_graph(
    executor: Any,
    llm_factory: LLMFactory | None = None,
    settings: Settings | None = None,
    sql_probe: Any | None = None,
    checkpointer: Any | None = None,
    use_mlflow: bool = False,
    code_version: str = "",
) -> Any:
    """Compile the campaign StateGraph.

    A checkpointer is required for the human checkpoints to suspend and resume; when
    none is supplied an in-memory saver is used (fine for tests and dry runs).
    """
    deps = NodeDeps(
        executor=executor,
        settings=settings or get_settings(),
        llm_factory=llm_factory,
        sql_probe=sql_probe,
        use_mlflow=use_mlflow,
        code_version=code_version,
    )

    graph = StateGraph(CampaignState)

    graph.add_node("problem_spec", make_problem_spec_node(deps))
    graph.add_node("hc_problem_spec", make_hc_problem_spec_node(deps))
    graph.add_node("evaluation_contract", make_evaluation_contract_node(deps))
    graph.add_node("materialize_splits", make_materialize_splits_node(deps))
    graph.add_node("hc_contract", make_hc_contract_node(deps))
    graph.add_node("run_baseline", make_run_baseline_node(deps))
    graph.add_node("planner_hypotheses", make_planner_hypotheses_node(deps))
    graph.add_node("hc_hypotheses", make_hc_hypotheses_node(deps))
    graph.add_node("planner_select_hypothesis", make_planner_select_hypothesis_node(deps))
    graph.add_node("planner_feature_fit", make_planner_feature_fit_node(deps))
    graph.add_node("hc_feature_fit", make_hc_feature_fit_node(deps))
    graph.add_node("trainer_propose", make_trainer_propose_node(deps))
    graph.add_node("execute_training", make_execute_training_node(deps))
    graph.add_node("trainer_interpret", make_trainer_interpret_node(deps))
    graph.add_node("select_hypothesis_champion", make_select_hypothesis_champion_node(deps))
    graph.add_node("planner_final_champion", make_planner_final_champion_node(deps))
    graph.add_node("freeze_champion", make_freeze_champion_node(deps))
    graph.add_node("qa_phase0", make_qa_phase0_node(deps))
    graph.add_node("qa_phase1", make_qa_phase1_node(deps))
    graph.add_node("qa_decision", make_qa_decision_node(deps))
    graph.add_node("conclusion", make_conclusion_node(deps))
    graph.add_node("log_final_mlflow", make_log_final_mlflow_node(deps))
    graph.add_node("hc_conclusion", make_hc_conclusion_node(deps))

    graph.add_edge(START, "problem_spec")
    graph.add_edge("problem_spec", "hc_problem_spec")
    graph.add_conditional_edges(
        "hc_problem_spec",
        route_after_hc_problem_spec,
        {"problem_spec": "problem_spec", "evaluation_contract": "evaluation_contract"},
    )
    graph.add_edge("evaluation_contract", "materialize_splits")
    graph.add_edge("materialize_splits", "hc_contract")
    graph.add_conditional_edges(
        "hc_contract",
        route_after_hc_contract,
        {"evaluation_contract": "evaluation_contract", "run_baseline": "run_baseline"},
    )
    graph.add_edge("run_baseline", "planner_hypotheses")
    graph.add_edge("planner_hypotheses", "hc_hypotheses")
    graph.add_conditional_edges(
        "hc_hypotheses",
        route_after_hc_hypotheses,
        {
            "planner_hypotheses": "planner_hypotheses",
            "planner_select_hypothesis": "planner_select_hypothesis",
        },
    )
    graph.add_conditional_edges(
        "planner_select_hypothesis",
        route_after_planner_select,
        {
            "trainer_propose": "trainer_propose",
            "planner_feature_fit": "planner_feature_fit",
            "planner_final_champion": "planner_final_champion",
        },
    )
    graph.add_edge("planner_feature_fit", "hc_feature_fit")
    graph.add_conditional_edges(
        "hc_feature_fit",
        route_after_hc_feature_fit,
        {
            "planner_feature_fit": "planner_feature_fit",
            "trainer_propose": "trainer_propose",
        },
    )
    graph.add_conditional_edges(
        "trainer_propose",
        route_after_trainer_propose,
        {
            "execute_training": "execute_training",
            "planner_feature_fit": "planner_feature_fit",
        },
    )
    graph.add_edge("execute_training", "trainer_interpret")
    graph.add_conditional_edges(
        "trainer_interpret",
        route_after_trainer_interpret,
        {
            "trainer_propose": "trainer_propose",
            "select_hypothesis_champion": "select_hypothesis_champion",
        },
    )
    graph.add_edge("select_hypothesis_champion", "planner_select_hypothesis")
    graph.add_edge("planner_final_champion", "freeze_champion")
    graph.add_edge("freeze_champion", "qa_phase0")
    graph.add_conditional_edges(
        "qa_phase0",
        route_after_qa_phase0,
        {"qa_phase0": "qa_phase0", "qa_phase1": "qa_phase1", "conclusion": "conclusion"},
    )
    graph.add_edge("qa_phase1", "qa_decision")
    graph.add_edge("qa_decision", "conclusion")
    graph.add_edge("conclusion", "log_final_mlflow")
    graph.add_edge("log_final_mlflow", "hc_conclusion")
    graph.add_edge("hc_conclusion", END)

    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
