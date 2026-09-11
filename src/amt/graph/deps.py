"""Dependency bundle handed to node factories (executor, agents, settings)."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from amt.agents.base import LLMFactory
from amt.agents.planner_agent import PlannerAgent
from amt.agents.problem_spec_agent import ProblemSpecAgent
from amt.agents.qa_agent import QAAgent
from amt.agents.trainer_agent import TrainerAgent
from amt.settings import Settings, get_settings


@dataclass
class NodeDeps:
    """Everything the nodes need. Agents are built lazily so tests stay cheap."""

    executor: Any
    settings: Settings = field(default_factory=get_settings)
    llm_factory: LLMFactory | None = None
    sql_probe: Any | None = None
    use_mlflow: bool = False
    code_version: str = ""

    @cached_property
    def problem_spec_agent(self) -> ProblemSpecAgent:
        return ProblemSpecAgent(
            llm_factory=self.llm_factory, settings=self.settings, sql_probe=self.sql_probe
        )

    @cached_property
    def planner_agent(self) -> PlannerAgent:
        return PlannerAgent(llm_factory=self.llm_factory, settings=self.settings)

    @cached_property
    def trainer_agent(self) -> TrainerAgent:
        return TrainerAgent(llm_factory=self.llm_factory, settings=self.settings)

    @cached_property
    def qa_agent(self) -> QAAgent:
        return QAAgent(llm_factory=self.llm_factory, settings=self.settings)
