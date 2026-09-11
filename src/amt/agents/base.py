"""Structured-output plumbing shared by every agent.

Agents are pure functions of their inputs: they read contracts and return a Pydantic
schema. They never touch Databricks, MLflow or campaign state — nodes do that.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, TypeVar

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from amt.settings import Settings, get_settings

T = TypeVar("T", bound=BaseModel)

#: ``(model_name) -> chat model`` with ``.with_structured_output(PydanticModel)``.
LLMFactory = Callable[[str], Any]

PROMPT_DIR = Path(__file__).parent / "prompts"


class AgentError(RuntimeError):
    """Raised when an agent cannot produce a valid structured result."""


@cache
def load_prompt(name: str) -> str:
    """Load ``prompts/<name>.md``."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        raise AgentError(f"Prompt not found: {path}")
    return path.read_text(encoding="utf-8")


def make_llm_factory(settings: Settings | None = None) -> LLMFactory:
    """Default factory: ``ChatAnthropic`` per agent model name."""
    settings = settings or get_settings()

    def factory(model: str) -> Any:
        from langchain_anthropic import ChatAnthropic  # lazy: no API key needed for tests

        kwargs: dict[str, Any] = {
            "model": model,
            "temperature": settings.llm_temperature,
            "max_retries": settings.llm_max_retries,
        }
        if settings.anthropic_api_key:
            kwargs["api_key"] = settings.anthropic_api_key
        return ChatAnthropic(**kwargs)

    return factory


def render(value: Any) -> str:
    """Render any contract/value as compact, deterministic text for a prompt."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    if isinstance(value, (list, tuple)):
        return json.dumps([json.loads(render(v)) if isinstance(v, BaseModel) else v for v in value],
                          indent=2, default=str)
    return json.dumps(value, indent=2, default=str, sort_keys=True)


def render_context(context: dict[str, Any]) -> str:
    """Render a mapping of section name -> value as a markdown document."""
    parts: list[str] = []
    for key, value in context.items():
        if value is None or value == [] or value == {}:
            continue
        parts.append(f"## {key}\n\n{render(value)}")
    return "\n\n".join(parts)


class StructuredAgent:
    """Wraps ``ChatAnthropic.with_structured_output(schema)`` behind a testable seam."""

    def __init__(
        self,
        name: str,
        prompt: str,
        output_model: type[T],
        model: str,
        llm_factory: LLMFactory | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.name = name
        self.prompt = prompt
        self.output_model = output_model
        self.model = model
        self.settings = settings or get_settings()
        self._llm_factory = llm_factory or make_llm_factory(self.settings)

    def invoke(self, context: dict[str, Any] | str, instruction: str = "") -> T:
        """Run the agent and return an instance of ``output_model``."""
        body = context if isinstance(context, str) else render_context(context)
        if instruction:
            body = f"{instruction}\n\n{body}"
        messages = [SystemMessage(content=self.prompt), HumanMessage(content=body)]
        llm = self._llm_factory(self.model)
        structured = llm.with_structured_output(self.output_model)
        result = structured.invoke(messages)
        if not isinstance(result, self.output_model):
            raise AgentError(
                f"{self.name} returned {type(result).__name__}, expected "
                f"{self.output_model.__name__}"
            )
        return result


class AgentBase:
    """Common wiring for the four campaign agents."""

    agent_key: str = "planner"

    def __init__(
        self,
        llm_factory: LLMFactory | None = None,
        settings: Settings | None = None,
        model: str | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.llm_factory = llm_factory
        self.model = model or self.settings.model_for(self.agent_key)

    def agent(self, name: str, prompt_name: str, output_model: type[T]) -> StructuredAgent:
        return StructuredAgent(
            name=name,
            prompt=load_prompt(prompt_name),
            output_model=output_model,
            model=self.model,
            llm_factory=self.llm_factory,
            settings=self.settings,
        )
