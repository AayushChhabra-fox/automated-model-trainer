"""Provider-agnostic chat-model construction.

Agents only ever need ``llm_factory(model).with_structured_output(schema)``, so nothing
above this module knows which vendor answers. Models are named ``provider:model_id``,
which keeps a campaign's configuration identical whether it runs against Anthropic,
OpenAI, or any OpenAI-compatible HTTP endpoint (Hugging Face Inference Endpoints, vLLM,
TGI, Databricks model serving).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from amt.settings import Settings, get_settings

#: ``(model_name) -> chat model`` with ``.with_structured_output(PydanticModel)``.
LLMFactory = Callable[[str], Any]

ANTHROPIC = "anthropic"
OPENAI = "openai"
HUGGINGFACE = "huggingface"
OPENAI_COMPAT = "openai-compat"

PROVIDERS = (ANTHROPIC, OPENAI, HUGGINGFACE, OPENAI_COMPAT)

#: Providers that are only reachable through an explicitly configured base URL.
REQUIRES_BASE_URL = (HUGGINGFACE, OPENAI_COMPAT)

_PROVIDER_HINT = "use 'provider:model_id', e.g. openai:gpt-4.1 — providers: " + ", ".join(PROVIDERS)


class LLMConfigError(RuntimeError):
    """Raised when a model reference or its provider credentials cannot be resolved."""


@dataclass(frozen=True)
class ModelRef:
    """A parsed ``provider:model_id`` reference."""

    provider: str
    model: str

    def __str__(self) -> str:
        return f"{self.provider}:{self.model}"


def _infer_provider(model: str) -> str:
    """Best-effort provider for an unprefixed model name."""
    lowered = model.lower()
    if lowered.startswith("claude"):
        return ANTHROPIC
    if lowered.startswith(("gpt-", "o1", "o3", "o4")):
        return OPENAI
    if "/" in model:  # Hugging Face repo ids are always org/name
        return HUGGINGFACE
    raise LLMConfigError(f"Cannot infer a provider for model {model!r}: {_PROVIDER_HINT}.")


def parse_model_ref(raw: str) -> ModelRef:
    """Parse ``provider:model_id``, falling back to inference for unprefixed names."""
    name = (raw or "").strip()
    if not name:
        raise LLMConfigError(f"Empty model name: {_PROVIDER_HINT}.")
    head, separator, tail = name.partition(":")
    if not separator:
        return ModelRef(_infer_provider(name), name)
    provider = head.strip().lower()
    model = tail.strip()
    if provider not in PROVIDERS:
        raise LLMConfigError(f"Unknown provider {head!r} in {name!r}: {_PROVIDER_HINT}.")
    if not model:
        raise LLMConfigError(f"Missing model id in {name!r}: {_PROVIDER_HINT}.")
    return ModelRef(provider, model)


def build_chat_model(model: str, settings: Settings | None = None) -> Any:
    """Construct the LangChain chat model named by ``model``."""
    settings = settings or get_settings()
    ref = parse_model_ref(model)

    if not settings.llm_api_key:
        raise LLMConfigError(f"No API key for {ref}; set AMT_LLM_API_KEY.")
    base_url = settings.llm_base_url
    if ref.provider in REQUIRES_BASE_URL and not base_url:
        raise LLMConfigError(
            f"{ref.provider} models are served over an OpenAI-compatible endpoint; "
            "set AMT_LLM_BASE_URL."
        )

    kwargs: dict[str, Any] = {
        "model": ref.model,
        "temperature": settings.llm_temperature,
        "max_retries": settings.llm_max_retries,
        "api_key": settings.llm_api_key,
    }

    # Lazy imports: provider SDKs stay optional and tests never load them.
    if ref.provider == ANTHROPIC:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**kwargs)

    from langchain_openai import ChatOpenAI

    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def make_llm_factory(settings: Settings | None = None) -> LLMFactory:
    """Default factory: one chat model per ``provider:model_id`` name."""
    settings = settings or get_settings()

    def factory(model: str) -> Any:
        return build_chat_model(model, settings)

    return factory
