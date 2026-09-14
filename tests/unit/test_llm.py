"""Provider dispatch for agent chat models. No provider SDK is imported here."""

from __future__ import annotations

import sys
import types

import pytest
from pydantic import BaseModel

from amt.llm import (
    ANTHROPIC,
    HUGGINGFACE,
    OPENAI,
    OPENAI_COMPAT,
    LLMConfigError,
    build_chat_model,
    make_llm_factory,
    parse_model_ref,
    structured_output,
)
from amt.settings import Settings


def make_settings(**overrides) -> Settings:
    """Settings isolated from the developer's environment."""
    return Settings(_env_file=None, **overrides)


class StubChatModel:
    """Records the kwargs a provider class would have been constructed with."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


@pytest.fixture
def stub_providers(monkeypatch):
    """Install fake ``langchain_anthropic`` / ``langchain_openai`` modules."""
    classes: dict[str, type] = {}
    for module_name, class_name in (
        ("langchain_anthropic", "ChatAnthropic"),
        ("langchain_openai", "ChatOpenAI"),
    ):
        module = types.ModuleType(module_name)
        stub = type(class_name, (StubChatModel,), {})
        setattr(module, class_name, stub)
        monkeypatch.setitem(sys.modules, module_name, module)
        classes[class_name] = stub
    return classes


# --- parsing ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "provider", "model"),
    [
        ("anthropic:claude-opus-5", ANTHROPIC, "claude-opus-5"),
        ("openai:gpt-4.1", OPENAI, "gpt-4.1"),
        (
            "huggingface:meta-llama/Llama-3.1-70B-Instruct",
            HUGGINGFACE,
            "meta-llama/Llama-3.1-70B-Instruct",
        ),
        ("openai-compat:my-served-model", OPENAI_COMPAT, "my-served-model"),
        ("OpenAI: gpt-4.1 ", OPENAI, "gpt-4.1"),
        ("openai:ft:gpt-4.1:acme", OPENAI, "ft:gpt-4.1:acme"),
        # Unprefixed names keep older .env files working.
        ("claude-sonnet-5", ANTHROPIC, "claude-sonnet-5"),
        ("gpt-4.1", OPENAI, "gpt-4.1"),
        ("o3-mini", OPENAI, "o3-mini"),
        ("meta-llama/Llama-3.1-8B-Instruct", HUGGINGFACE, "meta-llama/Llama-3.1-8B-Instruct"),
    ],
)
def test_parse_model_ref(raw: str, provider: str, model: str) -> None:
    ref = parse_model_ref(raw)
    assert (ref.provider, ref.model) == (provider, model)
    assert str(ref) == f"{provider}:{model}"


@pytest.mark.parametrize("raw", ["", "   ", "llama3", "openai:", "bedrock:claude-opus-5"])
def test_parse_model_ref_rejects(raw: str) -> None:
    with pytest.raises(LLMConfigError):
        parse_model_ref(raw)


# --- construction ------------------------------------------------------------------


def test_anthropic_uses_anthropic_sdk(stub_providers) -> None:
    settings = make_settings(AMT_LLM_API_KEY="k-anthropic")
    llm = build_chat_model("anthropic:claude-opus-5", settings)

    assert isinstance(llm, stub_providers["ChatAnthropic"])
    assert llm.kwargs["model"] == "claude-opus-5"
    assert llm.kwargs["api_key"] == "k-anthropic"
    assert llm.kwargs["temperature"] == settings.llm_temperature
    assert llm.kwargs["max_retries"] == settings.llm_max_retries
    assert "base_url" not in llm.kwargs


def test_anthropic_ignores_base_url(stub_providers) -> None:
    settings = make_settings(AMT_LLM_API_KEY="k", AMT_LLM_BASE_URL="https://proxy.internal/v1")
    llm = build_chat_model("anthropic:claude-opus-5", settings)

    assert "base_url" not in llm.kwargs


def test_openai_without_base_url(stub_providers) -> None:
    settings = make_settings(AMT_LLM_API_KEY="k-openai")
    llm = build_chat_model("openai:gpt-4.1", settings)

    assert isinstance(llm, stub_providers["ChatOpenAI"])
    assert llm.kwargs["model"] == "gpt-4.1"
    assert llm.kwargs["api_key"] == "k-openai"
    assert "base_url" not in llm.kwargs


@pytest.mark.parametrize(
    "model",
    ["huggingface:meta-llama/Llama-3.1-70B-Instruct", "openai-compat:my-served-model"],
)
def test_http_compatible_providers_use_base_url(stub_providers, model: str) -> None:
    settings = make_settings(AMT_LLM_API_KEY="k-hf", AMT_LLM_BASE_URL="https://endpoint.example/v1")
    llm = build_chat_model(model, settings)

    assert isinstance(llm, stub_providers["ChatOpenAI"])
    assert llm.kwargs["base_url"] == "https://endpoint.example/v1"
    assert llm.kwargs["api_key"] == "k-hf"


@pytest.mark.parametrize(
    "model",
    ["huggingface:meta-llama/Llama-3.1-70B-Instruct", "openai-compat:my-served-model"],
)
def test_http_compatible_providers_require_base_url(stub_providers, model: str) -> None:
    settings = make_settings(AMT_LLM_API_KEY="k")

    with pytest.raises(LLMConfigError, match="AMT_LLM_BASE_URL"):
        build_chat_model(model, settings)


def test_missing_api_key_fails_fast(stub_providers) -> None:
    with pytest.raises(LLMConfigError, match="AMT_LLM_API_KEY"):
        build_chat_model("openai:gpt-4.1", make_settings())


@pytest.mark.parametrize("alias", ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "HF_TOKEN"])
def test_vendor_key_aliases_populate_llm_api_key(alias: str) -> None:
    assert make_settings(**{alias: "k-vendor"}).llm_api_key == "k-vendor"


def test_factory_dispatches_per_model(stub_providers) -> None:
    settings = make_settings(AMT_LLM_API_KEY="k")
    factory = make_llm_factory(settings)

    assert isinstance(factory("anthropic:claude-opus-5"), stub_providers["ChatAnthropic"])
    assert isinstance(factory("openai:gpt-4.1"), stub_providers["ChatOpenAI"])


def test_default_agent_models_are_provider_qualified() -> None:
    settings = make_settings()

    for agent in ("planner", "trainer", "qa", "problem_spec"):
        assert parse_model_ref(settings.model_for(agent)).provider == ANTHROPIC


class RecordingChatModel:
    """Records how a schema was bound, the way a provider client would see it."""

    def __init__(self) -> None:
        self.bound: dict = {}

    def with_structured_output(self, schema: type, **kwargs) -> str:
        self.bound = {"schema": schema, **kwargs}
        return "runnable"


class OpenSchema(BaseModel):
    """A contract with an open mapping, which strict JSON schema cannot express."""

    params: dict = {}


@pytest.mark.parametrize(
    "model",
    [
        "openai:gpt-4.1",
        "huggingface:meta-llama/Llama-3.1-70B-Instruct",
        "openai-compat:my-served-model",
    ],
)
def test_strict_schema_providers_bind_via_tool_calling(model: str) -> None:
    llm = RecordingChatModel()

    assert structured_output(llm, OpenSchema, model) == "runnable"
    assert llm.bound == {"schema": OpenSchema, "method": "function_calling"}


@pytest.mark.parametrize("model", ["anthropic:claude-opus-5", "", "not-a-model-ref"])
def test_other_providers_keep_the_default_binding(model: str) -> None:
    llm = RecordingChatModel()

    assert structured_output(llm, OpenSchema, model) == "runnable"
    assert llm.bound == {"schema": OpenSchema}
