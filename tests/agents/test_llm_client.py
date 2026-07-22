"""Tests for the multi-provider chat model factory."""

import pytest
from langchain_core.messages import AIMessage

import adoption_accelerator.agents.llm.client as client_mod
from adoption_accelerator.agents.llm.client import (
    clear_model_cache,
    extract_usage,
    get_chat_model,
)
from adoption_accelerator.agents.llm.registry import (
    ModelPricing,
    ModelSpec,
    ResolvedModel,
)


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    clear_model_cache()
    captured = {}

    def fake_init_chat_model(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(client_mod, "init_chat_model", fake_init_chat_model)
    yield captured
    clear_model_cache()


def test_get_chat_model_passes_resolved_params(_fresh):
    get_chat_model("data_analyst")
    assert _fresh["model"] == "gpt-5-nano"
    assert _fresh["model_provider"] == "openai"
    assert _fresh["max_tokens"] == 2048
    assert _fresh["reasoning_effort"] == "minimal"


def test_get_chat_model_omits_reasoning_effort_for_non_effort_models(_fresh, monkeypatch):
    fake_spec = ModelSpec(
        api_model="fake",
        provider="openai",
        supports_vision=False,
        reasoning_kind="none",
        pricing=ModelPricing(input_usd_per_1m=0.1, output_usd_per_1m=0.1),
    )

    class _FakeConfig:
        catalog = {"fake": fake_spec}

    fake_resolved = ResolvedModel(
        role="some_role",
        model_key="fake",
        api_model="fake",
        provider="openai",
        supports_vision=False,
        reasoning_effort="minimal",
        max_output_tokens=2048,
        pricing=fake_spec.pricing,
    )

    monkeypatch.setattr(client_mod, "load_models_config", lambda: _FakeConfig())
    monkeypatch.setattr(client_mod, "resolve_role", lambda role: fake_resolved)

    get_chat_model("some_role")

    assert _fresh["model"] == "fake"
    assert "reasoning_effort" not in _fresh


def test_get_chat_model_is_cached(_fresh):
    first = get_chat_model("synthesizer")
    second = get_chat_model("synthesizer")
    assert first is second


def test_extract_usage_reads_usage_metadata():
    msg = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )
    usage = extract_usage(msg, model_key="gpt-5-mini")
    assert usage == {
        "input_tokens": 10,
        "output_tokens": 5,
        "model_key": "gpt-5-mini",
    }


def test_extract_usage_handles_missing_metadata():
    usage = extract_usage(AIMessage(content="ok"), model_key="gpt-5-nano")
    assert usage == {"input_tokens": 0, "output_tokens": 0, "model_key": "gpt-5-nano"}
