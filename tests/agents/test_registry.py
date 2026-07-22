"""Tests for the model catalog registry."""

from pathlib import Path

import pytest

from adoption_accelerator.agents.llm.registry import (
    VISION_REQUIRED_ROLES,
    ModelsConfig,
    clear_registry_cache,
    load_models_config,
    resolve_role,
)

CONFIG_PATH = Path("configs/agents/models.yaml")


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_registry_cache()
    yield
    clear_registry_cache()


def test_load_models_config_parses_repo_file():
    cfg = load_models_config(CONFIG_PATH)
    assert "gpt-5-mini" in cfg.catalog
    assert cfg.defaults.model in cfg.catalog
    assert "visual_analyst" in cfg.roles


def test_resolve_role_applies_defaults():
    resolved = resolve_role("data_analyst", load_models_config(CONFIG_PATH))
    assert resolved.api_model == "gpt-5-nano"
    assert resolved.provider == "openai"
    assert resolved.max_output_tokens == 2048  # from defaults
    assert resolved.reasoning_effort == "minimal"


def test_resolve_role_role_override_wins():
    resolved = resolve_role("recommendation_agent", load_models_config(CONFIG_PATH))
    assert resolved.max_output_tokens == 4096  # role override


def test_unknown_role_falls_back_to_defaults_model():
    resolved = resolve_role("nonexistent_role", load_models_config(CONFIG_PATH))
    assert resolved.model_key == "gpt-5-mini"


def test_vision_required_role_with_blind_model_fails():
    raw = {
        "catalog": {
            "blind": {
                "api_model": "blind",
                "provider": "openai",
                "supports_vision": False,
                "pricing": {"input_usd_per_1m": 0.1, "output_usd_per_1m": 0.1},
            }
        },
        "defaults": {"model": "blind"},
        "roles": {"visual_analyst": {"model": "blind"}},
    }
    with pytest.raises(ValueError, match="supports_vision"):
        ModelsConfig.model_validate(raw)


def test_role_referencing_missing_model_fails():
    raw = {
        "catalog": {
            "m": {
                "api_model": "m",
                "provider": "openai",
                "pricing": {"input_usd_per_1m": 0.1, "output_usd_per_1m": 0.1},
            }
        },
        "defaults": {"model": "m"},
        "roles": {"data_analyst": {"model": "ghost"}},
    }
    with pytest.raises(ValueError, match="ghost"):
        ModelsConfig.model_validate(raw)


def test_visual_analyst_is_vision_required():
    assert "visual_analyst" in VISION_REQUIRED_ROLES


def test_vision_required_role_absent_falls_back_to_blind_defaults_fails():
    raw = {
        "catalog": {
            "blind": {
                "api_model": "blind",
                "provider": "openai",
                "supports_vision": False,
                "pricing": {"input_usd_per_1m": 0.1, "output_usd_per_1m": 0.1},
            }
        },
        "defaults": {"model": "blind"},
        "roles": {},
    }
    with pytest.raises(ValueError, match="supports_vision"):
        ModelsConfig.model_validate(raw)
