"""Model catalog registry.

Loads and validates ``configs/agents/models.yaml`` (catalog / defaults /
roles) and resolves a role name into a fully-specified model choice.
Fail-fast rule: any role in ``VISION_REQUIRED_ROLES`` must resolve to a
model with ``supports_vision: true``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from adoption_accelerator import config

VISION_REQUIRED_ROLES: frozenset[str] = frozenset({"visual_analyst"})

_DEFAULT_CONFIG_PATH = config.PROJECT_ROOT / "configs" / "agents" / "models.yaml"

_config_cache: dict[Path, "ModelsConfig"] = {}


class ModelPricing(BaseModel):
    """USD per 1M tokens."""

    input_usd_per_1m: float
    output_usd_per_1m: float
    cached_input_usd_per_1m: Optional[float] = None


class ModelSpec(BaseModel):
    api_model: str
    provider: str
    display_name: str = ""
    supports_vision: bool = False
    reasoning_kind: Literal["effort", "budget", "level", "none"] = "none"
    notes: str = ""
    pricing: ModelPricing


class RoleConfig(BaseModel):
    model: str
    reasoning_effort: Optional[str] = None
    max_output_tokens: Optional[int] = None


class ModelsDefaults(BaseModel):
    model: str
    reasoning_effort: str = "minimal"
    max_output_tokens: int = 2048


class ResolvedModel(BaseModel):
    """A role resolved against the catalog and defaults."""

    role: str
    model_key: str
    api_model: str
    provider: str
    supports_vision: bool
    reasoning_effort: str
    max_output_tokens: int
    pricing: ModelPricing


class ModelsConfig(BaseModel):
    catalog: dict[str, ModelSpec]
    defaults: ModelsDefaults
    roles: dict[str, RoleConfig] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_references(self) -> "ModelsConfig":
        if self.defaults.model not in self.catalog:
            raise ValueError(
                f"defaults.model '{self.defaults.model}' not in catalog"
            )
        for role, role_cfg in self.roles.items():
            if role_cfg.model not in self.catalog:
                raise ValueError(
                    f"role '{role}' references model '{role_cfg.model}' "
                    f"which is not in catalog"
                )
            if role in VISION_REQUIRED_ROLES:
                spec = self.catalog[role_cfg.model]
                if not spec.supports_vision:
                    raise ValueError(
                        f"role '{role}' requires vision but model "
                        f"'{role_cfg.model}' has supports_vision=false"
                    )
        for role in VISION_REQUIRED_ROLES:
            role_cfg = self.roles.get(role)
            model_key = role_cfg.model if role_cfg is not None else self.defaults.model
            spec = self.catalog.get(model_key)
            if spec is not None and not spec.supports_vision:
                raise ValueError(
                    f"role '{role}' requires vision but resolved model "
                    f"'{model_key}' has supports_vision=false"
                )
        return self


def load_models_config(path: Path | None = None) -> ModelsConfig:
    """Load and validate the models config (cached per path)."""
    resolved_path = (path or _DEFAULT_CONFIG_PATH).resolve()
    if resolved_path not in _config_cache:
        with open(resolved_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        _config_cache[resolved_path] = ModelsConfig.model_validate(raw)
    return _config_cache[resolved_path]


def resolve_role(role: str, config: ModelsConfig | None = None) -> ResolvedModel:
    """Resolve a role name to a concrete model configuration.

    Unknown roles fall back to ``defaults.model`` with default params.
    """
    cfg = config or load_models_config()
    role_cfg = cfg.roles.get(role)
    model_key = role_cfg.model if role_cfg else cfg.defaults.model
    spec = cfg.catalog[model_key]

    reasoning_effort = cfg.defaults.reasoning_effort
    max_output_tokens = cfg.defaults.max_output_tokens
    if role_cfg is not None:
        if role_cfg.reasoning_effort is not None:
            reasoning_effort = role_cfg.reasoning_effort
        if role_cfg.max_output_tokens is not None:
            max_output_tokens = role_cfg.max_output_tokens

    return ResolvedModel(
        role=role,
        model_key=model_key,
        api_model=spec.api_model,
        provider=spec.provider,
        supports_vision=spec.supports_vision,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        pricing=spec.pricing,
    )


def clear_registry_cache() -> None:
    """Clear the config cache (tests)."""
    _config_cache.clear()
