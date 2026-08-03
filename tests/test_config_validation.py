"""Tests for the startup config validator.

Before this file, ``validate_all_configs()`` had zero test coverage:
deleting its call from ``app/api/main.py`` left the whole suite green.
These tests guard both that the validator actually calls every loader
and that ``main.py``'s lifespan calls it before any expensive startup
work (loading the inference pipeline).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from adoption_accelerator import config_validation


@pytest.fixture(autouse=True)
def _clear_caches():
    from adoption_accelerator import target_labels
    from adoption_accelerator.agents import runtime_config
    from adoption_accelerator.agents.llm import registry

    yield
    target_labels.clear_target_cache()
    runtime_config.clear_runtime_config_cache()
    registry.clear_registry_cache()


def test_validate_all_configs_calls_each_loader_once():
    with patch.object(config_validation, "load_models_config") as mock_models, \
         patch.object(config_validation, "load_node_timeouts") as mock_timeouts, \
         patch.object(config_validation, "load_target_config") as mock_target:
        config_validation.validate_all_configs()

    mock_models.assert_called_once()
    mock_timeouts.assert_called_once()
    mock_target.assert_called_once()


@pytest.mark.parametrize(
    "loader_attr, path_attr",
    [
        ("load_models_config", "_MODELS_CONFIG_PATH"),
        ("load_node_timeouts", "_DEFAULT_TIMEOUTS_PATH"),
        ("load_target_config", "_DEFAULT_SERVING_PATH"),
    ],
)
def test_malformed_config_raises_named_runtime_error(loader_attr, path_attr):
    """A failure in any one loader must surface as a RuntimeError naming
    the offending config file, chained from the original exception, not
    a bare pydantic traceback.
    """
    original_error = ValueError("simulated malformed config")

    def failing_loader():
        raise original_error

    with patch.object(config_validation, loader_attr, failing_loader):
        with pytest.raises(RuntimeError) as exc_info:
            config_validation.validate_all_configs()

    expected_path = str(getattr(config_validation, path_attr))
    assert expected_path in str(exc_info.value)
    assert exc_info.value.__cause__ is original_error


def test_lifespan_validates_configs_before_loading_pipeline():
    """``lifespan`` must call ``validate_all_configs`` before
    ``get_inference_pipeline`` so a bad config fails before any expensive
    startup work runs. Every other side-effecting step in the lifespan
    body is patched out so this test touches neither the filesystem nor
    a real model bundle.
    """
    from app.api import main as main_module

    order: list[str] = []

    def fake_validate():
        order.append("validate_all_configs")

    def fake_get_pipeline():
        order.append("get_inference_pipeline")
        return object()

    app = SimpleNamespace(state=SimpleNamespace())

    async def _drive_lifespan():
        async with main_module.lifespan(app):
            pass

    with patch.object(
        main_module, "validate_all_configs", side_effect=fake_validate
    ) as mock_validate, patch.object(
        main_module, "get_inference_pipeline", side_effect=fake_get_pipeline
    ) as mock_pipeline, patch.object(
        main_module, "compile_report_graph", return_value=object()
    ), patch.object(
        main_module.session_storage, "prune_all", return_value=0
    ), patch.object(
        main_module, "_load_json", return_value={}
    ), patch.object(
        main_module, "_load_model_meta", return_value={}
    ), patch.object(
        main_module, "_compute_modality_breakdown", return_value={}
    ), patch.object(
        main_module.job_store, "start_cleanup_loop"
    ), patch.object(
        main_module.job_store, "stop_cleanup_loop"
    ):
        asyncio.run(_drive_lifespan())

    mock_validate.assert_called_once()
    mock_pipeline.assert_called_once()
    assert order == ["validate_all_configs", "get_inference_pipeline"]
