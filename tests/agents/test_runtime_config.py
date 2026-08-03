"""Tests for the node runtime configuration loader.

The regression these guard: configs/agents/timeouts.yaml existed for
months while every node hardcoded its own timeout, so editing the file
had no effect and gave no sign of having no effect.
"""

from __future__ import annotations

import textwrap

import pytest
import yaml

from adoption_accelerator.agents import runtime_config


@pytest.fixture(autouse=True)
def _clear_cache():
    runtime_config.clear_runtime_config_cache()
    yield
    runtime_config.clear_runtime_config_cache()


def test_shipped_config_covers_every_graph_node():
    cfg = runtime_config.load_node_timeouts()
    assert set(cfg.node_timeouts) == set(runtime_config.GRAPH_NODES)


def test_every_timeout_is_positive():
    cfg = runtime_config.load_node_timeouts()
    for node, seconds in cfg.node_timeouts.items():
        assert seconds > 0, f"{node} has a non-positive timeout"


def test_editing_the_yaml_changes_the_effective_timeout(tmp_path):
    """The test that would have caught the dead config."""
    payload = {"node_timeouts": {node: 3 for node in runtime_config.GRAPH_NODES}}
    path = tmp_path / "timeouts.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    cfg = runtime_config.load_node_timeouts(path)
    assert cfg.node_timeouts["synthesizer"] == 3.0


def test_missing_node_is_rejected(tmp_path):
    payload = {"node_timeouts": {"orchestrator": 5}}
    path = tmp_path / "timeouts.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing timeout"):
        runtime_config.load_node_timeouts(path)


def test_unknown_node_is_rejected(tmp_path):
    payload = {"node_timeouts": {node: 5 for node in runtime_config.GRAPH_NODES}}
    payload["node_timeouts"]["not_a_node"] = 5
    path = tmp_path / "timeouts.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown node"):
        runtime_config.load_node_timeouts(path)


def test_node_timeout_rejects_unknown_node():
    with pytest.raises(KeyError):
        runtime_config.node_timeout("not_a_node")
