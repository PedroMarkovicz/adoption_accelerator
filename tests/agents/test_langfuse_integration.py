"""Langfuse handler wiring tests (no network)."""

import pytest

from adoption_accelerator.agents.observability.langfuse import (
    get_langfuse_handler,
)
from adoption_accelerator.agents.graph import get_graph_config


def test_no_keys_returns_none(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert get_langfuse_handler("s1") is None


def test_graph_config_empty_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert get_graph_config("s1") == {}
