"""Shared fixtures for app.api router tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
