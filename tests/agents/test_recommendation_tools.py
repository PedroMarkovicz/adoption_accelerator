"""Tests for model-validated recommendation tools."""

import json

import pytest

from adoption_accelerator.agents.tools.recommendation_tools import (
    make_recommendation_tools,
)
from adoption_accelerator.contracts_test_helpers import make_request
from adoption_accelerator.inference.serving import get_inference_pipeline
from adoption_accelerator.inference.feature_builder import build_feature_vector


@pytest.fixture(scope="module")
def context():
    request = make_request(description="Friendly young dog.")
    pipeline = get_inference_pipeline()
    fv = build_feature_vector(request, pipeline.feature_schema.get("features", []))
    baseline = pipeline.predict_single(fv)
    return request, baseline


def _tool(tools, name):
    return next(t for t in tools if t.name == name)


def test_counterfactual_returns_measured_delta(context):
    request, baseline = context
    tools, log = make_recommendation_tools(request, baseline)
    result = json.loads(_tool(tools, "run_counterfactual").invoke(
        {"feature": "Fee", "value": "50"}
    ))
    assert "measurement_id" in result
    assert result["class_before"] == baseline.prediction
    assert set(result["probability_shift"].keys()) == {"0", "1", "2", "3", "4"}
    assert result["measurement_id"] in log.measurements


def test_non_actionable_feature_is_rejected(context):
    request, baseline = context
    tools, log = make_recommendation_tools(request, baseline)
    result = json.loads(_tool(tools, "run_counterfactual").invoke(
        {"feature": "Age", "value": "1"}
    ))
    assert result["error"].startswith("Feature 'Age' is not actionable")
    assert log.measurements == {}


def test_what_if_combines_changes(context):
    request, baseline = context
    tools, log = make_recommendation_tools(request, baseline)
    result = json.loads(_tool(tools, "run_what_if").invoke(
        {"changes_json": json.dumps({"Fee": "0", "Sterilized": "1"})}
    ))
    assert "measurement_id" in result
    assert log.measurements[result["measurement_id"]]["changes"] == {
        "Fee": "0", "Sterilized": "1"
    }


def test_malformed_numeric_value_returns_error(context):
    request, baseline = context
    tools, log = make_recommendation_tools(request, baseline)
    result = json.loads(_tool(tools, "run_counterfactual").invoke(
        {"feature": "Vaccinated", "value": "yes"}
    ))
    assert "error" in result
    assert log.measurements == {}


def test_lookup_feature_returns_metadata(context):
    request, baseline = context
    tools, _ = make_recommendation_tools(request, baseline)
    result = json.loads(_tool(tools, "lookup_feature").invoke({"feature": "Fee"}))
    assert result["actionable"] is True
    assert result["current_value"] == "0.0"
