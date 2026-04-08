"""
Unit tests: API schema serialization against the 10 e2e output JSONs.

Each test loads a pre-built output JSON from tests/e2e_tests/outputs/,
reconstructs an AgentResponse from its ``final_response`` section, calls
translate_response() and asserts the result is a valid PredictionStatusResponse
with zero Pydantic validation errors.

This is the validation gate for Phase 1 (step 4.1.6): all 10 e2e scenarios
must produce a valid PredictionStatusResponse through the translation layer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agents.state import AgentResponse
from app.api.schemas.enums import AdoptionSpeedClass
from app.api.schemas.requests import PetProfileRequest
from app.api.schemas.responses import (
    FeatureFactorOut,
    Phase1Response,
    Phase2Response,
    PredictionStatusResponse,
    RecommendationOut,
    ResponseMetadataOut,
)
from app.api.services.prediction_service import translate_request, translate_response

OUTPUTS_DIR = Path(__file__).parent / "outputs"

SCENARIO_FILES = [
    "typical_dog_with_description.json",
    "typical_cat_no_description.json",
    "fully_documented_pet.json",
    "minimal_input.json",
    "high_fee_unvaccinated.json",
    "young_puppy_healthy.json",
    "senior_cat_health_issues.json",
    "multiple_pets_listing.json",
    "whitespace_only_description.json",
    "very_long_description.json",
]


def _load_agent_response(scenario_file: str) -> AgentResponse:
    """Load the final_response section of an e2e output JSON as an AgentResponse."""
    path = OUTPUTS_DIR / scenario_file
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return AgentResponse.model_validate(data["final_response"])


def _make_mock_state(agent_response: AgentResponse) -> dict:
    """Build a minimal mock AgentState dict for translate_response."""
    return {"response": agent_response}


# ---------------------------------------------------------------------------
# AdoptionSpeedClass enum tests
# ---------------------------------------------------------------------------


class TestAdoptionSpeedClassEnum:
    def test_all_classes_have_labels(self):
        for cls in AdoptionSpeedClass:
            assert isinstance(cls.label, str)
            assert len(cls.label) > 0

    def test_labels_match_pipeline_class_labels(self):
        from adoption_accelerator.inference.pipeline import CLASS_LABELS

        for cls in AdoptionSpeedClass:
            assert cls.label == CLASS_LABELS[cls.value]

    def test_class_values(self):
        assert AdoptionSpeedClass.SAME_DAY == 0
        assert AdoptionSpeedClass.WITHIN_1_WEEK == 1
        assert AdoptionSpeedClass.WITHIN_1_MONTH == 2
        assert AdoptionSpeedClass.WITHIN_1_TO_3_MONTHS == 3
        assert AdoptionSpeedClass.NOT_ADOPTED == 4


# ---------------------------------------------------------------------------
# PetProfileRequest validation tests
# ---------------------------------------------------------------------------


class TestPetProfileRequest:
    def _base_request(self) -> dict:
        return {
            "pet_type": "Dog",
            "name": "Buddy",
            "age_months": 3,
            "gender": "Male",
            "breed1": 265,
            "maturity_size": 2,
            "fur_length": 2,
            "vaccinated": "Yes",
            "dewormed": "Yes",
            "sterilized": "No",
            "health": "Healthy",
        }

    def test_valid_dog_request(self):
        req = PetProfileRequest(**self._base_request())
        assert req.pet_type == "Dog"
        assert req.age_months == 3

    def test_valid_cat_request(self):
        data = self._base_request()
        data["pet_type"] = "Cat"
        req = PetProfileRequest(**data)
        assert req.pet_type == "Cat"

    def test_defaults(self):
        req = PetProfileRequest(**self._base_request())
        assert req.breed2 == 0
        assert req.color1 == 0
        assert req.color2 == 0
        assert req.color3 == 0
        assert req.fee == 0.0
        assert req.quantity == 1
        assert req.state == 0
        assert req.video_amt == 0
        assert req.description == ""

    def test_invalid_pet_type(self):
        with pytest.raises(Exception):
            PetProfileRequest(**{**self._base_request(), "pet_type": "Fish"})

    def test_invalid_vaccinated(self):
        with pytest.raises(Exception):
            PetProfileRequest(**{**self._base_request(), "vaccinated": "Maybe"})

    def test_age_out_of_range(self):
        with pytest.raises(Exception):
            PetProfileRequest(**{**self._base_request(), "age_months": 300})

    def test_negative_fee(self):
        with pytest.raises(Exception):
            PetProfileRequest(**{**self._base_request(), "fee": -10.0})


# ---------------------------------------------------------------------------
# translate_request tests
# ---------------------------------------------------------------------------


class TestTranslateRequest:
    def _make_pet(self, **overrides) -> PetProfileRequest:
        base = {
            "pet_type": "Dog",
            "age_months": 3,
            "gender": "Male",
            "breed1": 265,
            "maturity_size": 2,
            "fur_length": 2,
            "vaccinated": "Yes",
            "dewormed": "No",
            "sterilized": "Not Sure",
            "health": "Healthy",
        }
        base.update(overrides)
        return PetProfileRequest(**base)

    def test_dog_maps_to_1(self):
        req = translate_request(self._make_pet(pet_type="Dog"))
        assert req.tabular.type == 1

    def test_cat_maps_to_2(self):
        req = translate_request(self._make_pet(pet_type="Cat"))
        assert req.tabular.type == 2

    def test_gender_mapping(self):
        assert translate_request(self._make_pet(gender="Male")).tabular.gender == 1
        assert translate_request(self._make_pet(gender="Female")).tabular.gender == 2
        assert translate_request(self._make_pet(gender="Mixed")).tabular.gender == 3

    def test_tristate_mapping(self):
        req_yes = translate_request(self._make_pet(vaccinated="Yes"))
        req_no = translate_request(self._make_pet(vaccinated="No"))
        req_ns = translate_request(self._make_pet(vaccinated="Not Sure"))
        assert req_yes.tabular.vaccinated == 1
        assert req_no.tabular.vaccinated == 2
        assert req_ns.tabular.vaccinated == 3

    def test_health_mapping(self):
        assert translate_request(self._make_pet(health="Healthy")).tabular.health == 1
        assert translate_request(self._make_pet(health="Minor Injury")).tabular.health == 2
        assert translate_request(self._make_pet(health="Serious Injury")).tabular.health == 3

    def test_description_passed_through(self):
        req = translate_request(self._make_pet(description="A lovely dog."))
        assert req.description == "A lovely dog."

    def test_zero_breed2_becomes_none(self):
        req = translate_request(self._make_pet(breed2=0))
        assert req.tabular.breed2 is None

    def test_nonzero_breed2_preserved(self):
        req = translate_request(self._make_pet(breed2=264))
        assert req.tabular.breed2 == 264

    def test_empty_name_becomes_none(self):
        req = translate_request(self._make_pet(name=""))
        assert req.tabular.name is None

    def test_nonempty_name_preserved(self):
        req = translate_request(self._make_pet(name="Buddy"))
        assert req.tabular.name == "Buddy"


# ---------------------------------------------------------------------------
# translate_response tests (schema serialization against e2e outputs)
# ---------------------------------------------------------------------------


class TestTranslateResponse:
    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_response_is_valid_pydantic(self, scenario_file: str):
        """translate_response must produce a valid PredictionStatusResponse
        for each of the 10 e2e scenarios with zero Pydantic validation errors."""
        agent_response = _load_agent_response(scenario_file)
        state = _make_mock_state(agent_response)

        result = translate_response(state)

        assert isinstance(result, PredictionStatusResponse)

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_status_is_complete(self, scenario_file: str):
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        assert result.status == "complete"

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_session_id_is_non_empty(self, scenario_file: str):
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        assert result.session_id != ""

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_phase1_is_populated(self, scenario_file: str):
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        assert result.phase1 is not None
        assert isinstance(result.phase1, Phase1Response)

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_prediction_in_valid_range(self, scenario_file: str):
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        assert 0 <= result.phase1.prediction <= 4

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_probabilities_have_string_keys(self, scenario_file: str):
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        assert all(isinstance(k, str) for k in result.phase1.probabilities)
        assert set(result.phase1.probabilities.keys()) == {"0", "1", "2", "3", "4"}

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_probabilities_sum_to_one(self, scenario_file: str):
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        total = sum(result.phase1.probabilities.values())
        assert abs(total - 1.0) < 1e-3

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_confidence_in_valid_range(self, scenario_file: str):
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        assert 0.0 < result.phase1.confidence <= 1.0

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_phase2_is_populated(self, scenario_file: str):
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        assert result.phase2 is not None
        assert isinstance(result.phase2, Phase2Response)

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_metadata_is_populated(self, scenario_file: str):
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        assert result.metadata is not None
        assert isinstance(result.metadata, ResponseMetadataOut)
        assert result.metadata.model_version == "tuned_v1"
        assert result.metadata.model_type == "SoftVotingEnsemble"
        assert result.metadata.inference_time_ms > 0

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_feature_factors_have_valid_modality(self, scenario_file: str):
        valid_modalities = {"tabular", "text", "image", "metadata"}
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        all_factors = result.phase1.top_positive_factors + result.phase1.top_negative_factors
        for factor in all_factors:
            assert factor.modality in valid_modalities
            assert factor.direction in {"positive", "negative"}

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_recommendations_have_valid_category(self, scenario_file: str):
        valid_categories = {"photo", "description", "health", "listing_details"}
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        for rec in result.phase2.recommendations:
            assert rec.category in valid_categories

    @pytest.mark.parametrize("scenario_file", SCENARIO_FILES)
    def test_response_roundtrips_json(self, scenario_file: str):
        """PredictionStatusResponse must be JSON-serializable without errors."""
        agent_response = _load_agent_response(scenario_file)
        result = translate_response(_make_mock_state(agent_response))
        json_str = result.model_dump_json()
        assert len(json_str) > 0
        reparsed = PredictionStatusResponse.model_validate_json(json_str)
        assert reparsed.session_id == result.session_id
        assert reparsed.phase1.prediction == result.phase1.prediction


# ---------------------------------------------------------------------------
# Validation gate: all 10 scenarios pass
# ---------------------------------------------------------------------------


class TestValidationGate:
    """Step 4.1.6: All 10 e2e scenarios must produce a valid
    PredictionStatusResponse with zero Pydantic validation errors."""

    def test_all_10_scenarios_produce_valid_response(self):
        passed = []
        failed = []

        for scenario_file in SCENARIO_FILES:
            try:
                agent_response = _load_agent_response(scenario_file)
                state = _make_mock_state(agent_response)
                result = translate_response(state)

                # Core validation checks
                assert isinstance(result, PredictionStatusResponse)
                assert result.status == "complete"
                assert result.phase1 is not None
                assert result.phase2 is not None
                assert result.metadata is not None
                assert 0 <= result.phase1.prediction <= 4
                assert set(result.phase1.probabilities.keys()) == {"0", "1", "2", "3", "4"}
                assert abs(sum(result.phase1.probabilities.values()) - 1.0) < 1e-3

                # JSON round-trip
                result.model_dump_json()

                passed.append(scenario_file)
            except Exception as exc:
                failed.append((scenario_file, str(exc)))

        assert len(failed) == 0, (
            f"Validation gate FAILED for {len(failed)}/10 scenarios:\n"
            + "\n".join(f"  {name}: {err}" for name, err in failed)
        )
        assert len(passed) == 10, f"Expected 10 passing scenarios, got {len(passed)}"
