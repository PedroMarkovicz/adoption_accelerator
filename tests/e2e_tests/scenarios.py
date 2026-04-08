"""
Test scenario definitions for E2E testing.

Each scenario defines a named input configuration with a description
of what it exercises and the expected behavior of the system.
Scenarios cover typical inputs, edge cases, minimal inputs, and
ambiguous or unusual configurations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adoption_accelerator.inference.contracts import (
    PredictionRequest,
    TabularInput,
)


@dataclass
class TestScenario:
    """A single E2E test scenario."""

    name: str
    description: str
    request: PredictionRequest
    expected_behavior: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)


def build_all_scenarios() -> list[TestScenario]:
    """Build and return all E2E test scenarios."""
    return [
        _scenario_typical_dog_with_description(),
        _scenario_typical_cat_no_description(),
        _scenario_fully_documented_pet(),
        _scenario_minimal_input(),
        _scenario_high_fee_unvaccinated(),
        _scenario_young_puppy_healthy(),
        _scenario_senior_cat_health_issues(),
        _scenario_multiple_pets_listing(),
        _scenario_whitespace_only_description(),
        _scenario_very_long_description(),
    ]


# -----------------------------------------------------------------------
# Scenario definitions
# -----------------------------------------------------------------------


def _scenario_typical_dog_with_description() -> TestScenario:
    """Scenario 1: Typical dog listing with a standard description."""
    return TestScenario(
        name="typical_dog_with_description",
        description=(
            "A typical dog listing with all standard fields populated "
            "and a moderate-length description. Exercises the full "
            "pipeline including description_writer."
        ),
        request=PredictionRequest(
            tabular=TabularInput(
                type=1,
                name="Buddy",
                age=3,
                breed1=265,
                breed2=0,
                gender=1,
                color1=1,
                color2=0,
                color3=0,
                maturity_size=2,
                fur_length=2,
                vaccinated=1,
                dewormed=1,
                sterilized=2,
                health=1,
                quantity=1,
                fee=0,
                state=41326,
                video_amt=0,
            ),
            description=(
                "Buddy is a friendly and playful dog who loves to run "
                "around the yard. He is great with children and other "
                "pets. He has been vaccinated and dewormed. Looking for "
                "a loving forever home where he can be part of the family."
            ),
            images=[],
        ),
        expected_behavior={
            "description_writer_runs": True,
            "pet_type_label": "Dog",
            "has_name": True,
        },
        tags=["typical", "dog", "with_description"],
    )


def _scenario_typical_cat_no_description() -> TestScenario:
    """Scenario 2: Typical cat listing without a description."""
    return TestScenario(
        name="typical_cat_no_description",
        description=(
            "A cat listing with standard fields but no description. "
            "The description_writer node should be skipped via "
            "conditional routing."
        ),
        request=PredictionRequest(
            tabular=TabularInput(
                type=2,
                name="Whiskers",
                age=6,
                breed1=300,
                breed2=0,
                gender=2,
                color1=2,
                color2=5,
                color3=0,
                maturity_size=1,
                fur_length=1,
                vaccinated=1,
                dewormed=1,
                sterilized=1,
                health=1,
                quantity=1,
                fee=50,
                state=41326,
                video_amt=0,
            ),
            description="",
            images=[],
        ),
        expected_behavior={
            "description_writer_runs": False,
            "pet_type_label": "Cat",
            "has_name": True,
        },
        tags=["typical", "cat", "no_description"],
    )


def _scenario_fully_documented_pet() -> TestScenario:
    """Scenario 3: A pet with all care completed and rich listing details."""
    return TestScenario(
        name="fully_documented_pet",
        description=(
            "A well-documented pet with vaccinations, deworming, and "
            "sterilization completed, a low fee, multiple videos, and "
            "a detailed description. Should receive a favorable prediction."
        ),
        request=PredictionRequest(
            tabular=TabularInput(
                type=1,
                name="Max",
                age=12,
                breed1=307,
                breed2=264,
                gender=1,
                color1=1,
                color2=2,
                color3=0,
                maturity_size=3,
                fur_length=2,
                vaccinated=1,
                dewormed=1,
                sterilized=1,
                health=1,
                quantity=1,
                fee=0,
                state=41401,
                video_amt=2,
            ),
            description=(
                "Max is a gentle and loyal mixed-breed dog who has been "
                "our faithful companion for over a year. He is fully "
                "vaccinated, dewormed, and sterilized. Max is house-trained, "
                "knows basic commands, and is excellent with children of all "
                "ages. He enjoys daily walks and loves to play fetch. Max "
                "has a calm temperament and adapts well to new environments. "
                "He would thrive in a family with a backyard but is also "
                "comfortable in apartment living. Max comes with all his "
                "medical records and favorite toys."
            ),
            images=[],
        ),
        expected_behavior={
            "description_writer_runs": True,
            "pet_type_label": "Dog",
            "all_care_completed": True,
        },
        tags=["typical", "dog", "fully_documented"],
    )


def _scenario_minimal_input() -> TestScenario:
    """Scenario 4: Minimal required fields only, no optional data."""
    return TestScenario(
        name="minimal_input",
        description=(
            "The bare minimum required input: only mandatory tabular "
            "fields. No name, no description, no images, no videos. "
            "Tests system robustness with sparse data."
        ),
        request=PredictionRequest(
            tabular=TabularInput(
                type=2,
                name="",
                age=1,
                breed1=0,
                breed2=0,
                gender=1,
                color1=1,
                color2=0,
                color3=0,
                maturity_size=2,
                fur_length=2,
                vaccinated=3,
                dewormed=3,
                sterilized=3,
                health=1,
                quantity=1,
                fee=0,
                state=41326,
                video_amt=0,
            ),
            description="",
            images=[],
        ),
        expected_behavior={
            "description_writer_runs": False,
            "pet_type_label": "Cat",
            "has_name": False,
            "sparse_input": True,
        },
        tags=["edge_case", "minimal"],
    )


def _scenario_high_fee_unvaccinated() -> TestScenario:
    """Scenario 5: High fee and no medical care -- negative signals."""
    return TestScenario(
        name="high_fee_unvaccinated",
        description=(
            "A pet with a high adoption fee and no medical care "
            "documentation (not vaccinated, not dewormed, not sterilized). "
            "These are strong negative SHAP signals. The recommender "
            "should suggest actionable improvements."
        ),
        request=PredictionRequest(
            tabular=TabularInput(
                type=1,
                name="Rex",
                age=24,
                breed1=265,
                breed2=0,
                gender=1,
                color1=3,
                color2=0,
                color3=0,
                maturity_size=4,
                fur_length=3,
                vaccinated=2,
                dewormed=2,
                sterilized=2,
                health=1,
                quantity=1,
                fee=500,
                state=41326,
                video_amt=0,
            ),
            description=(
                "Rex is a large adult dog looking for a new home. "
                "He is healthy and energetic."
            ),
            images=[],
        ),
        expected_behavior={
            "description_writer_runs": True,
            "pet_type_label": "Dog",
            "likely_slow_adoption": True,
            "recommendations_should_address": ["vaccination", "fee"],
        },
        tags=["edge_case", "negative_signals"],
    )


def _scenario_young_puppy_healthy() -> TestScenario:
    """Scenario 6: Very young puppy with good health -- positive signals."""
    return TestScenario(
        name="young_puppy_healthy",
        description=(
            "A very young, healthy puppy with free adoption and full "
            "medical care. Should receive a favorable (fast adoption) "
            "prediction."
        ),
        request=PredictionRequest(
            tabular=TabularInput(
                type=1,
                name="Luna",
                age=1,
                breed1=265,
                breed2=0,
                gender=2,
                color1=1,
                color2=7,
                color3=0,
                maturity_size=2,
                fur_length=1,
                vaccinated=1,
                dewormed=1,
                sterilized=1,
                health=1,
                quantity=1,
                fee=0,
                state=41401,
                video_amt=1,
            ),
            description=(
                "Luna is an adorable puppy who was rescued from the "
                "streets. Despite her rough start, she is incredibly "
                "sweet, playful, and loves belly rubs. Luna has been "
                "fully vaccinated, dewormed, and sterilized. She is "
                "looking for a patient family who will shower her with "
                "love and give her the happy life she deserves."
            ),
            images=[],
        ),
        expected_behavior={
            "description_writer_runs": True,
            "pet_type_label": "Dog",
            "likely_fast_adoption": True,
        },
        tags=["typical", "dog", "positive_signals"],
    )


def _scenario_senior_cat_health_issues() -> TestScenario:
    """Scenario 7: Senior cat with a minor health issue."""
    return TestScenario(
        name="senior_cat_health_issues",
        description=(
            "An older cat with a minor health condition and high fee. "
            "Tests how the system handles pets with health challenges "
            "and whether recommendations are appropriate."
        ),
        request=PredictionRequest(
            tabular=TabularInput(
                type=2,
                name="Grandpa",
                age=120,
                breed1=300,
                breed2=0,
                gender=1,
                color1=6,
                color2=0,
                color3=0,
                maturity_size=2,
                fur_length=2,
                vaccinated=1,
                dewormed=1,
                sterilized=1,
                health=2,
                quantity=1,
                fee=200,
                state=41326,
                video_amt=0,
            ),
            description=(
                "Grandpa is a gentle senior cat who has been a beloved "
                "companion for many years. He has a minor leg injury "
                "that is being treated. He is calm, affectionate, and "
                "loves to sit on laps. Perfect for a quiet household."
            ),
            images=[],
        ),
        expected_behavior={
            "description_writer_runs": True,
            "pet_type_label": "Cat",
            "has_health_issues": True,
        },
        tags=["edge_case", "cat", "health_issues", "senior"],
    )


def _scenario_multiple_pets_listing() -> TestScenario:
    """Scenario 8: A listing for multiple pets at once."""
    return TestScenario(
        name="multiple_pets_listing",
        description=(
            "A listing for 3 kittens bundled together. Tests how the "
            "system handles quantity > 1."
        ),
        request=PredictionRequest(
            tabular=TabularInput(
                type=2,
                name="The Three Amigos",
                age=2,
                breed1=300,
                breed2=0,
                gender=3,
                color1=1,
                color2=2,
                color3=7,
                maturity_size=1,
                fur_length=1,
                vaccinated=2,
                dewormed=2,
                sterilized=2,
                health=1,
                quantity=3,
                fee=0,
                state=41401,
                video_amt=0,
            ),
            description=(
                "Three adorable kittens looking for homes together or "
                "separately. They are playful, curious, and love each "
                "other's company."
            ),
            images=[],
        ),
        expected_behavior={
            "description_writer_runs": True,
            "pet_type_label": "Cat",
            "quantity_gt_1": True,
        },
        tags=["edge_case", "cat", "multiple_pets"],
    )


def _scenario_whitespace_only_description() -> TestScenario:
    """Scenario 9: Description with only whitespace characters."""
    return TestScenario(
        name="whitespace_only_description",
        description=(
            "A listing where the description field contains only "
            "whitespace. The description_writer should be skipped "
            "as if the description were empty."
        ),
        request=PredictionRequest(
            tabular=TabularInput(
                type=1,
                name="Shadow",
                age=8,
                breed1=265,
                breed2=0,
                gender=1,
                color1=1,
                color2=0,
                color3=0,
                maturity_size=3,
                fur_length=2,
                vaccinated=1,
                dewormed=1,
                sterilized=2,
                health=1,
                quantity=1,
                fee=30,
                state=41326,
                video_amt=0,
            ),
            description="    \n\t   \n  ",
            images=[],
        ),
        expected_behavior={
            "description_writer_runs": False,
            "pet_type_label": "Dog",
            "whitespace_description": True,
        },
        tags=["edge_case", "whitespace"],
    )


def _scenario_very_long_description() -> TestScenario:
    """Scenario 10: An unusually long description."""
    return TestScenario(
        name="very_long_description",
        description=(
            "A listing with an extremely long description (500+ words). "
            "Tests how text feature extraction and the description "
            "writer handle large text inputs."
        ),
        request=PredictionRequest(
            tabular=TabularInput(
                type=2,
                name="Cleopatra",
                age=36,
                breed1=300,
                breed2=0,
                gender=2,
                color1=1,
                color2=6,
                color3=0,
                maturity_size=2,
                fur_length=3,
                vaccinated=1,
                dewormed=1,
                sterilized=1,
                health=1,
                quantity=1,
                fee=0,
                state=41326,
                video_amt=1,
            ),
            description=(
                "Cleopatra is a majestic long-haired cat with a regal "
                "bearing and a heart of gold. She was found wandering "
                "near a temple and immediately captivated everyone with "
                "her stunning amber eyes and silky coat. Cleopatra is "
                "approximately three years old and in perfect health. "
                "She has been fully vaccinated, dewormed, and sterilized. "
                "Despite her royal appearance, Cleopatra is surprisingly "
                "down-to-earth and affectionate. She loves being brushed "
                "and will purr contentedly for hours while you groom her "
                "luxurious fur. She enjoys sunbathing by the window and "
                "watching birds from her favorite perch. Cleopatra is "
                "litter-trained and has impeccable manners. She gets "
                "along well with other cats but prefers to be the queen "
                "of her domain. She is somewhat reserved around dogs "
                "but tolerates them with dignified patience. Cleopatra "
                "would be ideal for a calm household where she can be "
                "pampered and adored. She does not do well with very "
                "young children who might handle her roughly, but she "
                "is wonderful with older children and adults. Her "
                "favorite activities include chasing feather toys, "
                "kneading soft blankets, and supervising her humans "
                "from a high vantage point. She has a melodic meow that "
                "she uses sparingly, usually to request dinner or "
                "attention. Cleopatra comes with all her medical records, "
                "a carrier, her favorite blanket, and a month's supply "
                "of premium cat food. The adoption fee is waived as we "
                "simply want to find her the perfect forever home where "
                "she will be cherished and treated like the royalty she is."
            ),
            images=[],
        ),
        expected_behavior={
            "description_writer_runs": True,
            "pet_type_label": "Cat",
            "long_description": True,
        },
        tags=["edge_case", "cat", "long_description"],
    )
