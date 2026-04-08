"""
Pydantic contracts for the inference pipeline.

Define the formal interface between the inference pipeline and all
external consumers (frontend, agents).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TabularInput(BaseModel):
    """Tabular features for a pet listing."""

    type: int = Field(..., description="1 = Dog, 2 = Cat")
    name: Optional[str] = Field(None, description="Pet name")
    age: int = Field(..., description="Age in months")
    breed1: int = Field(..., description="Primary breed ID")
    breed2: Optional[int] = Field(None, description="Secondary breed ID")
    gender: int = Field(..., description="1=Male, 2=Female, 3=Mixed")
    color1: int = Field(..., description="Primary color ID")
    color2: Optional[int] = Field(None, description="Secondary color ID")
    color3: Optional[int] = Field(None, description="Tertiary color ID")
    maturity_size: int = Field(..., description="1-4 size category")
    fur_length: int = Field(..., description="1-3 fur length")
    vaccinated: int = Field(..., description="1=Yes, 2=No, 3=Not Sure")
    dewormed: int = Field(..., description="1=Yes, 2=No, 3=Not Sure")
    sterilized: int = Field(..., description="1=Yes, 2=No, 3=Not Sure")
    health: int = Field(..., description="1=Healthy, 2=Minor, 3=Serious")
    quantity: int = Field(..., description="Number of pets in listing")
    fee: float = Field(..., description="Adoption fee")
    state: int = Field(..., description="Malaysian state ID")
    video_amt: int = Field(0, description="Number of videos")


class PredictionOptions(BaseModel):
    """Options for prediction output."""

    include_explanation: bool = False
    include_counterfactuals: bool = False


class PredictionRequest(BaseModel):
    """Input contract for a single prediction request."""

    tabular: TabularInput
    description: str = ""
    images: list[str] = Field(default_factory=list, description="Image file paths")
    options: PredictionOptions = Field(default_factory=PredictionOptions)


class PredictionResult(BaseModel):
    """Output contract for a prediction response."""

    prediction: int = Field(..., description="Predicted AdoptionSpeed 0-4")
    prediction_label: str = Field(..., description="Human-readable label")
    probabilities: dict[int, float] = Field(..., description="Per-class probabilities")

    # Issue #1 fix: confidence now reflects the thresholded predicted class
    # probability, not the global argmax. These two can differ when threshold
    # optimization selects a class other than the argmax class.
    confidence: float = Field(
        ...,
        description=(
            "Probability of the thresholded predicted class. "
            "Semantically: P(predicted_class). "
            "Use max_class_probability for the global argmax probability."
        ),
    )
    predicted_class_probability: float = Field(
        0.0,
        description="Explicit alias for confidence: P(predicted_class).",
    )
    max_class_probability: float = Field(
        0.0,
        description=(
            "Probability of the class with the highest raw probability "
            "(global argmax). May differ from confidence when threshold "
            "optimization selects a different class."
        ),
    )
    max_class: int = Field(
        0,
        description="Class index that holds max_class_probability.",
    )

    explanation: Optional[dict] = None
    metadata: dict = Field(default_factory=dict)
