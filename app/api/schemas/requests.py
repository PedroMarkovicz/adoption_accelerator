"""
Request schemas for the Adoption Accelerator API.

PetProfileRequest is the frontend-friendly input contract.  It uses
human-readable string values (e.g. "Dog", "Yes", "Healthy") which the
prediction service translates into the integer codes expected by the
backend TabularInput.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PetProfileRequest(BaseModel):
    # Basic Info
    pet_type: Literal["Dog", "Cat"]
    name: str = ""
    age_months: int = Field(..., ge=0, le=255)
    gender: Literal["Male", "Female", "Mixed"]

    # Appearance
    breed1: int = Field(..., description="Primary breed ID")
    breed2: int = Field(0, description="Secondary breed ID, 0 = none")
    color1: int = Field(0, description="Primary color ID, 0 = unspecified")
    color2: int = Field(0, description="Secondary color ID, 0 = none")
    color3: int = Field(0, description="Tertiary color ID, 0 = none")
    maturity_size: Literal[1, 2, 3, 4]
    fur_length: Literal[1, 2, 3]

    # Health
    vaccinated: Literal["Yes", "No", "Not Sure"]
    dewormed: Literal["Yes", "No", "Not Sure"]
    sterilized: Literal["Yes", "No", "Not Sure"]
    health: Literal["Healthy", "Minor Injury", "Serious Injury"]

    # Listing Details
    fee: float = Field(0.0, ge=0)
    quantity: int = Field(1, ge=1)
    state: int = Field(0, description="Malaysian state ID, 0 = unspecified")
    video_amt: int = Field(0, ge=0)

    # Content
    description: str = ""
