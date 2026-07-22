"""Shared test fixture builders (used by tests only)."""

from adoption_accelerator.inference.contracts import PredictionRequest, TabularInput


def make_request(images: list[str] | None = None,
                 description: str = "") -> PredictionRequest:
    return PredictionRequest(
        tabular=TabularInput(
            type=1, name="Rex", age=6, breed1=307, gender=1, color1=1,
            maturity_size=2, fur_length=1, vaccinated=1, dewormed=1,
            sterilized=2, health=1, quantity=1, fee=0.0, state=41326,
        ),
        description=description,
        images=images or [],
    )
