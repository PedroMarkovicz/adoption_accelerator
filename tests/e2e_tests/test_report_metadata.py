"""Tests for report metadata fields assembled by the aggregator."""

from adoption_accelerator.agents.contracts import ReportMetadata


def test_report_metadata_defaults_image_count_to_zero():
    meta = ReportMetadata(session_id="s")
    assert meta.image_count == 0


def test_report_metadata_accepts_image_count():
    meta = ReportMetadata(session_id="s", image_count=3)
    assert meta.image_count == 3


def test_aggregator_reports_image_count_from_request():
    from adoption_accelerator.agents.contracts import PredictionEvidence
    from adoption_accelerator.agents.nodes.aggregator import aggregator_node
    from adoption_accelerator.inference.contracts import PredictionRequest, TabularInput

    tabular = TabularInput(
        type=1, name=None, age=12, breed1=307, breed2=None, gender=1,
        color1=1, color2=None, color3=None, maturity_size=2, fur_length=1,
        vaccinated=1, dewormed=1, sterilized=2, health=1, quantity=1,
        fee=0.0, state=41326, video_amt=0,
    )
    request = PredictionRequest(
        tabular=tabular, description="", images=["a.jpg", "b.jpg"]
    )
    # A real PredictionEvidence is required: aggregator_node short-circuits and
    # returns report=None when prediction_evidence is missing, which would make
    # the assertion below unreachable.
    prediction = PredictionEvidence(
        source="ensemble",
        confidence="high",
        generated_by="deterministic",
        notes=[],
        predicted_class=1,
        prediction_label="Adopted within 1 week",
        probabilities={0: 0.1, 1: 0.6, 2: 0.2, 3: 0.05, 4: 0.05},
        class_confidence=0.6,
        modality_contributions={"tabular": 1.0},
        modality_available={"tabular": True},
        key_drivers=[],
        uncertainty_reading="",
    )
    state = {
        "request": request,
        "session_id": "s",
        "prediction_evidence": prediction,
        "errors": [],
        "trace": [],
        "timestamp": "",
    }

    result = aggregator_node(state)

    assert result["report"].metadata.image_count == 2
