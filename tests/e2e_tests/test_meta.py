"""Tests for the /meta reference-data service and endpoint."""

from app.api.services import meta_service


def test_adoption_speed_classes_has_five_ordered_entries():
    classes = meta_service.adoption_speed_classes()
    assert [c["index"] for c in classes] == [0, 1, 2, 3, 4]
    assert classes[0]["label"] == "Same-day adoption"
    assert classes[4]["label"] == "Not adopted (100+ days)"


def test_load_breeds_includes_type_and_name():
    breeds = meta_service.load_breeds()
    assert len(breeds) > 100
    first = breeds[0]
    assert set(first.keys()) == {"id", "type", "name"}
    assert first["type"] in (1, 2)


def test_load_colors_and_states_nonempty():
    assert len(meta_service.load_colors()) >= 5
    states = meta_service.load_states()
    assert any(s["name"] == "Johor" for s in states)


def test_maturity_and_fur_labels():
    assert meta_service.maturity_sizes() == [
        {"id": 1, "label": "Small"},
        {"id": 2, "label": "Medium"},
        {"id": 3, "label": "Large"},
        {"id": 4, "label": "Extra Large"},
    ]
    assert meta_service.fur_lengths() == [
        {"id": 1, "label": "Short"},
        {"id": 2, "label": "Medium"},
        {"id": 3, "label": "Long"},
    ]


def test_build_meta_shape():
    payload = meta_service.build_meta("tuned_v1", {"tabular": 10, "text": 5})
    assert payload["model_version"] == "tuned_v1"
    assert payload["modality_breakdown"] == {"tabular": 10, "text": 5}
    assert len(payload["adoption_speed_classes"]) == 5
    assert payload["breeds"] and payload["colors"] and payload["states"]


from fastapi.testclient import TestClient

from app.api.main import app


def test_meta_endpoint_returns_200_and_shape():
    with TestClient(app) as client:
        resp = client.get("/meta")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["adoption_speed_classes"]) == 5
    assert body["breeds"] and body["colors"] and body["states"]
    assert body["maturity_sizes"][0] == {"id": 1, "label": "Small"}
