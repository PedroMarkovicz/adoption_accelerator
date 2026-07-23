"""Tests for per-session uploaded-image storage."""

import uuid

from fastapi.testclient import TestClient

from app.api.main import app
from app.api.services import session_storage


def _sid() -> str:
    return str(uuid.uuid4())


def test_save_image_writes_indexed_file():
    sid = _sid()
    try:
        path = session_storage.save_image(sid, 0, "my photo.JPG", b"fake-bytes")
        assert path.name == "0.jpg"
        assert path.read_bytes() == b"fake-bytes"
        assert path.parent == session_storage.images_dir(sid)
    finally:
        session_storage.delete_session(sid)


def test_save_image_ignores_user_filename_for_path():
    sid = _sid()
    try:
        path = session_storage.save_image(sid, 1, "../../evil.png", b"x")
        assert path.name == "1.png"
        assert path.parent == session_storage.images_dir(sid)
    finally:
        session_storage.delete_session(sid)


def test_unknown_extension_falls_back_to_jpg():
    sid = _sid()
    try:
        path = session_storage.save_image(sid, 0, "photo.tiff", b"x")
        assert path.name == "0.jpg"
    finally:
        session_storage.delete_session(sid)


def test_find_image_returns_saved_file_and_none_otherwise():
    sid = _sid()
    try:
        session_storage.save_image(sid, 0, "a.png", b"x")
        assert session_storage.find_image(sid, 0).name == "0.png"
        assert session_storage.find_image(sid, 5) is None
    finally:
        session_storage.delete_session(sid)


def test_find_image_rejects_non_uuid_session_id():
    assert session_storage.find_image("../etc", 0) is None
    assert session_storage.find_image("not-a-uuid", 0) is None


def test_delete_session_removes_directory():
    sid = _sid()
    session_storage.save_image(sid, 0, "a.jpg", b"x")
    assert session_storage.images_dir(sid).is_dir()
    session_storage.delete_session(sid)
    assert not (session_storage.SESSIONS_DIR / sid).exists()


def test_prune_all_removes_every_session_directory():
    sid_a, sid_b = _sid(), _sid()
    session_storage.save_image(sid_a, 0, "a.jpg", b"x")
    session_storage.save_image(sid_b, 0, "b.jpg", b"x")
    removed = session_storage.prune_all()
    assert removed >= 2
    assert not (session_storage.SESSIONS_DIR / sid_a).exists()
    assert not (session_storage.SESSIONS_DIR / sid_b).exists()


def test_save_image_rejects_non_uuid_session_id():
    import pytest
    with pytest.raises(ValueError):
        session_storage.save_image("../etc", 0, "a.jpg", b"x")
    assert not (session_storage.SESSIONS_DIR / "../etc").exists()


def test_multipart_without_profile_returns_422_and_leaves_no_session_dir(tmp_path):
    before = set(p.name for p in session_storage.SESSIONS_DIR.iterdir()) if session_storage.SESSIONS_DIR.is_dir() else set()
    with TestClient(app) as client:
        resp = client.post("/predict", files={"images": ("a.jpg", b"x", "image/jpeg")})
    assert resp.status_code == 422
    after = set(p.name for p in session_storage.SESSIONS_DIR.iterdir()) if session_storage.SESSIONS_DIR.is_dir() else set()
    assert after == before
