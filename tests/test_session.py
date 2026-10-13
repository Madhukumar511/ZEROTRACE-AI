import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from server import app
from zerotrace.core.session import SessionManager

client = TestClient(app)


def test_session_manager_crud(tmp_path):
    mgr = SessionManager(storage_dir=tmp_path)
    sess = mgr.create_session(provider="antigravity", model="gemini-3.8-flash-high")
    assert sess["session_id"] is not None
    assert sess["provider"] == "antigravity"

    # Append messages
    mgr.append_message(sess, role="user", content="Hello test")
    mgr.append_message(sess, role="assistant", content="Hi user", metrics={"tokens_saved": 5, "energy_saved_j": 1.2, "cache_hit": False})

    # Reload
    loaded = mgr.load_session(sess["session_id"])
    assert loaded is not None
    assert len(loaded["messages"]) == 2
    assert loaded["total_tokens_saved"] == 5
    assert loaded["total_energy_saved_j"] == 1.2

    # Get last session
    last = mgr.get_last_session()
    assert last["session_id"] == sess["session_id"]

    # List sessions
    all_s = mgr.list_sessions()
    assert len(all_s) == 1
    assert all_s[0]["session_id"] == sess["session_id"]


def test_session_api_endpoints():
    resp = client.get("/api/chat/sessions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert isinstance(data["sessions"], list)
