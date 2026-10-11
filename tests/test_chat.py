import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from server import app
from zerotrace.core.state import state

client = TestClient(app)


def test_get_chat_providers():
    resp = client.get("/api/chat/providers")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "local" in data
    assert "cloud" in data
    assert data["local"]["id"] == "ollama"

    cloud_ids = [c["id"] for c in data["cloud"]]
    assert "gemini" in cloud_ids
    assert "openai" in cloud_ids
    assert "groq" in cloud_ids
    assert "anthropic" in cloud_ids


def test_cache_stats_endpoint():
    resp = client.get("/api/cache/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "cached_queries_count" in data
    assert data["dimension"] == 384
    assert data["cached_queries_count"] >= 4  # seeded queries


def test_chat_instant_cache_hit():
    """Testing that exact or semantic seed query results in instant 0-watt cache hit."""
    resp = client.post("/api/chat", json={
        "message": "what is zerotrace ai",
        "provider": "ollama",
        "use_cache": True,
        "cache_threshold": 0.85,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["cache_hit"] is True
    assert "ZeroTrace AI" in data["answer"]
    assert data["similarity_score"] >= 0.85
    assert data["metrics"]["energy_cost_j"] == 0.0
    assert "Zero-Watt" in data["metrics"]["status"]


def test_chat_empty_message_validation():
    resp = client.post("/api/chat", json={
        "message": "   ",
        "provider": "ollama",
    })
    assert resp.status_code == 400


def test_chat_mock_cloud_gemini_success():
    mock_resp = {
        "success": True,
        "text": "Energy-efficient green AI answer.",
        "provider": "gemini",
        "model": "gemini-1.5-flash",
        "prompt_tokens": 12,
        "completion_tokens": 8,
        "total_tokens": 20,
        "duration_s": 0.35,
    }

    with patch("zerotrace.models.cloud.CloudLLMClient.call_gemini", return_value=mock_resp):
        resp = client.post("/api/chat", json={
            "message": "can you explain how solar panels work in detail please",
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "api_key": "test_fake_key",
            "use_caveman": True,
            "caveman_intensity": "full",
            "use_cache": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["cache_hit"] is False
        assert data["provider"] == "gemini"
        assert data["answer"] == "Energy-efficient green AI answer."
        assert data["tokens_saved"] > 0
        assert "caveman_compressed" in data["metrics"]


def test_chat_mock_cloud_openai_success():
    mock_resp = {
        "success": True,
        "text": "def reverse_str(s): return s[::-1]",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "prompt_tokens": 15,
        "completion_tokens": 10,
        "total_tokens": 25,
        "duration_s": 0.28,
    }

    with patch("zerotrace.models.cloud.CloudLLMClient.call_openai_compatible", return_value=mock_resp):
        resp = client.post("/api/chat", json={
            "message": "could you kindly write a python function to reverse a string",
            "provider": "openai",
            "model": "gpt-4o-mini",
            "api_key": "test_fake_key",
            "use_caveman": True,
            "caveman_intensity": "full",
            "use_cache": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["provider"] == "openai"
        assert "reverse_str" in data["answer"]
