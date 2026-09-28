import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from server import (
    app,
    get_cpu_power_estimate,
    get_total_system_power,
    get_carbon,
    apply_token_limit,
    apply_context_reduction,
    apply_gpu_routing,
)

client = TestClient(app)

def test_frontend_served():
    """Verify that root endpoint serves the index.html dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "html" in response.headers.get("content-type", "").lower()

def test_health_check():
    """Verify health endpoint returns status, power, and carbon telemetry."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "ollama_online" in data
    assert "current_power_w" in data
    assert "live_carbon" in data
    assert "phase" in data
    assert data["current_power_w"] >= 0

def test_live_telemetry_endpoint():
    """Verify live telemetry provides hardware power and utilization metrics."""
    response = client.get("/api/telemetry/live")
    assert response.status_code == 200
    data = response.json()
    assert "cpu_pct" in data
    assert "ram_pct" in data
    assert "gpu_util" in data
    assert "total_power_w" in data
    assert "timestamp" in data
    assert data["ram_pct"] > 0

def test_session_state_endpoint():
    """Verify session metadata, baseline snapshot, and active profile."""
    response = client.get("/api/session")
    assert response.status_code == 200
    data = response.json()
    assert "profile" in data
    assert "phase" in data
    assert "carbon_gco2_kwh" in data
    assert "telemetry_points" in data
    assert data["carbon_gco2_kwh"] > 0

def test_emissions_endpoint():
    """Verify emissions endpoint tracks history."""
    response = client.get("/emissions")
    assert response.status_code == 200
    data = response.json()
    assert "records" in data
    assert "count" in data
    assert isinstance(data["records"], list)

def test_benchmark_requires_model():
    """Verify running benchmark without model selection raises 400."""
    response = client.post("/api/benchmark")
    assert "model" in response.json()["detail"].lower()
    assert "selected" in response.json()["detail"].lower()

def test_stress_test_requires_model():
    """Verify toggle stress test fails safely without active model."""
    response = client.post("/api/telemetry/toggle-load", json={"enable": True})
    assert response.status_code == 400
    assert "load a model" in response.json()["detail"].lower()

def test_cpu_power_calculation():
    """Verify CPU power estimation math."""
    p_zero = get_cpu_power_estimate(0.0)
    p_full = get_cpu_power_estimate(100.0)
    assert p_zero == 0.0
    assert p_full > 30.0  # Min TDP is 35W for i3/ryzen 3

def test_system_power_aggregation():
    """Verify aggregated total system power."""
    power = get_total_system_power(50.0)
    assert "cpu_w" in power
    assert "gpu_w" in power
    assert "total_w" in power
    assert power["total_w"] >= power["cpu_w"]

def test_optimization_techniques():
    """Verify optimization modifiers adjust configuration options."""
    base_opts = {"num_predict": 250, "temperature": 0.7}

    # 1. Token limit
    limited = apply_token_limit(base_opts, limit=60)
    assert limited["num_predict"] == 60

    # 2. Context reduction
    reduced = apply_context_reduction(base_opts, ctx=256)
    assert reduced["num_ctx"] == 256
    assert reduced["num_predict"] <= 64

    # 3. GPU routing
    gpu_lean = apply_gpu_routing(base_opts, mode="gpu_lean")
    assert gpu_lean["num_gpu"] == 99
    assert gpu_lean["num_thread"] == 2

    cpu_lean = apply_gpu_routing(base_opts, mode="cpu_lean")
    assert cpu_lean["num_gpu"] == 0
    assert cpu_lean["num_thread"] == 4
