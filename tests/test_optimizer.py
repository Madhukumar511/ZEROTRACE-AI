import pytest
from zerotrace.core.optimizer import (
    apply_token_limit,
    apply_context_reduction,
    apply_gpu_routing,
    apply_caveman_compression_optimization,
)
from zerotrace.telemetry.hardware import apply_gpu_power_cap, reset_gpu_power_cap
from zerotrace.core.state import state


@pytest.fixture(autouse=True)
def reset_global_state():
    state.reset_all()
    yield
    state.reset_all()


def test_token_limit_modifier():
    opts = {"temperature": 0.8, "num_predict": 300}
    modified = apply_token_limit(opts, limit=75)
    assert modified["num_predict"] == 75
    assert modified["temperature"] == 0.8


def test_context_reduction_modifier():
    opts = {"num_ctx": 2048, "num_predict": 512}
    modified = apply_context_reduction(opts, ctx=256)
    assert modified["num_ctx"] == 256
    assert modified["num_predict"] <= 64


def test_gpu_routing_modes():
    opts = {}
    gpu_mode = apply_gpu_routing(opts, mode="gpu_lean")
    assert gpu_mode["num_gpu"] == 99
    assert gpu_mode["num_thread"] == 2

    cpu_mode = apply_gpu_routing(opts, mode="cpu_lean")
    assert cpu_mode["num_gpu"] == 0
    assert cpu_mode["num_thread"] == 4


def test_caveman_compression_optimization_profile():
    opts = {"num_predict": 250}
    new_opts, profile, desc = apply_caveman_compression_optimization(opts, intensity="full")
    assert "caveman_full" in profile
    assert new_opts["num_predict"] < 250
    assert "65%" in desc


def test_gpu_power_cap_safe_fallback():
    # If NVML not available or no permissions, returns structured error dict without crashing
    cap_res = apply_gpu_power_cap(60)
    assert "success" in cap_res

    reset_res = reset_gpu_power_cap()
    assert "success" in reset_res


def test_state_manager_transitions():
    state.set_active_model("llama3:latest")
    assert state.phase == "model_selected"
    assert state.active_ollama_model == "llama3:latest"

    state.update_baseline({"power_w": 45.0, "tokens_per_sec": 30.0})
    assert state.phase == "benchmarked"
    assert state.baseline_snapshot["power_w"] == 45.0

    state.reset_optimizations()
    assert state.phase == "benchmarked"
    assert state.ollama_options == {}
