import time
import json
import asyncio
import threading
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import psutil

from zerotrace.core.state import state
from zerotrace.core.emissions import get_carbon
from zerotrace.telemetry.process import get_isolated_ai_cpu
from zerotrace.telemetry.hardware import get_gpu_telemetry, get_cpu_power_estimate
from zerotrace.models.ollama import ollama_client

router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])


@router.get("/live")
def live_telemetry():
    cpu_pct = get_isolated_ai_cpu()
    gpu = get_gpu_telemetry()
    gpu_util = gpu["gpu_util_pct"]
    gpu_w = gpu["gpu_power_w"]

    is_actually_running = state.inference_active or state.is_stress_testing or (cpu_pct > 2.0) or (gpu_util > 5.0)

    # Zero out background OS noise when idle
    if not is_actually_running:
        cpu_pct = 0.0
        gpu_util = 0.0

    cpu_w = get_cpu_power_estimate(cpu_pct)
    total_w = round(cpu_w + gpu_w, 2)
    carbon_kwh = get_carbon()

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "cpu_pct": cpu_pct,
        "ram_pct": round(psutil.virtual_memory().percent, 1),
        "gpu_util": gpu_util,
        "gpu_mem_pct": gpu["gpu_mem_pct"],
        "gpu_mem_used_gb": gpu["gpu_mem_used_gb"],
        "gpu_power_w": gpu_w,
        "cpu_power_w": cpu_w,
        "total_power_w": total_w,
        "gpu_available": gpu["available"],
        "active_model": state.active_ollama_model,
        "profile": state.ollama_profile,
        "is_testing": state.is_stress_testing,
        "inference_active": is_actually_running,
        "model_status": "running" if is_actually_running else "idle",
        "phase": state.phase,
        "baseline_power_w": state.baseline_snapshot["power_w"],
        "caveman_active": state.caveman_active,
        "total_energy_saved_j": round(state.total_energy_saved_j, 3),
        "total_energy_saved_wh": round(state.total_energy_saved_j / 3600.0, 4),
        "total_carbon_saved_g": round((state.total_energy_saved_j / 3_600_000.0) * carbon_kwh, 6),
        "total_tokens_saved": state.total_tokens_saved,
        "total_cache_hits": state.total_cache_hits,
        "cache_size": state.vector_cache.size(),
    }
    state.record_telemetry(snapshot)
    return snapshot


@router.get("/stream")
async def metrics_stream():
    async def gen():
        get_isolated_ai_cpu()  # Prime the sweeper

        while True:
            await asyncio.sleep(1.0)
            cpu_pct = get_isolated_ai_cpu()
            gpu = get_gpu_telemetry()
            gpu_util = gpu["gpu_util_pct"]
            gpu_w = gpu["gpu_power_w"]

            is_actually_running = state.inference_active or state.is_stress_testing or (cpu_pct > 2.0) or (gpu_util > 5.0)

            if not is_actually_running:
                cpu_pct = 0.0
                gpu_util = 0.0

            cpu_w = get_cpu_power_estimate(cpu_pct)
            total_w = round(cpu_w + gpu_w, 2)
            baseline_w = state.baseline_snapshot["power_w"]
            carbon_kwh = get_carbon()

            frame = {
                "cpu_pct": cpu_pct,
                "ram_pct": round(psutil.virtual_memory().percent, 1),
                "gpu_util": gpu_util,
                "gpu_power_w": gpu_w,
                "total_power_w": total_w,
                "is_testing": state.is_stress_testing,
                "inference_active": is_actually_running,
                "model_status": "running" if is_actually_running else "idle",
                "active_model": state.active_ollama_model,
                "profile": state.ollama_profile,
                "phase": state.phase,
                "baseline_w": baseline_w,
                "baseline_cpu_pct": state.baseline_snapshot.get("cpu_pct"),
                "baseline_gpu_util": state.baseline_snapshot.get("gpu_util"),
                "baseline_gpu_pw": state.baseline_snapshot.get("gpu_power_w"),
                "saving_w": round(baseline_w - total_w, 2) if baseline_w else None,
                "saving_pct": (
                    round((baseline_w - total_w) / baseline_w * 100, 1)
                    if baseline_w and baseline_w > 0 else None
                ),
                "caveman_active": state.caveman_active,
                "total_energy_saved_j": round(state.total_energy_saved_j, 3),
                "total_energy_saved_wh": round(state.total_energy_saved_j / 3600.0, 4),
                "total_carbon_saved_g": round((state.total_energy_saved_j / 3_600_000.0) * carbon_kwh, 6),
                "total_tokens_saved": state.total_tokens_saved,
                "total_cache_hits": state.total_cache_hits,
                "cache_size": state.vector_cache.size(),
            }
            yield f"data: {json.dumps(frame)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _stress_worker():
    prompts = [
        "What is 2 + 2?",
        "Name a color.",
        "What is the capital of France?",
        "Say hello in Spanish.",
        "What year did WW2 end?",
    ]
    idx = 0
    while state.is_stress_testing:
        try:
            target_model = state.active_ollama_model or state.pytorch_model_name
            ollama_client.generate(
                model=target_model,
                prompt=prompts[idx % len(prompts)],
                options=state.ollama_options,
                timeout=30,
            )
            idx += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"[STRESS] {e}")
            time.sleep(1)


@router.post("/toggle-load")
def toggle_stress_test(payload: dict):
    enable = payload.get("enable", False)

    if enable and not state.is_stress_testing:
        if not state.active_ollama_model and not state.pytorch_model:
            raise HTTPException(400, "Load a model first.")
        if state.baseline_snapshot["power_w"] is None and state.active_ollama_model:
            raise HTTPException(400, "Run /api/benchmark first to establish baseline.")
        state.is_stress_testing = True
        state.stress_thread = threading.Thread(target=_stress_worker, daemon=True, name="stress-worker")
        state.stress_thread.start()
        return {"status": "started", "options_in_use": state.ollama_options}

    elif not enable and state.is_stress_testing:
        state.is_stress_testing = False
        if state.stress_thread:
            state.stress_thread.join(timeout=15)
        return {"status": "stopped"}

    return {"status": "unchanged"}


class RecordSavingsRequest(BaseModel):
    tokens_saved: int = Field(0, description="Tokens saved by Caveman compression or caching")
    energy_saved_j: float = Field(0.0, description="Electricity saved in Joules (optional, auto-estimated if 0)")
    cache_hits: int = Field(0, description="Number of 0-Watt cache hits (optional)")
    source: str = Field("antigravity", description="Client identifier (e.g. 'antigravity', 'cursor', 'claude_code')")
    note: str = Field("", description="Optional note or query summary")


@router.post("/record")
def record_external_savings(req: RecordSavingsRequest):
    """
    Allows external AI assistants (Antigravity CLI, Cursor, Claude Code, Aider)
    to dynamically log token and electrical energy savings into ZeroTrace's live telemetry dashboard.
    """
    carbon_kwh = get_carbon()
    energy_added = req.energy_saved_j

    if req.tokens_saved > 0:
        state.total_tokens_saved += req.tokens_saved
        # Physics model: ~0.08 Joules per LLM token on typical multi-socket / GPU cluster inference
        if energy_added <= 0.0:
            energy_added = round(req.tokens_saved * 0.08, 4)

    if energy_added > 0:
        state.total_energy_saved_j += energy_added

    if req.cache_hits > 0:
        state.total_cache_hits += req.cache_hits

    co2_g = (state.total_energy_saved_j / 3_600_000.0) * carbon_kwh

    return {
        "success": True,
        "recorded": {
            "tokens_saved": req.tokens_saved,
            "energy_saved_j": energy_added,
            "cache_hits": req.cache_hits,
            "source": req.source,
        },
        "cumulative": {
            "total_tokens_saved": state.total_tokens_saved,
            "total_energy_saved_j": round(state.total_energy_saved_j, 3),
            "total_energy_saved_wh": round(state.total_energy_saved_j / 3600.0, 4),
            "total_carbon_saved_g": round(co2_g, 6),
            "total_cache_hits": state.total_cache_hits,
        },
    }


@router.get("/autodiscover")
def get_autodiscover():
    """Returns detected offline models, AI agents, and auto-connected hardware profile."""
    from zerotrace.core.autodiscover import autodiscover
    return autodiscover.auto_connect()
