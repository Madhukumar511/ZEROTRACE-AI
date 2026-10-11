import statistics
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

from zerotrace.config import INDEX_HTML_PATH
from zerotrace.core.state import state
from zerotrace.core.emissions import get_carbon
from zerotrace.telemetry.aggregator import get_total_system_power
from zerotrace.models.ollama import ollama_client
from zerotrace.models.pytorch_loader import (
    TORCH_OK,
    load_local_archive_or_weights,
    load_huggingface_model,
)

router = APIRouter(tags=["models"])
_executor = ThreadPoolExecutor(max_workers=2)


@router.get("/")
def serve_frontend():
    if INDEX_HTML_PATH.exists():
        return FileResponse(str(INDEX_HTML_PATH))
    return {"status": "ok", "service": "Zerotrace AI"}


@router.get("/health")
def health():
    ollama_online = ollama_client.is_online()
    models = ollama_client.list_models() if ollama_online else []
    model_names = [m["name"] for m in models]
    power = get_total_system_power(0.0)

    return {
        "status": "ok",
        "ollama_online": ollama_online,
        "ollama_models": model_names,
        "active_model": state.active_ollama_model,
        "live_carbon": get_carbon(),
        "current_power_w": power["total_w"],
        "optimization_profile": state.ollama_profile,
        "active_options": state.ollama_options,
        "phase": state.phase,
        "baseline_measured": state.baseline_snapshot["power_w"] is not None,
    }


@router.get("/api/session")
def get_session():
    recent = list(state.telemetry_history)[-60:] if state.telemetry_history else []
    powers = [t["total_power_w"] for t in recent if t.get("total_power_w")]
    avg_p = statistics.mean(powers) if powers else 0

    return {
        "active_model": state.active_ollama_model,
        "profile": state.ollama_profile,
        "options": state.ollama_options,
        "phase": state.phase,
        "baseline": state.baseline_snapshot,
        "avg_power_1min_w": round(avg_p, 2),
        "carbon_gco2_kwh": get_carbon(),
        "telemetry_points": len(state.telemetry_history),
    }


@router.get("/emissions")
def get_emissions():
    return {
        "records": list(state.telemetry_history),
        "count": len(state.telemetry_history),
    }


@router.post("/api/load-local-model")
async def load_local_model_endpoint(model_file: UploadFile = File(...)):
    state.active_ollama_model = None
    state.phase = "idle"
    if not TORCH_OK:
        return {"success": False, "error": "PyTorch is required for local models. Install with: pip install torch transformers"}

    result = load_local_archive_or_weights(model_file.file, model_file.filename)
    if result.get("success"):
        state.pytorch_model = result.get("model")
        state.pytorch_tokenizer = result.get("tokenizer")
        state.pytorch_model_name = result.get("model_name", "local_model")
        return {
            "success": True,
            "model_name": state.pytorch_model_name,
            "params_b": result.get("params_b", 0.0),
            "note": result.get("note"),
        }
    return result


@router.post("/api/load-hf-model")
async def load_hf_model_endpoint(payload: dict):
    state.active_ollama_model = None
    state.phase = "idle"
    if not TORCH_OK:
        return {"success": False, "error": "PyTorch is required for HuggingFace models. Install with: pip install torch transformers"}

    model_id = payload.get("model_id", "").strip()
    if not model_id:
        raise HTTPException(400, "model_id is required")

    loop = asyncio.get_event_loop()
    t0 = time.time()
    result = await loop.run_in_executor(_executor, lambda: load_huggingface_model(model_id))
    if result.get("success"):
        state.pytorch_model = result.get("model")
        state.pytorch_tokenizer = result.get("tokenizer")
        state.pytorch_model_name = model_id
        return {
            "success": True,
            "model_name": model_id,
            "params_b": result.get("params_b", 0.0),
            "load_time_s": round(time.time() - t0, 1),
        }
    return result
