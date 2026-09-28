# ==========================================
# Zerotrace AI — Real Measurements Only
# Zero fake data. Real Ollama integration.
# Real hardware telemetry. 5 live techniques.
# ==========================================

import os
import time
import math
import threading
import asyncio
import json
import io
import copy
import shutil
import tempfile
import zipfile
import statistics
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel

import requests
import psutil
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.nn.utils.prune as prune
    import torch.quantization
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_OK = True
except Exception:
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None
    TORCH_OK = False


# ── Hardware Integrations ──
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_OK = True
except Exception:
    NVML_OK = False

try:
    from zeus.monitor import ZeusMonitor
    # approx_instant_energy=True gives highly accurate joule counts on Volta+ architectures
    _zeus_monitor = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)
    ZEUS_OK = True
except Exception:
    _zeus_monitor = None
    ZEUS_OK = False

# ==========================================
# SECTION 1 — GLOBAL STATE
# ==========================================

# Phase State Machine: "idle" -> "model_selected" -> "benchmarked" -> "optimized"
_phase: str = "idle"
_inference_active: bool = False

BENCHMARK_PROMPT = (
    "Explain how transformer neural networks work. "
    "Cover self-attention, positional encoding, feed-forward layers, and training. "
    "Give a detailed explanation with at least 250 words."
)

# Ollama state
_active_ollama_model: Optional[str] = None
_ollama_options: dict = {}          
_ollama_profile: str = "none"       

# PyTorch state 
_pytorch_model = None
_pytorch_tokenizer = None
_pytorch_model_name: str = "None"

# Stress test state
_is_stress_testing: bool = False
_stress_thread: Optional[threading.Thread] = None

# Baseline snapshot
_baseline_snapshot: Dict = {
    "power_w":          None,   
    "tokens_per_sec":   None,   
    "latency_ms":       None,   
    "energy_j_per_tok": None,   
    "energy_j":         None,   
    "tokens_generated": None,
    "cpu_pct":          None,
    "gpu_util":         None,
    "gpu_power_w":      None,
    "model":            None,
    "measured_at":      None,
    "n_runs":           None,
    "benchmark_prompt": None,
}

_telemetry_history: deque = deque(maxlen=300)

_carbon_gco2_kwh: float = 214.0
_carbon_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=2)

_ui_smooth_state = {
    "gpu_util": 0.0,
    "gpu_w": 0.0
}

# Process Tracker Dictionary (Tracks all active Ollama/Llama processes globally)
_ollama_procs: Dict[int, psutil.Process] = {}


# ==========================================
# SECTION 2 — REAL PROCESS & HARDWARE TELEMETRY
# ==========================================

def get_isolated_ai_cpu() -> float:
    """
    Sweeps the entire computer for ANY process related to Ollama or Llama.
    Sums up their CPU usage and divides by core count to get the true
    system-wide percentage used ONLY by the AI, ignoring Windows noise.
    """
    global _ollama_procs
    current_pids = []
    total_cpu = 0.0
    
    for p in psutil.process_iter(['name']):
        try:
            name = p.info['name'].lower()
            if 'ollama' in name or 'llama' in name:
                pid = p.pid
                current_pids.append(pid)
                if pid not in _ollama_procs:
                    _ollama_procs[pid] = p
                    p.cpu_percent(interval=None) # Prime the pump
                else:
                    total_cpu += _ollama_procs[pid].cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
    # Clean up processes that closed
    keys_to_remove = [pid for pid in _ollama_procs if pid not in current_pids]
    for pid in keys_to_remove:
        del _ollama_procs[pid]
        
    # psutil returns >100% for multi-threaded processes. Divide by total cores.
    core_count = psutil.cpu_count() or 1
    sys_cpu = total_cpu / core_count
    return round(sys_cpu, 1)


def get_gpu_telemetry() -> dict:
    """Returns real GPU stats via pynvml (NVIDIA Management Library)."""
    if NVML_OK:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util    = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem     = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            name    = pynvml.nvmlDeviceGetName(handle)
            return {
                "gpu_util_pct":    round(util.gpu, 1),
                "gpu_mem_pct":     round(mem.used / mem.total * 100, 1),
                "gpu_mem_used_gb": round(mem.used / 1e9, 2),
                "gpu_mem_total_gb":round(mem.total / 1e9, 2),
                "gpu_power_w":     round(power_mw / 1000.0, 2),
                "gpu_name":        name if isinstance(name, str) else name.decode(),
                "available":       True,
            }
        except Exception:
            pass

    return {
        "gpu_util_pct": 0, "gpu_mem_pct": 0,
        "gpu_mem_used_gb": 0, "gpu_mem_total_gb": 0,
        "gpu_power_w": 0, "gpu_name": "N/A", "available": False,
    }


def get_cpu_power_estimate(cpu_pct: float) -> float:
    """Estimates CPU package power using CPU% × TDP."""
    tdp_w = 65.0 
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if "model name" in line.lower():
                    name = line.lower()
                    if "i9" in name or "ryzen 9" in name:
                        tdp_w = 125.0
                    elif "i7" in name or "ryzen 7" in name:
                        tdp_w = 95.0
                    elif "i5" in name or "ryzen 5" in name:
                        tdp_w = 65.0
                    elif "i3" in name or "ryzen 3" in name:
                        tdp_w = 35.0
                    break
    except Exception:
        pass

    return round(tdp_w * (cpu_pct / 100.0), 2)


def get_total_system_power(cpu_pct: float) -> dict:
    gpu    = get_gpu_telemetry()
    cpu_w  = get_cpu_power_estimate(cpu_pct)
    gpu_w  = gpu["gpu_power_w"]
    total  = round(cpu_w + gpu_w, 2)
    return {
        "cpu_w":   cpu_w,
        "gpu_w":   gpu_w,
        "total_w": total,
        "source":  "NVML (GPU) + TDP×util estimate (CPU)",
        "gpu_info": gpu,
    }


# ==========================================
# SECTION 3 — CARBON INTENSITY
# ==========================================

def _carbon_refresh_loop():
    global _carbon_gco2_kwh
    while True:
        try:
            r = requests.get(
                "https://api.carbonintensity.org.uk/intensity",
                headers={"Accept": "application/json"},
                timeout=4,
            )
            if r.status_code == 200:
                val = r.json()["data"][0]["intensity"]["actual"]
                if val is not None:
                    with _carbon_lock:
                        _carbon_gco2_kwh = float(val)
        except Exception:
            pass
        time.sleep(300)

threading.Thread(target=_carbon_refresh_loop, daemon=True).start()

def get_carbon() -> float:
    with _carbon_lock:
        return _carbon_gco2_kwh


# ==========================================
# SECTION 4 — REAL OLLAMA MEASUREMENT
# Concurrent hardware sampler + Zeus integration
# ==========================================

def measure_ollama_inference(model: str, prompt: str, options: dict = None) -> dict:
    global _inference_active
    samples = []
    stop_flag = threading.Event()

    def _sampler():
        get_isolated_ai_cpu() # Prime the processes
        while not stop_flag.is_set():
            time.sleep(0.2)
            
            # Isolated AI CPU tracking
            cpu = get_isolated_ai_cpu()
            gpu = get_gpu_telemetry()
            cpu_w = get_cpu_power_estimate(cpu)
            total_w = round(cpu_w + gpu["gpu_power_w"], 2)
            
            samples.append({
                "total_w":    total_w,
                "cpu_pct":    cpu,
                "gpu_util":   gpu["gpu_util_pct"],
                "gpu_power_w":gpu["gpu_power_w"],
            })

    sampler = threading.Thread(target=_sampler, daemon=True)
    sampler.start()
    _inference_active = True
    t_start = time.perf_counter()
    
    window_key = f"inference_{int(time.time()*1000)}"
    if ZEUS_OK and _zeus_monitor:
        try:
            _zeus_monitor.begin_window(window_key, sync_execution=False)
        except Exception:
            pass

    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False, "options": options or {}},
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        _inference_active = False
        stop_flag.set()
        sampler.join(timeout=5)
        
        zeus_energy_j = None
        if ZEUS_OK and _zeus_monitor:
            try:
                zeus_result = _zeus_monitor.end_window(window_key, sync_execution=False)
                zeus_energy_j = zeus_result.total_energy
            except Exception:
                pass

    wall_s = time.perf_counter() - t_start

    if not samples:
        return {"success": False, "error": "No hardware samples collected — inference was too fast"}

    avg_power_w  = statistics.mean(s["total_w"]     for s in samples)
    avg_cpu_pct  = statistics.mean(s["cpu_pct"]     for s in samples)
    avg_gpu_util = statistics.mean(s["gpu_util"]    for s in samples)
    avg_gpu_pw   = statistics.mean(s["gpu_power_w"] for s in samples)
    energy_j     = avg_power_w * wall_s

    final_energy_j = zeus_energy_j if zeus_energy_j is not None else energy_j

    eval_count   = data.get("eval_count", 0)
    eval_dur_ns  = data.get("eval_duration", 1)
    prompt_count = data.get("prompt_eval_count", 0)
    tok_per_sec  = (eval_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0.0

    return {
        "success":          True,
        "model":            model,
        "options_used":     options or {},
        "wall_time_s":      round(wall_s, 3),
        "tokens_generated": eval_count,
        "prompt_tokens":    prompt_count,
        "tokens_per_sec":   round(tok_per_sec, 2),
        "avg_power_w":      round(avg_power_w, 2),
        "avg_cpu_pct":      round(avg_cpu_pct, 1),
        "avg_gpu_util":     round(avg_gpu_util, 1),
        "avg_gpu_power_w":  round(avg_gpu_pw, 2),
        "energy_j":         round(final_energy_j, 4),
        "energy_source":    "zeus_nvml" if zeus_energy_j is not None else "power_x_time_estimate",
        "energy_j_per_tok": round(final_energy_j / max(eval_count, 1), 6),
        "energy_kwh":       round(final_energy_j / 3_600_000, 9),
        "co2_g":            round((final_energy_j / 3_600_000) * get_carbon(), 9),
        "sample_count":     len(samples),
        "response_preview": data.get("response", "")[:200],
    }


# ==========================================
# SECTION 5 — THE 5 REAL ENERGY TECHNIQUES
# ==========================================

def apply_token_limit(current_options: dict, limit: int = 80) -> dict:
    new_opts = dict(current_options)
    new_opts["num_predict"] = limit
    return new_opts

def apply_context_reduction(current_options: dict, ctx: int = 512) -> dict:
    new_opts = dict(current_options)
    new_opts["num_ctx"]     = ctx
    new_opts["num_predict"] = min(current_options.get("num_predict", 200), ctx // 4)
    return new_opts

def apply_gpu_routing(current_options: dict, mode: str = "gpu_lean") -> dict:
    new_opts = dict(current_options)
    if mode == "gpu_lean":
        new_opts["num_gpu"]    = 99   
        new_opts["num_thread"] = 2    
    elif mode == "cpu_lean":
        new_opts["num_gpu"]    = 0    
        new_opts["num_thread"] = 4    
    return new_opts

def get_quantization_variants(model_name: str) -> dict:
    base = model_name.split(":")[0]
    variants = {
        "q2_k":   {"bits": 2, "energy_factor": 0.35, "quality": "Low",    "cmd": f"ollama pull {base}:q2_K"},
        "q4_0":   {"bits": 4, "energy_factor": 0.50, "quality": "Medium", "cmd": f"ollama pull {base}:q4_0"},
        "q4_k_m": {"bits": 4, "energy_factor": 0.52, "quality": "Good",   "cmd": f"ollama pull {base}:q4_K_M"},
        "q5_k_m": {"bits": 5, "energy_factor": 0.65, "quality": "Better", "cmd": f"ollama pull {base}:q5_K_M"},
        "q8_0":   {"bits": 8, "energy_factor": 0.90, "quality": "Best",   "cmd": f"ollama pull {base}:q8_0"},
    }
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        downloaded = [m["name"] for m in r.json().get("models", [])]
        for k, v in variants.items():
            tag = f"{base}:{k.replace('_', '-')}"
            v["downloaded"] = any(tag in d or k in d for d in downloaded)
    except Exception:
        for v in variants.values():
            v["downloaded"] = False
    return {"base_model": base, "variants": variants}

def apply_gpu_power_cap(power_limit_w: int) -> dict:
    if not NVML_OK:
        return {"success": False, "error": "pynvml not available"}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        target_mw = max(min_mw, min(int(power_limit_w * 1000), max_mw))
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, target_mw)
        actual_w = target_mw / 1000
        return {
            "success": True,
            "requested_w": power_limit_w,
            "actual_w": actual_w,
            "min_w": min_mw / 1000,
            "max_w": max_mw / 1000,
        }
    except pynvml.NVMLError as e:
        return {"success": False, "error": str(e), "note": "Needs admin/root privileges"}

def reset_gpu_power_cap() -> dict:
    if not NVML_OK:
        return {"success": False, "error": "pynvml not available"}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_mw)
        return {"success": True, "restored_w": default_mw / 1000}
    except Exception as e:
        return {"success": False, "error": str(e)}

# ==========================================
# SECTION 6 — FASTAPI APP + ALL ENDPOINTS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(title="Zerotrace AI — Global AI Tracker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_frontend():
    idx = BASE_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return {"status": "ok", "service": "Zerotrace AI"}

@app.get("/health")
def health():
    ollama_online = False
    ollama_models: List[str] = []
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        if r.status_code == 200:
            ollama_online = True
            ollama_models = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass

    power = get_total_system_power(0.0)
    return {
        "status":               "ok",
        "ollama_online":        ollama_online,
        "ollama_models":        ollama_models,
        "active_model":         _active_ollama_model,
        "live_carbon":          get_carbon(),
        "current_power_w":      power["total_w"],
        "optimization_profile": _ollama_profile,
        "active_options":       _ollama_options,
        "phase":                _phase,
        "baseline_measured":    _baseline_snapshot["power_w"] is not None,
    }


@app.get("/api/telemetry/live")
def live_telemetry():
    cpu_pct = get_isolated_ai_cpu()
    gpu = get_gpu_telemetry()
    gpu_util = gpu["gpu_util_pct"]
    gpu_w    = gpu["gpu_power_w"]

    # IS OLLAMA GLOBALLY RUNNING ANYWHERE? (CMD or Dashboard)
    is_actually_running = _inference_active or _is_stress_testing or (cpu_pct > 2.0) or (gpu_util > 5.0)

    # ZERO OUT ALL NOISE IF IDLE
    if not is_actually_running:
        cpu_pct = 0.0
        gpu_util = 0.0

    cpu_w   = get_cpu_power_estimate(cpu_pct)
    total_w = round(cpu_w + gpu_w, 2)

    snapshot = {
        "timestamp":        datetime.now().isoformat(),
        "cpu_pct":          cpu_pct,
        "ram_pct":          round(psutil.virtual_memory().percent, 1),
        "gpu_util":         gpu_util,
        "gpu_mem_pct":      gpu["gpu_mem_pct"],
        "gpu_mem_used_gb":  gpu["gpu_mem_used_gb"],
        "gpu_power_w":      gpu_w,
        "cpu_power_w":      cpu_w,
        "total_power_w":    total_w,
        "gpu_available":    gpu["available"],
        "active_model":     _active_ollama_model,
        "profile":          _ollama_profile,
        "is_testing":       _is_stress_testing,
        "inference_active": is_actually_running,
        "model_status":     "running" if is_actually_running else "idle",
        "phase":            _phase,
        "baseline_power_w": _baseline_snapshot["power_w"],
    }
    _telemetry_history.append(snapshot)
    return snapshot


@app.get("/api/metrics/stream")
async def metrics_stream():
    async def gen():
        get_isolated_ai_cpu() # Prime the sweeper
        
        while True:
            await asyncio.sleep(1.0) 
            
            # 1. Sweep all Ollama processes globally
            cpu_pct = get_isolated_ai_cpu()
            
            # 2. Get hardware GPU data
            gpu = get_gpu_telemetry()
            gpu_util = gpu["gpu_util_pct"]
            gpu_w    = gpu["gpu_power_w"]

            # 3. If any metric spikes, flag it as RUNNING globally
            is_actually_running = _inference_active or _is_stress_testing or (cpu_pct > 2.0) or (gpu_util > 5.0)

            # 4. Force strict silence if idle
            if not is_actually_running:
                cpu_pct = 0.0
                gpu_util = 0.0

            cpu_w    = get_cpu_power_estimate(cpu_pct)
            total_w  = round(cpu_w + gpu_w, 2)
            baseline_w = _baseline_snapshot["power_w"]

            frame = {
                "cpu_pct":       cpu_pct,
                "ram_pct":       round(psutil.virtual_memory().percent, 1),
                "gpu_util":      gpu_util,
                "gpu_power_w":   gpu_w,
                "total_power_w": total_w,
                "is_testing":    _is_stress_testing,
                "inference_active": is_actually_running,
                "model_status":  "running" if is_actually_running else "idle",
                "active_model":  _active_ollama_model,
                "profile":       _ollama_profile,
                "phase":         _phase,
                "baseline_w":    baseline_w,
                "baseline_cpu_pct":  _baseline_snapshot.get("cpu_pct"),
                "baseline_gpu_util": _baseline_snapshot.get("gpu_util"),
                "baseline_gpu_pw":   _baseline_snapshot.get("gpu_power_w"),
                "saving_w":      round(baseline_w - total_w, 2) if baseline_w else None,
                "saving_pct":    (
                    round((baseline_w - total_w) / baseline_w * 100, 1)
                    if baseline_w and baseline_w > 0 else None
                ),
            }
            yield f"data: {json.dumps(frame)}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/ollama/models")
def get_ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        if r.status_code == 200:
            raw    = r.json().get("models", [])
            models = []
            for m in raw:
                models.append({
                    "name":    m["name"],
                    "size_gb": round(m.get("size", 0) / 1e9, 2),
                    "family":  m.get("details", {}).get("family", "unknown"),
                    "params":  m.get("details", {}).get("parameter_size", "unknown"),
                    "quant":   m.get("details", {}).get("quantization_level", "unknown"),
                })
            return {"success": True, "models": models}
    except Exception:
        pass
    return {
        "success": False,
        "models": [],
        "error": "Ollama not running. Start with: ollama serve",
    }


@app.post("/api/ollama/set-active")
def set_active_ollama(payload: dict):
    global _active_ollama_model, _ollama_options, _ollama_profile
    global _pytorch_model, _baseline_snapshot, _phase

    model_name = payload.get("model_name", "").strip()
    if not model_name:
        raise HTTPException(400, "model_name is required")

    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        available = [m["name"] for m in r.json().get("models", [])]
        if model_name not in available:
            raise HTTPException(
                404,
                f"Model '{model_name}' not found in Ollama. Available: {available}",
            )
    except HTTPException:
        raise
    except Exception:
        pass  

    _active_ollama_model = model_name
    _pytorch_model       = None   
    _ollama_options      = {}     
    _ollama_profile      = "none"
    _phase               = "model_selected"

    for k in _baseline_snapshot:
        _baseline_snapshot[k] = None
    _baseline_snapshot["model"] = model_name

    reset_gpu_power_cap()

    return {
        "success": True,
        "active":  model_name,
        "phase":   _phase,
        "message": "Model set. Run /api/benchmark to measure real baseline.",
    }


@app.post("/api/benchmark")
def run_benchmark(payload: dict = {}):
    global _baseline_snapshot, _phase

    if not _active_ollama_model:
        raise HTTPException(400, "No Ollama model selected. Call /api/ollama/set-active first.")
    
    if _phase not in ["model_selected", "benchmarked", "optimized"]:
        raise HTTPException(400, "Invalid phase. Call /api/ollama/set-active first.")

    prompt = payload.get("prompt", BENCHMARK_PROMPT)
    n_runs = int(payload.get("n_runs", 3))

    results = []
    for _ in range(n_runs):
        r = measure_ollama_inference(
            model=_active_ollama_model,
            prompt=prompt,
            options={"num_predict": 250},  
        )
        if r.get("success"):
            results.append(r)
        time.sleep(0.5)

    if not results:
        raise HTTPException(500, "All benchmark runs failed. Is Ollama running?")

    avg = {
        "power_w":          statistics.mean(r["avg_power_w"]       for r in results),
        "tokens_per_sec":   statistics.mean(r["tokens_per_sec"]    for r in results),
        "latency_ms":       statistics.mean(r["wall_time_s"] * 1000 for r in results),
        "energy_j_per_tok": statistics.mean(r["energy_j_per_tok"]  for r in results),
        "energy_j":         statistics.mean(r["energy_j"]          for r in results),
        "tokens_generated": statistics.mean(r["tokens_generated"]  for r in results),
        "cpu_pct":          statistics.mean(r["avg_cpu_pct"]       for r in results),
        "gpu_util":         statistics.mean(r["avg_gpu_util"]      for r in results),
        "gpu_power_w":      statistics.mean(r["avg_gpu_power_w"]   for r in results),
        "model":            _active_ollama_model,
        "measured_at":      datetime.now().isoformat(),
        "n_runs":           len(results),
        "benchmark_prompt": prompt,
    }

    _baseline_snapshot.update(avg)
    _phase = "benchmarked"

    return {
        "success":  True,
        "baseline": avg,
        "phase":    _phase,
        "message": (
            f"Baseline measured from {len(results)} runs. "
            f"Power: {avg['power_w']:.1f} W | "
            f"Speed: {avg['tokens_per_sec']:.1f} tok/s | "
            f"Energy: {avg['energy_j']:.3f} J per call"
        ),
    }


@app.post("/api/ollama/apply-optimization")
def apply_ollama_optimization(payload: dict):
    global _ollama_options, _ollama_profile, _phase

    if not _active_ollama_model:
        raise HTTPException(400, "No Ollama model selected.")

    strategy = payload.get("strategy")
    if not strategy:
        raise HTTPException(400, "strategy is required.")

    if _phase not in ["benchmarked", "optimized"]:
        raise HTTPException(400, "Run /api/benchmark first to establish baseline.")

    if strategy == "token_limit":
        limit = int(payload.get("limit", 80))
        _ollama_options = apply_token_limit(_ollama_options, limit)
        _ollama_profile = f"token_limit_{limit}"
        description = f"Output capped at {limit} tokens."

    elif strategy == "context_reduction":
        ctx = int(payload.get("ctx", 512))
        _ollama_options = apply_context_reduction(_ollama_options, ctx)
        _ollama_profile = f"ctx_{ctx}"
        description = f"Context window reduced to {ctx} tokens."

    elif strategy == "gpu_lean":
        _ollama_options = apply_gpu_routing(_ollama_options, "gpu_lean")
        _ollama_profile = "gpu_lean"
        description = "All transformer layers on GPU. Min CPU overhead."

    elif strategy == "cpu_lean":
        _ollama_options = apply_gpu_routing(_ollama_options, "cpu_lean")
        _ollama_profile = "cpu_lean"
        description = "All layers on CPU. Low peak wattage."

    elif strategy == "power_cap":
        cap_w = int(payload.get("cap_w", 50))  
        cap_result = apply_gpu_power_cap(cap_w)
        if not cap_result["success"]:
            raise HTTPException(400, f"Power cap failed: {cap_result['error']}")
        _ollama_profile = f"power_cap_{cap_result['actual_w']}w"
        description = f"GPU capped at {cap_result['actual_w']}W (was {cap_result.get('max_w', 'unknown')}W)"

    elif strategy == "quant_switch":
        variant   = payload.get("variant", "q4_0")
        base      = _active_ollama_model.split(":")[0]
        new_model = f"{base}:{variant}"
        cmd       = f"ollama pull {base}:{variant}"
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            available   = [m["name"] for m in r.json().get("models", [])]
            already_have = new_model in available
        except Exception:
            already_have = False

        return {
            "success":           True,
            "strategy":          "quant_switch",
            "target_model":      new_model,
            "already_downloaded": already_have,
            "pull_command":      cmd,
            "instructions": (
                f"Run `{cmd}` in your terminal, then call "
                f"/api/ollama/set-active with model_name={new_model}."
            ) if not already_have else "Model already downloaded.",
        }
    else:
        raise HTTPException(400, f"Unknown strategy: {strategy}.")

    bench_prompt = _baseline_snapshot.get("benchmark_prompt", BENCHMARK_PROMPT)
    after = measure_ollama_inference(
        model=_active_ollama_model,
        prompt=bench_prompt,
        options=_ollama_options,
    )

    if not after.get("success"):
        raise HTTPException(500, f"Inference failed: {after.get('error')}")

    baseline_e = _baseline_snapshot["energy_j"]
    after_e    = after.get("energy_j", baseline_e)
    saving_pct = round((baseline_e - after_e) / max(baseline_e, 1e-9) * 100, 1)

    _phase = "optimized"

    # ==============================================================
    # BAKE INTO OLLAMA NATIVELY (OVERWRITE THE MODEL)
    # ==============================================================
    if strategy not in ["power_cap"]: 
        try:
            modelfile_lines = [f"FROM {_active_ollama_model}"]
            for key, value in _ollama_options.items():
                modelfile_lines.append(f"PARAMETER {key} {value}")
                
            requests.post(
                "http://localhost:11434/api/create",
                json={
                    "name": _active_ollama_model, 
                    "modelfile": "\n".join(modelfile_lines),
                    "stream": False
                },
                timeout=30
            )
            description += f" (Permanently baked into {_active_ollama_model})"
        except Exception as e:
            print(f"[WARNING] Failed to bake model into Ollama: {e}")
    # ==============================================================

    return {
        "success":         True,
        "strategy":        strategy,
        "profile":         _ollama_profile,
        "options_applied": _ollama_options,
        "description":     description,
        "phase":           _phase,
        "before": {
            "power_w":     _baseline_snapshot["power_w"],
            "energy_j":    _baseline_snapshot["energy_j"],
            "tok_per_sec": _baseline_snapshot["tokens_per_sec"],
            "cpu_pct":     _baseline_snapshot.get("cpu_pct", 0),
            "gpu_util":    _baseline_snapshot.get("gpu_util", 0),
            "gpu_power_w": _baseline_snapshot.get("gpu_power_w", 0),
        },
        "after": {
            "power_w":     after.get("avg_power_w"),
            "energy_j":    after_e,
            "tok_per_sec": after.get("tokens_per_sec"),
            "cpu_pct":     after.get("avg_cpu_pct", 0),
            "gpu_util":    after.get("avg_gpu_util", 0),
            "gpu_power_w": after.get("avg_gpu_power_w", 0),
        },
        "real_saving_pct": saving_pct,
        "message": f"Real measurement: {saving_pct:.1f}% energy reduction vs baseline.",
    }


@app.post("/api/ollama/reset-optimization")
def reset_optimization():
    global _ollama_options, _ollama_profile, _phase
    _ollama_options = {}
    _ollama_profile = "none"
    reset_gpu_power_cap()
    if _phase == "optimized":
        _phase = "benchmarked"
    return {
        "success": True,
        "phase":   _phase,
        "message": "All optimizations cleared. Back to default Ollama settings.",
    }


@app.get("/api/ollama/quant-variants")
def quant_variants():
    if not _active_ollama_model:
        raise HTTPException(400, "No model selected.")
    return get_quantization_variants(_active_ollama_model)


@app.post("/api/telemetry/toggle-load")
def toggle_stress_test(payload: dict):
    global _is_stress_testing, _stress_thread

    enable = payload.get("enable", False)

    if enable and not _is_stress_testing:
        if not _active_ollama_model and not _pytorch_model:
            raise HTTPException(400, "Load a model first.")
        if _baseline_snapshot["power_w"] is None and _active_ollama_model:
            raise HTTPException(400, "Run /api/benchmark first to establish baseline.")
        _is_stress_testing = True
        _stress_thread = threading.Thread(target=_stress_worker, daemon=True)
        _stress_thread.start()
        return {"status": "started", "options_in_use": _ollama_options}

    elif not enable and _is_stress_testing:
        _is_stress_testing = False
        if _stress_thread:
            _stress_thread.join(timeout=15)
        return {"status": "stopped"}

    return {"status": "unchanged"}


def _stress_worker():
    prompts = [
        "What is 2 + 2?",
        "Name a color.",
        "What is the capital of France?",
        "Say hello in Spanish.",
        "What year did WW2 end?",
    ]
    idx = 0
    while _is_stress_testing:
        try:
            payload = {
                "model":   _active_ollama_model or _pytorch_model_name,
                "prompt":  prompts[idx % len(prompts)],
                "stream":  False,
                "options": _ollama_options,
            }
            requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=30,
            )
            idx += 1
            time.sleep(0.2)
        except Exception as e:
            print(f"[STRESS] {e}")
            time.sleep(1)


@app.get("/api/session")
def get_session():
    recent = list(_telemetry_history)[-60:] if _telemetry_history else []
    powers = [t["total_power_w"] for t in recent if t.get("total_power_w")]
    avg_p  = statistics.mean(powers) if powers else 0

    return {
        "active_model":      _active_ollama_model,
        "profile":           _ollama_profile,
        "options":           _ollama_options,
        "phase":             _phase,
        "baseline":          _baseline_snapshot,
        "avg_power_1min_w":  round(avg_p, 2),
        "carbon_gco2_kwh":   get_carbon(),
        "telemetry_points":  len(_telemetry_history),
    }


@app.post("/api/load-local-model")
async def load_local_model(model_file: UploadFile = File(...)):
    global _pytorch_model, _pytorch_tokenizer, _pytorch_model_name
    global _active_ollama_model, _phase

    _active_ollama_model = None
    _phase = "idle"
    if not TORCH_OK:
        return {"success": False, "error": "PyTorch is required for local models. Install with: pip install torch transformers"}
    filename = model_file.filename
    ext      = os.path.splitext(filename)[1].lower()
    tmp_dir  = tempfile.mkdtemp()

    try:
        file_path = os.path.join(tmp_dir, filename)
        with open(file_path, "wb") as f:
            shutil.copyfileobj(model_file.file, f)

        if ext == ".zip":
            extract_dir = os.path.join(tmp_dir, "extracted")
            os.makedirs(extract_dir)
            with zipfile.ZipFile(file_path, "r") as z:
                z.extractall(extract_dir)
            if not os.path.exists(os.path.join(extract_dir, "config.json")):
                subdirs = [
                    d for d in os.listdir(extract_dir)
                    if os.path.isdir(os.path.join(extract_dir, d))
                ]
                if subdirs:
                    extract_dir = os.path.join(extract_dir, subdirs[0])
            _pytorch_tokenizer = AutoTokenizer.from_pretrained(extract_dir)
            _pytorch_model     = AutoModelForCausalLM.from_pretrained(
                extract_dir, torch_dtype=torch.float32, device_map="auto"
            )
            _pytorch_model_name = filename.replace(".zip", "")
            params = sum(p.numel() for p in _pytorch_model.parameters())
            return {
                "success":    True,
                "model_name": _pytorch_model_name,
                "params_b":   round(params / 1e9, 4),
            }

        elif ext in [".pt", ".pth", ".bin", ".safetensors"]:
            sd     = torch.load(file_path, map_location="cpu", weights_only=True)
            params = sum(
                v.numel() for v in sd.values() if isinstance(v, torch.Tensor)
            )
            _pytorch_model_name = filename
            return {
                "success":    True,
                "model_name": filename,
                "params_b":   round(params / 1e9, 4),
                "note":       "State dict loaded. Tokenizer not available for raw weights.",
            }

        else:
            return {
                "success": False,
                "error":   f"Unsupported: {ext}. Use .zip (full HF model) or .pt/.pth/.safetensors",
            }

    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/api/load-hf-model")
async def load_hf_model(payload: dict):
    global _pytorch_model, _pytorch_tokenizer, _pytorch_model_name
    global _active_ollama_model, _phase

    _active_ollama_model = None
    _phase = "idle"
    if not TORCH_OK:
        return {"success": False, "error": "PyTorch is required for HuggingFace models. Install with: pip install torch transformers"}
    model_id = payload.get("model_id", "").strip()
    if not model_id:
        raise HTTPException(400, "model_id is required")

    def _download():
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32,
            device_map="auto", low_cpu_mem_usage=True,
        )
        return tok, mdl

    try:
        loop = asyncio.get_event_loop()
        t0   = time.time()
        tok, mdl = await loop.run_in_executor(_executor, _download)
        _pytorch_tokenizer  = tok
        _pytorch_model      = mdl
        _pytorch_model_name = model_id
        params = sum(p.numel() for p in mdl.parameters())
        return {
            "success":     True,
            "model_name":  model_id,
            "params_b":    round(params / 1e9, 4),
            "load_time_s": round(time.time() - t0, 1),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/emissions")
def get_emissions():
    return {
        "records": list(_telemetry_history),
        "count":   len(_telemetry_history),
    }

if __name__ == "__main__":
    import uvicorn
    print("\n[OK] Zerotrace AI - Global Telemetry Active.")
    print("     Start Ollama first: ollama serve")
    print("     Then open http://localhost:8000\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)