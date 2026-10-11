import time
import threading
import statistics
from datetime import datetime
from typing import Dict, Any, List, Optional

from zerotrace.config import BENCHMARK_PROMPT, DEFAULT_BENCHMARK_RUNS
from zerotrace.telemetry.process import get_isolated_ai_cpu
from zerotrace.telemetry.hardware import get_gpu_telemetry, get_cpu_power_estimate
from zerotrace.telemetry.zeus_monitor import ZEUS_OK, zeus_manager
from zerotrace.core.emissions import get_carbon
from zerotrace.core.state import state
from zerotrace.models.ollama import ollama_client


def measure_ollama_inference(model: str, prompt: str, options: Optional[dict] = None) -> Dict[str, Any]:
    """
    Executes a real inference call against Ollama while high-frequency background
    hardware telemetry threads sample true CPU & GPU power dissipation.
    """
    samples = []
    stop_flag = threading.Event()

    def _sampler():
        get_isolated_ai_cpu()  # Prime the sweeper
        while not stop_flag.is_set():
            time.sleep(0.2)
            cpu = get_isolated_ai_cpu()
            gpu = get_gpu_telemetry()
            cpu_w = get_cpu_power_estimate(cpu)
            total_w = round(cpu_w + gpu["gpu_power_w"], 2)
            samples.append({
                "total_w": total_w,
                "cpu_pct": cpu,
                "gpu_util": gpu["gpu_util_pct"],
                "gpu_power_w": gpu["gpu_power_w"],
            })

    sampler = threading.Thread(target=_sampler, daemon=True, name="bench-sampler")
    sampler.start()
    state.inference_active = True
    t_start = time.perf_counter()

    window_key = f"inference_{int(time.time() * 1000)}"
    if ZEUS_OK:
        zeus_manager.begin_window(window_key)

    data = {}
    try:
        data = ollama_client.generate(model=model, prompt=prompt, options=options)
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        state.inference_active = False
        stop_flag.set()
        sampler.join(timeout=5)
        zeus_energy_j = zeus_manager.end_window(window_key) if ZEUS_OK else None

    wall_s = time.perf_counter() - t_start

    if not samples:
        return {"success": False, "error": "No hardware samples collected — inference was too fast"}

    avg_power_w = statistics.mean(s["total_w"] for s in samples)
    avg_cpu_pct = statistics.mean(s["cpu_pct"] for s in samples)
    avg_gpu_util = statistics.mean(s["gpu_util"] for s in samples)
    avg_gpu_pw = statistics.mean(s["gpu_power_w"] for s in samples)
    energy_j = avg_power_w * wall_s

    final_energy_j = zeus_energy_j if zeus_energy_j is not None else energy_j

    eval_count = data.get("eval_count", 0)
    eval_dur_ns = data.get("eval_duration", 1)
    prompt_count = data.get("prompt_eval_count", 0)
    tok_per_sec = (eval_count / (eval_dur_ns / 1e9)) if eval_dur_ns > 0 else 0.0

    return {
        "success": True,
        "model": model,
        "options_used": options or {},
        "wall_time_s": round(wall_s, 3),
        "tokens_generated": eval_count,
        "prompt_tokens": prompt_count,
        "tokens_per_sec": round(tok_per_sec, 2),
        "avg_power_w": round(avg_power_w, 2),
        "avg_cpu_pct": round(avg_cpu_pct, 1),
        "avg_gpu_util": round(avg_gpu_util, 1),
        "avg_gpu_power_w": round(avg_gpu_pw, 2),
        "energy_j": round(final_energy_j, 4),
        "energy_source": "zeus_nvml" if zeus_energy_j is not None else "power_x_time_estimate",
        "energy_j_per_tok": round(final_energy_j / max(eval_count, 1), 6),
        "energy_kwh": round(final_energy_j / 3_600_000, 9),
        "co2_g": round((final_energy_j / 3_600_000) * get_carbon(), 9),
        "sample_count": len(samples),
        "response_preview": data.get("response", "")[:200],
    }


def run_benchmark_suite(model: str, prompt: Optional[str] = None, n_runs: int = DEFAULT_BENCHMARK_RUNS) -> Dict[str, Any]:
    """Runs n_runs inference executions to compute statistically sound baseline metrics."""
    bench_prompt = prompt or BENCHMARK_PROMPT
    results = []

    for _ in range(n_runs):
        r = measure_ollama_inference(
            model=model,
            prompt=bench_prompt,
            options={"num_predict": 250},
        )
        if r.get("success"):
            results.append(r)
        time.sleep(0.5)

    if not results:
        return {"success": False, "error": "All benchmark runs failed. Is Ollama running?"}

    avg = {
        "power_w": statistics.mean(r["avg_power_w"] for r in results),
        "tokens_per_sec": statistics.mean(r["tokens_per_sec"] for r in results),
        "latency_ms": statistics.mean(r["wall_time_s"] * 1000 for r in results),
        "energy_j_per_tok": statistics.mean(r["energy_j_per_tok"] for r in results),
        "energy_j": statistics.mean(r["energy_j"] for r in results),
        "tokens_generated": statistics.mean(r["tokens_generated"] for r in results),
        "cpu_pct": statistics.mean(r["avg_cpu_pct"] for r in results),
        "gpu_util": statistics.mean(r["avg_gpu_util"] for r in results),
        "gpu_power_w": statistics.mean(r["avg_gpu_power_w"] for r in results),
        "model": model,
        "measured_at": datetime.now().isoformat(),
        "n_runs": len(results),
        "benchmark_prompt": bench_prompt,
    }

    state.update_baseline(avg)
    return {
        "success": True,
        "baseline": avg,
        "phase": state.phase,
        "message": (
            f"Baseline measured from {len(results)} runs. "
            f"Power: {avg['power_w']:.1f} W | "
            f"Speed: {avg['tokens_per_sec']:.1f} tok/s | "
            f"Energy: {avg['energy_j']:.3f} J per call"
        ),
    }
