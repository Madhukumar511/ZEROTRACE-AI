from fastapi import APIRouter, HTTPException
from zerotrace.core.state import state
from zerotrace.config import BENCHMARK_PROMPT
from zerotrace.models.ollama import ollama_client
from zerotrace.telemetry.hardware import apply_gpu_power_cap, reset_gpu_power_cap
from zerotrace.core.benchmark import measure_ollama_inference, run_benchmark_suite
from zerotrace.core.optimizer import (
    apply_token_limit,
    apply_context_reduction,
    apply_gpu_routing,
    apply_caveman_compression_optimization,
    bake_ollama_optimization,
)
from zerotrace.compression.caveman import CavemanCompressor

router = APIRouter(tags=["ollama"])


@router.get("/api/ollama/models")
def get_ollama_models():
    models = ollama_client.list_models()
    if models:
        return {"success": True, "models": models}
    return {
        "success": False,
        "models": [],
        "error": "Ollama not running. Start with: ollama serve",
    }


@router.post("/api/ollama/set-active")
def set_active_ollama(payload: dict):
    model_name = payload.get("model_name", "").strip()
    if not model_name:
        raise HTTPException(400, "model_name is required")

    models = ollama_client.list_models()
    available = [m["name"] for m in models]
    if available and model_name not in available:
        raise HTTPException(
            404,
            f"Model '{model_name}' not found in Ollama. Available: {available}",
        )

    state.set_active_model(model_name)
    reset_gpu_power_cap()

    return {
        "success": True,
        "active": model_name,
        "phase": state.phase,
        "message": "Model set. Run /api/benchmark to measure real baseline.",
    }


@router.post("/api/benchmark")
def run_benchmark(payload: dict = {}):
    if not state.active_ollama_model:
        raise HTTPException(400, "No Ollama model selected. Call /api/ollama/set-active first.")

    if state.phase not in ["model_selected", "benchmarked", "optimized"]:
        raise HTTPException(400, "Invalid phase. Call /api/ollama/set-active first.")

    prompt = payload.get("prompt", BENCHMARK_PROMPT)
    n_runs = int(payload.get("n_runs", 3))

    res = run_benchmark_suite(model=state.active_ollama_model, prompt=prompt, n_runs=n_runs)
    if not res.get("success"):
        raise HTTPException(500, res.get("error", "Benchmark failed"))

    return res


@router.post("/api/ollama/apply-optimization")
def apply_ollama_optimization(payload: dict):
    if not state.active_ollama_model:
        raise HTTPException(400, "No Ollama model selected.")

    strategy = payload.get("strategy")
    if not strategy:
        raise HTTPException(400, "strategy is required.")

    if state.phase not in ["benchmarked", "optimized"]:
        raise HTTPException(400, "Run /api/benchmark first to establish baseline.")

    system_prompt = None

    if strategy == "token_limit":
        limit = int(payload.get("limit", 80))
        state.ollama_options = apply_token_limit(state.ollama_options, limit)
        state.ollama_profile = f"token_limit_{limit}"
        description = f"Output capped at {limit} tokens."

    elif strategy == "context_reduction":
        ctx = int(payload.get("ctx", 512))
        state.ollama_options = apply_context_reduction(state.ollama_options, ctx)
        state.ollama_profile = f"ctx_{ctx}"
        description = f"Context window reduced to {ctx} tokens."

    elif strategy == "gpu_lean":
        state.ollama_options = apply_gpu_routing(state.ollama_options, "gpu_lean")
        state.ollama_profile = "gpu_lean"
        description = "All transformer layers on GPU. Min CPU overhead."

    elif strategy == "cpu_lean":
        state.ollama_options = apply_gpu_routing(state.ollama_options, "cpu_lean")
        state.ollama_profile = "cpu_lean"
        description = "All layers on CPU. Low peak wattage."

    elif strategy == "power_cap":
        cap_w = int(payload.get("cap_w", 50))
        cap_result = apply_gpu_power_cap(cap_w)
        if not cap_result["success"]:
            raise HTTPException(400, f"Power cap failed: {cap_result['error']}")
        state.ollama_profile = f"power_cap_{cap_result['actual_w']}w"
        description = f"GPU capped at {cap_result['actual_w']}W (was {cap_result.get('max_w', 'unknown')}W)"

    elif strategy in ["caveman", "caveman_compression"]:
        intensity = payload.get("intensity", "full")
        state.ollama_options, state.ollama_profile, description = apply_caveman_compression_optimization(
            state.ollama_options, intensity=intensity
        )
        state.caveman_active = True
        state.caveman_intensity = intensity
        system_prompt = CavemanCompressor.get_system_prompt(intensity=intensity)

    elif strategy == "quant_switch":
        variant = payload.get("variant", "q4_0")
        base = state.active_ollama_model.split(":")[0]
        new_model = f"{base}:{variant}"
        cmd = f"ollama pull {base}:{variant}"
        models = ollama_client.list_models()
        available = [m["name"] for m in models]
        already_have = new_model in available

        return {
            "success": True,
            "strategy": "quant_switch",
            "target_model": new_model,
            "already_downloaded": already_have,
            "pull_command": cmd,
            "instructions": (
                f"Run `{cmd}` in your terminal, then call "
                f"/api/ollama/set-active with model_name={new_model}."
            ) if not already_have else "Model already downloaded.",
        }
    else:
        raise HTTPException(400, f"Unknown strategy: {strategy}.")

    # Evaluate inference with new optimization settings
    bench_prompt = state.baseline_snapshot.get("benchmark_prompt", BENCHMARK_PROMPT)
    if state.caveman_active:
        bench_prompt = CavemanCompressor.compress(bench_prompt, intensity=state.caveman_intensity)

    after = measure_ollama_inference(
        model=state.active_ollama_model,
        prompt=bench_prompt,
        options=state.ollama_options,
    )

    if not after.get("success"):
        raise HTTPException(500, f"Inference failed: {after.get('error')}")

    baseline_e = state.baseline_snapshot["energy_j"]
    after_e = after.get("energy_j", baseline_e)
    saving_pct = round((baseline_e - after_e) / max(baseline_e, 1e-9) * 100, 1)

    state.phase = "optimized"

    # Bake into Ollama Modelfile if not hardware power cap
    if strategy not in ["power_cap"]:
        baked = bake_ollama_optimization(
            model_name=state.active_ollama_model,
            options=state.ollama_options,
            system_prompt=system_prompt,
        )
        if baked:
            description += f" (Permanently baked into {state.active_ollama_model})"

    return {
        "success": True,
        "strategy": strategy,
        "profile": state.ollama_profile,
        "options_applied": state.ollama_options,
        "description": description,
        "phase": state.phase,
        "before": {
            "power_w": state.baseline_snapshot["power_w"],
            "energy_j": state.baseline_snapshot["energy_j"],
            "tok_per_sec": state.baseline_snapshot["tokens_per_sec"],
            "cpu_pct": state.baseline_snapshot.get("cpu_pct", 0),
            "gpu_util": state.baseline_snapshot.get("gpu_util", 0),
            "gpu_power_w": state.baseline_snapshot.get("gpu_power_w", 0),
        },
        "after": {
            "power_w": after.get("avg_power_w"),
            "energy_j": after_e,
            "tok_per_sec": after.get("tokens_per_sec"),
            "cpu_pct": after.get("avg_cpu_pct", 0),
            "gpu_util": after.get("avg_gpu_util", 0),
            "gpu_power_w": after.get("avg_gpu_power_w", 0),
        },
        "real_saving_pct": saving_pct,
        "message": f"Real measurement: {saving_pct:.1f}% energy reduction vs baseline.",
    }


@router.post("/api/ollama/reset-optimization")
def reset_optimization():
    state.reset_optimizations()
    reset_gpu_power_cap()
    return {
        "success": True,
        "phase": state.phase,
        "message": "All optimizations cleared. Back to default Ollama settings.",
    }


@router.get("/api/ollama/quant-variants")
def quant_variants():
    if not state.active_ollama_model:
        raise HTTPException(400, "No model selected.")
    return ollama_client.get_quantization_variants(state.active_ollama_model)
