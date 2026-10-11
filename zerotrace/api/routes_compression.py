from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from zerotrace.compression.caveman import CavemanCompressor
from zerotrace.compression.tokenizer import estimate_token_count, calculate_savings
from zerotrace.compression.prompt_optimizer import PromptOptimizer
from zerotrace.core.emissions import get_carbon
from zerotrace.core.state import state
from zerotrace.telemetry.aggregator import get_total_system_power

router = APIRouter(prefix="/api", tags=["compression"])


class CompressRequest(BaseModel):
    text: str = Field(..., description="Text prompt to compress using Caveman heuristics")
    intensity: str = Field("full", description="Compression intensity: 'lite', 'full' (~65%), 'ultra'")


class BenchmarkPromptRequest(BaseModel):
    prompt: Optional[str] = Field(None, description="Prompt to evaluate through Caveman compression")
    intensity: str = Field("full", description="Compression intensity: 'lite', 'full', 'ultra'")


@router.post("/compress")
def compress_endpoint(req: CompressRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(400, "text cannot be empty")

    power = get_total_system_power(0.0)
    current_w = power["total_w"] if power["total_w"] > 0 else 65.0

    result = PromptOptimizer.optimize(
        prompt=req.text,
        intensity=req.intensity,
        power_w=current_w,
        tok_per_sec=25.0,
        carbon_gco2_kwh=get_carbon(),
    )

    # Accumulate global session savings
    metrics = result["metrics"]
    state.total_tokens_saved += metrics.get("tokens_saved", 0)
    state.total_energy_saved_j += metrics.get("energy_saved_j", 0.0)

    return {
        "success": True,
        "intensity": result["intensity"],
        "original_text": result["original_prompt"],
        "compressed_text": result["compressed_prompt"],
        "system_prompt": result["system_prompt"],
        "metrics": metrics,
    }


@router.post("/caveman/benchmark")
def caveman_benchmark_endpoint(req: BenchmarkPromptRequest):
    prompt = req.prompt or (
        "Explain how transformer neural networks work. "
        "Cover self-attention, positional encoding, feed-forward layers, and training. "
        "Give a detailed explanation with at least 250 words."
    )

    result = PromptOptimizer.optimize(
        prompt=prompt,
        intensity=req.intensity,
        carbon_gco2_kwh=get_carbon(),
    )
    return {
        "success": True,
        "benchmark": result,
    }


@router.get("/caveman/stats")
def caveman_stats():
    return {
        "caveman_active": state.caveman_active,
        "intensity": state.caveman_intensity,
        "total_tokens_saved": state.total_tokens_saved,
        "total_energy_saved_j": round(state.total_energy_saved_j, 4),
        "total_energy_saved_kwh": round(state.total_energy_saved_j / 3_600_000, 9),
        "total_co2_saved_g": round((state.total_energy_saved_j / 3_600_000) * get_carbon(), 9),
    }
