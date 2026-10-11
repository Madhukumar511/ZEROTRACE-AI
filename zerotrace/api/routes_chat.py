import time
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from zerotrace.core.state import state
from zerotrace.core.embedder import embedder
from zerotrace.core.emissions import get_carbon
from zerotrace.compression.prompt_optimizer import PromptOptimizer
from zerotrace.compression.tokenizer import estimate_token_count
from zerotrace.telemetry.aggregator import get_total_system_power
from zerotrace.models.ollama import ollama_client
from zerotrace.models.cloud import cloud_client, CLOUD_PROVIDERS

router = APIRouter(prefix="/api", tags=["chat"])


class ChatRequest(BaseModel):
    message: str = Field(..., description="User prompt or question")
    provider: str = Field("ollama", description="Provider: 'ollama', 'gemini', 'openai', 'groq', 'anthropic', 'custom'")
    model: Optional[str] = Field(None, description="Model identifier (e.g. 'llama3:latest', 'gemini-1.5-flash', 'gpt-4o-mini')")
    api_key: Optional[str] = Field(None, description="API Key for cloud provider (optional if configured in environment)")
    base_url: Optional[str] = Field(None, description="Custom base URL for OpenAI-compatible endpoints")
    use_caveman: bool = Field(True, description="Enable Caveman prompt compression to save tokens and energy")
    caveman_intensity: str = Field("full", description="Caveman compression intensity: 'lite', 'full', 'ultra'")
    use_cache: bool = Field(True, description="Enable Semantic Vector Cache to answer with 0W on similar past queries")
    cache_threshold: float = Field(0.88, description="Cosine similarity threshold for cache hit (0.0 to 1.0)")


@router.get("/chat/providers")
def get_providers():
    """Returns available local and cloud providers with their model choices."""
    ollama_online = ollama_client.is_online()
    ollama_models = ollama_client.list_models() if ollama_online else []
    ollama_model_names = [m["name"] for m in ollama_models]

    cloud_list = []
    for prov_id, prov_info in CLOUD_PROVIDERS.items():
        has_key = bool(cloud_client.resolve_key(prov_id))
        cloud_list.append({
            "id": prov_id,
            "name": prov_info["name"],
            "default_model": prov_info["default_model"],
            "models": prov_info["models"],
            "has_key": has_key,
        })

    return {
        "success": True,
        "local": {
            "id": "ollama",
            "name": "Local Ollama",
            "online": ollama_online,
            "models": ollama_model_names,
            "default_model": ollama_model_names[0] if ollama_model_names else "llama3:latest",
        },
        "cloud": cloud_list,
        "cache_size": state.vector_cache.size(),
        "total_cache_hits": state.total_cache_hits,
    }


@router.get("/cache/stats")
def get_cache_stats():
    """Returns telemetry and statistics from the Semantic Vector Cache."""
    return {
        "success": True,
        "cached_queries_count": state.vector_cache.size(),
        "dimension": state.vector_cache.dimension,
        "total_cache_hits": state.total_cache_hits,
        "total_energy_saved_j": round(state.total_energy_saved_j, 3),
        "total_tokens_saved": state.total_tokens_saved,
    }


@router.post("/chat")
def chat_endpoint(req: ChatRequest):
    """
    Unified Green Inference Gateway:
    1. Checks 384-D Semantic Vector Cache (0 Watts / 0 Tokens on hit).
    2. Runs Caveman Token Compression on prompt (~65% token drop).
    3. Dispatches to selected Local or Cloud LLM.
    4. Auto-caches answer in VectorDatastore for future zero-energy recall.
    """
    user_query = req.message.strip()
    if not user_query:
        raise HTTPException(400, "message cannot be empty")

    provider = req.provider.lower().strip()
    power_now = get_total_system_power(0.0)
    current_w = power_now["total_w"] if power_now["total_w"] > 0 else 65.0
    carbon_kwh = get_carbon()
    t_start = time.time()

    # ─────────────────────────────────────────────────────────────
    # STEP 1: Semantic Vector Cache Check
    # ─────────────────────────────────────────────────────────────
    query_vec = None
    if req.use_cache:
        query_vec = embedder.embed(user_query)
        neighbors = state.vector_cache.search(query_vec, top_k=3)
        cache_eval = state.knn_combiner.evaluate(
            neighbors=neighbors,
            default_power_w=current_w,
            est_llm_time_s=2.5,
        )

        if cache_eval["hit"] and cache_eval["similarity_score"] >= req.cache_threshold:
            # Cache Hit! 100% compute avoided.
            state.total_cache_hits += 1
            energy_saved_j = cache_eval.get("energy_saved_j", 65.0 * 2.5)
            # Estimate tokens avoided (~200 tokens output + prompt tokens)
            tokens_saved = estimate_token_count(user_query) + 200
            carbon_saved_g = (energy_saved_j / 3_600_000.0) * carbon_kwh

            state.total_energy_saved_j += energy_saved_j
            state.total_tokens_saved += tokens_saved

            best_record = cache_eval["best_match"]
            return {
                "success": True,
                "cache_hit": True,
                "answer": best_record.get("answer", ""),
                "matched_query": best_record.get("query", ""),
                "similarity_score": cache_eval["similarity_score"],
                "provider": "vector_cache (ZeroTrace)",
                "model": "Semantic Vector Memory (384-D)",
                "tokens_saved": tokens_saved,
                "energy_saved_j": energy_saved_j,
                "carbon_saved_g": round(carbon_saved_g, 6),
                "duration_ms": round((time.time() - t_start) * 1000.0, 2),
                "metrics": {
                    "energy_cost_j": 0.0,
                    "gpu_watts": 0.0,
                    "cloud_api_cost": 0.0,
                    "status": "🟢 100% Zero-Watt Cache Hit",
                },
            }

    # ─────────────────────────────────────────────────────────────
    # STEP 2: Caveman Prompt Compression
    # ─────────────────────────────────────────────────────────────
    prompt_to_send = user_query
    prompt_tokens_saved = 0
    comp_metrics = None

    if req.use_caveman:
        comp_res = PromptOptimizer.optimize(
            prompt=user_query,
            intensity=req.caveman_intensity,
            power_w=current_w,
            carbon_gco2_kwh=carbon_kwh,
        )
        prompt_to_send = comp_res["compressed_prompt"]
        comp_metrics = comp_res["metrics"]
        prompt_tokens_saved = comp_metrics.get("tokens_saved", 0)
        state.total_tokens_saved += prompt_tokens_saved

    # ─────────────────────────────────────────────────────────────
    # STEP 3: Dispatch to Model (Local or Cloud)
    # ─────────────────────────────────────────────────────────────
    answer_text = ""
    model_used = req.model or "default"
    cloud_usage = {}

    if provider == "ollama":
        if not ollama_client.is_online():
            raise HTTPException(503, "Local Ollama is not running. Please start it with 'ollama serve' or select a Cloud Provider (Gemini, Groq, OpenAI).")

        model_to_use = req.model or state.active_ollama_model or "llama3:latest"
        model_used = model_to_use

        res = ollama_client.generate(
            model=model_to_use,
            prompt=prompt_to_send,
            options=state.ollama_options,
        )
        if not res.get("success"):
            raise HTTPException(502, f"Ollama generation failed: {res.get('error')}")

        answer_text = res.get("response", "")

    elif provider in CLOUD_PROVIDERS:
        res = cloud_client.generate(
            provider=provider,
            prompt=prompt_to_send,
            model=req.model,
            api_key=req.api_key,
            base_url=req.base_url,
        )
        if not res.get("success"):
            raise HTTPException(400, res.get("error", f"{provider} generation failed"))

        answer_text = res.get("text", "")
        model_used = res.get("model", req.model or "cloud")
        cloud_usage = {
            "prompt_tokens": res.get("prompt_tokens", 0),
            "completion_tokens": res.get("completion_tokens", 0),
            "total_tokens": res.get("total_tokens", 0),
        }
    else:
        raise HTTPException(400, f"Unknown provider: '{provider}'. Choose from: 'ollama', 'gemini', 'openai', 'groq', 'anthropic', 'custom'.")

    duration_s = max(time.time() - t_start, 0.01)

    # ─────────────────────────────────────────────────────────────
    # STEP 4: Auto-Cache Verified Response
    # ─────────────────────────────────────────────────────────────
    if answer_text and req.use_cache:
        try:
            if query_vec is None:
                query_vec = embedder.embed(user_query)
            state.vector_cache.add(
                query=user_query,
                answer=answer_text,
                vector=query_vec,
                metadata={"provider": provider, "model": model_used, "timestamp": time.time()},
            )
        except Exception:
            pass

    # Physical energy & carbon math:
    energy_saved_j = comp_metrics.get("energy_saved_j", 0.0) if comp_metrics else 0.0
    state.total_energy_saved_j += energy_saved_j
    carbon_saved_g = (energy_saved_j / 3_600_000.0) * carbon_kwh

    return {
        "success": True,
        "cache_hit": False,
        "answer": answer_text,
        "prompt_original": user_query,
        "prompt_sent": prompt_to_send,
        "provider": provider,
        "model": model_used,
        "tokens_saved": prompt_tokens_saved,
        "energy_saved_j": round(energy_saved_j, 4),
        "carbon_saved_g": round(carbon_saved_g, 6),
        "duration_s": round(duration_s, 2),
        "cloud_usage": cloud_usage,
        "metrics": {
            "caveman_compressed": req.use_caveman,
            "tokens_saved": prompt_tokens_saved,
            "status": f"⚡ Processed via {provider.upper()} ({model_used})",
        },
    }
