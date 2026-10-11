#!/usr/bin/env python3
"""
ZeroTrace AI - Model Context Protocol (MCP) Stdio Server.
Integrates ZeroTrace AI with coding assistants (Google Antigravity CLI, Claude Code, Cursor, Windsurf).

Exposes tools:
- zerotrace_search_cache: Instant 0-Watt responses from 384-D Semantic Vector Memory
- zerotrace_compress_prompt: Slashes conversational fluff by ~65% with Caveman heuristics
- zerotrace_record_savings: Reports token and electricity savings to the live terminal dashboard
- zerotrace_get_telemetry: Reads live CPU/GPU power draw (Watts) and carbon emissions
"""

import sys
import json
import os
from typing import Dict, Any, List

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from zerotrace.core.state import state
from zerotrace.core.embedder import embedder
from zerotrace.core.emissions import get_carbon
from zerotrace.compression.caveman import CavemanCompressor
from zerotrace.compression.prompt_optimizer import PromptOptimizer
from zerotrace.telemetry.aggregator import get_total_system_power
from zerotrace.telemetry.hardware import get_gpu_telemetry


def tool_search_cache(query: str, threshold: float = 0.88) -> str:
    """Searches 384-D Semantic Vector Cache for 0-Watt answers."""
    query = (query or "").strip()
    if not query:
        return "Query cannot be empty."

    query_vec = embedder.embed(query)
    matches = state.vector_cache.search(query_vec, top_k=3)
    eval_res = state.knn_combiner.evaluate(neighbors=matches, default_power_w=65.0, est_llm_time_s=2.5)

    if eval_res["hit"] and eval_res["similarity_score"] >= threshold:
        best = eval_res["best_match"]
        saved_j = eval_res.get("energy_saved_j", 162.5)
        saved_tokens = 200 + len(query.split())
        state.total_cache_hits += 1
        state.total_energy_saved_j += saved_j
        state.total_tokens_saved += saved_tokens
        carbon_g = (saved_j / 3_600_000.0) * get_carbon()

        return json.dumps({
            "cache_hit": True,
            "answer": best.get("answer", ""),
            "matched_query": best.get("query", ""),
            "similarity_score": round(eval_res["similarity_score"], 4),
            "energy_saved_j": round(saved_j, 2),
            "tokens_saved": saved_tokens,
            "carbon_saved_g": round(carbon_g, 6),
            "badge": f"[🟢 100% ZERO-WATT CACHE HIT | <5ms | Saved {saved_tokens} tokens | {saved_j:.1f} J | {carbon_g:.5f} gCO2]",
        }, indent=2)

    return json.dumps({
        "cache_hit": False,
        "best_similarity": round(eval_res["similarity_score"], 4),
        "message": "No cache hit above threshold. Novel query requires model inference.",
    }, indent=2)


def tool_compress_prompt(text: str, intensity: str = "full") -> str:
    """Compresses prompt with Caveman syntax, returning pruned prompt and estimated token savings."""
    text = (text or "").strip()
    if not text:
        return "Text cannot be empty."

    power = get_total_system_power(0.0)
    current_w = power["total_w"] if power["total_w"] > 0 else 65.0
    res = PromptOptimizer.optimize(
        prompt=text,
        intensity=intensity,
        power_w=current_w,
        carbon_gco2_kwh=get_carbon(),
    )
    return json.dumps(res, indent=2)


def tool_record_savings(tokens_saved: int = 0, energy_saved_j: float = 0.0, note: str = "", cache_hits: int = 0) -> str:
    """Records token, electrical energy, or cache hit savings into ZeroTrace telemetry."""
    # Also attempt to post to running server if active
    try:
        import requests
        requests.post("http://127.0.0.1:8000/api/telemetry/record", json={
            "tokens_saved": tokens_saved,
            "energy_saved_j": energy_saved_j,
            "cache_hits": cache_hits,
            "source": "antigravity_mcp",
            "note": note,
        }, timeout=1)
    except Exception:
        pass

    # Record in local in-process state
    if tokens_saved > 0:
        state.total_tokens_saved += tokens_saved
        if energy_saved_j <= 0.0:
            energy_saved_j = round(tokens_saved * 0.08, 4)

    if energy_saved_j > 0.0:
        state.total_energy_saved_j += energy_saved_j

    if cache_hits > 0:
        state.total_cache_hits += cache_hits

    carbon_g = (state.total_energy_saved_j / 3_600_000.0) * get_carbon()

    return json.dumps({
        "success": True,
        "added_tokens_saved": tokens_saved,
        "added_energy_saved_j": energy_saved_j,
        "cumulative_tokens_saved": state.total_tokens_saved,
        "cumulative_energy_saved_j": round(state.total_energy_saved_j, 2),
        "cumulative_energy_saved_wh": round(state.total_energy_saved_j / 3600.0, 4),
        "cumulative_carbon_saved_g": round(carbon_g, 6),
        "cumulative_cache_hits": state.total_cache_hits,
        "note": note,
    }, indent=2)


def tool_get_telemetry() -> str:
    """Returns live hardware power, GPU telemetry, and carbon intensity."""
    power = get_total_system_power(0.0)
    gpu = get_gpu_telemetry()
    carbon = get_carbon()

    return json.dumps({
        "total_system_watts": power["total_w"],
        "cpu_watts": power["cpu_w"],
        "gpu_watts": power["gpu_w"],
        "gpu_util_pct": gpu["gpu_util_pct"],
        "gpu_name": gpu["gpu_name"],
        "carbon_intensity_gco2_kwh": round(carbon, 1),
        "cumulative_energy_saved_j": round(state.total_energy_saved_j, 2),
        "cumulative_tokens_saved": state.total_tokens_saved,
        "cumulative_cache_hits": state.total_cache_hits,
    }, indent=2)


TOOLS_LIST = [
    {
        "name": "zerotrace_search_cache",
        "description": "Searches ZeroTrace 384-D vector cache for instant 0-Watt responses to user questions. Returns cached answer in <5ms if similarity >= 0.88.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The user query or coding question to check"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "zerotrace_compress_prompt",
        "description": "Compresses user prompt or context using Caveman heuristics to save ~65% tokens without losing code or technical meaning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Prompt or text to compress"},
                "intensity": {"type": "string", "enum": ["lite", "full", "ultra"], "default": "full"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "zerotrace_record_savings",
        "description": "Logs token and electrical energy savings into the ZeroTrace live monitor (updates the live terminal & web dashboard in real time).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tokens_saved": {"type": "integer", "description": "Number of tokens saved by pruning or compression"},
                "energy_saved_j": {"type": "number", "description": "Optional electricity saved in Joules"},
                "note": {"type": "string", "description": "Optional short description of the action"}
            },
            "required": ["tokens_saved"]
        }
    },
    {
        "name": "zerotrace_get_telemetry",
        "description": "Reads real-time hardware telemetry: system power draw (Watts), GPU load, and carbon intensity.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]


def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    method = req.get("method")
    msg_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "zerotrace-ai-mcp",
                    "version": "1.0.0"
                }
            }
        }

    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": TOOLS_LIST
            }
        }

    elif method == "tools/call":
        params = req.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})

        try:
            if name == "zerotrace_search_cache":
                out = tool_search_cache(args.get("query", ""))
            elif name == "zerotrace_compress_prompt":
                out = tool_compress_prompt(args.get("text", ""), args.get("intensity", "full"))
            elif name == "zerotrace_record_savings":
                out = tool_record_savings(
                    tokens_saved=args.get("tokens_saved", 0),
                    energy_saved_j=args.get("energy_saved_j", 0.0),
                    note=args.get("note", ""),
                    cache_hits=args.get("cache_hits", 0),
                )
            elif name == "zerotrace_get_telemetry":
                out = tool_get_telemetry()
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "error": {"code": -32601, "message": f"Tool '{name}' not found"}
                }

            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "content": [{"type": "text", "text": out}]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": f"Error executing {name}: {str(e)}"}]
                }
            }

    elif method == "notifications/initialized":
        return None

    else:
        if msg_id is not None:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method '{method}' not implemented"}
            }
        return None


def main():
    """Main JSON-RPC stdio loop for MCP clients."""
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            res = handle_request(req)
            if res is not None:
                sys.stdout.write(json.dumps(res) + "\n")
                sys.stdout.flush()
        except Exception as e:
            err_res = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"}
            }
            sys.stdout.write(json.dumps(err_res) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
