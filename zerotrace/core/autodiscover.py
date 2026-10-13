"""
ZeroTrace AI - Auto-Discovery & Zero-Config Connection Engine
Automatically scans the host machine for:
1. Offline / Local LLMs (Ollama daemon, GGUF/SafeTensors/PyTorch weight files)
2. AI Developer Agents & CLIs (Google Antigravity CLI, Claude Code, Cursor)
3. Physical Telemetry Sensors (Intel RAPL, NVIDIA NVML, Zeus, psutil)
4. Cloud LLM Environment Credentials (.env / system env)

Auto-connects to the best available configuration with zero manual steps.
"""

import os
import glob
import json
import shutil
import requests
from typing import Dict, Any, List, Optional

from zerotrace.models.ollama import ollama_client
from zerotrace.telemetry.hardware import NVML_OK, get_gpu_telemetry, get_cpu_base_tdp
from zerotrace.telemetry.zeus_monitor import ZEUS_OK
from zerotrace.core.state import state
from zerotrace.models.cloud import CLOUD_PROVIDERS, cloud_client


class AutoDiscoveryEngine:
    """Scans system and auto-connects to offline models, agents, and hardware telemetry."""

    @staticmethod
    def scan_offline_models() -> List[Dict[str, Any]]:
        """Detects all offline / locally available models."""
        found = []

        # 1. Ollama Daemon
        if ollama_client.is_online():
            models = ollama_client.list_models()
            for m in models:
                found.append({
                    "type": "ollama",
                    "name": m["name"],
                    "path": f"http://127.0.0.1:11434 ({m.get('family', 'ollama')})",
                    "size_gb": m.get("size_gb", 0),
                    "quant": m.get("quant", "unknown"),
                    "status": "ready",
                })

        # 2. Local Model Directories (GGUF, PyTorch, SafeTensors)
        from zerotrace.config import BASE_DIR
        search_dirs = [
            str(BASE_DIR / "models"),
            os.path.abspath("./models"),
            os.path.abspath("../models"),
            os.path.expanduser("~/.ollama/models/manifests/registry.ollama.ai/library"),
            os.path.expanduser("~/.cache/lm-studio/models"),
            os.path.expanduser("~/.cache/huggingface/hub"),
        ]

        extensions = ("*.gguf", "*.bin", "*.safetensors", "*.pt", "*.pth")

        for d in search_dirs:
            if os.path.isdir(d):
                for ext in extensions:
                    for f in glob.glob(os.path.join(d, "**", ext), recursive=True):
                        try:
                            sz = round(os.path.getsize(f) / (1024**3), 2)
                            bname = os.path.basename(f)
                            found.append({
                                "type": "local_file",
                                "name": bname,
                                "path": f,
                                "size_gb": sz,
                                "quant": "gguf" if f.endswith(".gguf") else "weights",
                                "status": "ready",
                            })
                        except Exception:
                            continue

        return found

    @staticmethod
    def scan_agents() -> List[Dict[str, Any]]:
        """Detects active AI developer agents and coding assistants."""
        agents = []

        # 1. Google Antigravity CLI / Gemini Config
        agy_mcp = os.path.expanduser("~/.gemini/config/mcp_config.json")
        if os.path.exists(agy_mcp):
            has_zerotrace = False
            try:
                with open(agy_mcp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    has_zerotrace = "zerotrace" in data.get("mcpServers", {})
            except Exception:
                pass
            agents.append({
                "name": "Google Antigravity CLI (agy)",
                "detected": True,
                "mcp_connected": has_zerotrace,
                "status": "Connected & Optimizing" if has_zerotrace else "Detected (MCP available)",
            })

        # 2. Claude Code
        claude_cfg = os.path.expanduser("~/.claude.json")
        claude_cli = shutil.which("claude")
        if os.path.exists(claude_cfg) or claude_cli:
            agents.append({
                "name": "Claude Code",
                "detected": True,
                "mcp_connected": False,
                "status": "Detected (Proxy ready on :8000/v1)",
            })

        # 3. Cursor Editor
        cursor_dir = os.path.expanduser("~/.cursor")
        if os.path.isdir(cursor_dir):
            agents.append({
                "name": "Cursor Editor",
                "detected": True,
                "mcp_connected": False,
                "status": "Detected (OpenAI proxy :8000/v1 compatible)",
            })

        # 4. Standard OpenAI Proxy Clients
        agents.append({
            "name": "ZeroTrace OpenAI Proxy (:8000/v1)",
            "detected": True,
            "mcp_connected": True,
            "status": "Active (Universal API Gateway)",
        })

        return agents

    @staticmethod
    def scan_hardware() -> Dict[str, Any]:
        """Detects physical energy and hardware telemetry capabilities."""
        rapl_ok = os.path.exists("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")
        gpu = get_gpu_telemetry()

        return {
            "rapl_hardware_joules": rapl_ok,
            "nvml_gpu": NVML_OK,
            "gpu_name": gpu.get("gpu_name", "None"),
            "zeus_monitor": ZEUS_OK,
            "cpu_tdp_w": get_cpu_base_tdp(),
            "telemetry_source": "Intel RAPL + NVML" if (rapl_ok and NVML_OK) else ("NVML" if NVML_OK else "Isolated AI CPU Sweeper"),
        }

    @classmethod
    def auto_connect(cls) -> Dict[str, Any]:
        """
        Runs auto-discovery and automatically connects state to the best available model,
        hardware sensors, and detected agents.
        """
        offline_models = cls.scan_offline_models()
        agents = cls.scan_agents()
        hardware = cls.scan_hardware()

        auto_provider = "ollama"
        auto_model = "llama3:latest"
        offline_ready = False

        # If offline models exist, pick the first one
        if offline_models:
            first_m = offline_models[0]
            auto_model = first_m["name"]
            offline_ready = True
            state.active_ollama_model = auto_model
            state.active_model_name = auto_model
        else:
            # Check cloud credentials
            for p_id in ["gemini", "groq", "openai", "anthropic"]:
                if cloud_client.resolve_key(p_id):
                    auto_provider = p_id
                    auto_model = CLOUD_PROVIDERS[p_id]["default_model"]
                    state.active_model_name = f"{p_id}:{auto_model}"
                    break

        connected_agent_name = "Antigravity CLI"
        for ag in agents:
            if ag.get("mcp_connected"):
                connected_agent_name = ag["name"]
                break

        report = {
            "success": True,
            "auto_connected": {
                "provider": auto_provider,
                "model": auto_model,
                "offline_ready": offline_ready,
                "active_agent": connected_agent_name,
            },
            "offline_models": offline_models,
            "agents": agents,
            "hardware": hardware,
            "cache_size": state.vector_cache.size(),
            "caveman_active": True,
        }

        return report


autodiscover = AutoDiscoveryEngine()
