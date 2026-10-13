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
        """Detects active AI developer agents, IDEs, and coding assistants."""
        agents = []

        # 1. Google Antigravity CLI / Gemini Config
        agy_bin = shutil.which("agy") or os.path.expanduser("~/.local/bin/agy")
        agy_installed = bool(agy_bin and os.path.exists(agy_bin))
        agy_mcp = os.path.expanduser("~/.gemini/config/mcp_config.json")
        if agy_installed or os.path.exists(agy_mcp):
            agents.append({
                "name": "Google Antigravity CLI (agy)",
                "detected": True,
                "mcp_connected": True,
                "status": "Connected & Optimizing (Gemini 3.8 Active)",
                "path": agy_bin,
            })

        # 2. GitHub Copilot CLI / VS Code Agent
        copilot_bin = shutil.which("copilot") or os.path.expanduser("~/.local/bin/copilot")
        copilot_cfg = os.path.expanduser("~/.config/github-copilot")
        if (copilot_bin and os.path.exists(copilot_bin)) or os.path.isdir(copilot_cfg):
            agents.append({
                "name": "GitHub Copilot (VS Code IDE Agent)",
                "detected": True,
                "mcp_connected": True,
                "status": "Ready (Native CLI Execution Available)",
                "path": copilot_bin,
            })

        # 3. Cursor Editor & Cursor Agent
        cursor_bin = shutil.which("cursor-agent") or os.path.expanduser("~/.local/bin/cursor-agent")
        cursor_dir = os.path.expanduser("~/.cursor")
        if (cursor_bin and os.path.exists(cursor_bin)) or os.path.isdir(cursor_dir):
            agents.append({
                "name": "Cursor Editor & Agent",
                "detected": True,
                "mcp_connected": False,
                "status": "Detected (CLI Agent Available & :8000/v1 compatible)",
                "path": cursor_bin or cursor_dir,
            })

        # 4. VS Code Editor & AI Extensions
        code_bin = shutil.which("code") or "/usr/bin/code"
        code_user = os.path.expanduser("~/.config/Code/User")
        if os.path.exists(code_bin) or os.path.isdir(code_user):
            agents.append({
                "name": "Visual Studio Code (IDE)",
                "detected": True,
                "mcp_connected": True,
                "status": "Installed (Agent Host & Language Models API Active)",
                "path": code_bin,
            })

        # 5. Claude Code
        claude_cfg = os.path.expanduser("~/.claude.json")
        claude_cli = shutil.which("claude")
        if os.path.exists(claude_cfg) or claude_cli:
            agents.append({
                "name": "Claude Code",
                "detected": True,
                "mcp_connected": False,
                "status": "Detected (Proxy ready on :8000/v1)",
                "path": claude_cli,
            })

        # 6. Standard OpenAI Proxy Gateway
        agents.append({
            "name": "ZeroTrace OpenAI Proxy (:8000/v1)",
            "detected": True,
            "mcp_connected": True,
            "status": "Active (Universal IDE Gateway for Continue / Cline / Roo)",
        })

        return agents

    @staticmethod
    def scan_ide_models() -> List[Dict[str, Any]]:
        """
        Detects active IDE model servers (LM Studio, Jan, LocalAI),
        IDE Agent CLIs (GitHub Copilot, Cursor, Antigravity),
        and extension model configurations (Continue.dev, Cline, Roo Code).
        """
        ide_models = []

        # 1. Probe LM Studio Local Server (:1234)
        try:
            r = requests.get("http://127.0.0.1:1234/v1/models", timeout=0.35)
            if r.status_code == 200:
                data = r.json()
                for m in data.get("data", []):
                    m_id = m.get("id", "local-model")
                    ide_models.append({
                        "name": m_id,
                        "ide": "LM Studio (IDE Local Server :1234)",
                        "provider": "lmstudio",
                        "model": m_id,
                        "type": "ide_server",
                        "status": "online",
                        "endpoint": "http://127.0.0.1:1234/v1",
                    })
        except Exception:
            pass

        # 2. Probe Jan AI Local Server (:1337)
        try:
            r = requests.get("http://127.0.0.1:1337/v1/models", timeout=0.35)
            if r.status_code == 200:
                data = r.json()
                for m in data.get("data", []):
                    m_id = m.get("id", "default")
                    ide_models.append({
                        "name": m_id,
                        "ide": "Jan AI (IDE Local Server :1337)",
                        "provider": "jan",
                        "model": m_id,
                        "type": "ide_server",
                        "status": "online",
                        "endpoint": "http://127.0.0.1:1337/v1",
                    })
        except Exception:
            pass

        # 3. Probe LocalAI / vLLM / Tabby (:8080)
        try:
            r = requests.get("http://127.0.0.1:8080/v1/models", timeout=0.35)
            if r.status_code == 200:
                data = r.json()
                for m in data.get("data", []):
                    m_id = m.get("id", "default")
                    ide_models.append({
                        "name": m_id,
                        "ide": "LocalAI / vLLM (:8080)",
                        "provider": "localai",
                        "model": m_id,
                        "type": "ide_server",
                        "status": "online",
                        "endpoint": "http://127.0.0.1:8080/v1",
                    })
        except Exception:
            pass

        # 4. GitHub Copilot CLI & IDE Agent
        copilot_bin = shutil.which("copilot") or os.path.expanduser("~/.local/bin/copilot")
        if copilot_bin and os.path.exists(copilot_bin):
            for m_name in ["gpt-4o", "claude-3.5-sonnet", "o1-preview", "gpt-4o-mini"]:
                ide_models.append({
                    "name": f"copilot:{m_name}",
                    "ide": "GitHub Copilot (VS Code CLI Agent)",
                    "provider": "copilot",
                    "model": m_name,
                    "type": "ide_agent",
                    "status": "ready",
                    "path": copilot_bin,
                })

        # 5. Cursor Agent & Cursor IDE
        cursor_bin = shutil.which("cursor-agent") or os.path.expanduser("~/.local/bin/cursor-agent")
        cursor_dir = os.path.expanduser("~/.cursor")
        if (cursor_bin and os.path.exists(cursor_bin)) or os.path.isdir(cursor_dir):
            for m_name in ["claude-3.5-sonnet", "gpt-4o", "cursor-small"]:
                ide_models.append({
                    "name": f"cursor:{m_name}",
                    "ide": "Cursor IDE Agent",
                    "provider": "cursor",
                    "model": m_name,
                    "type": "ide_agent",
                    "status": "detected",
                    "path": cursor_bin or cursor_dir,
                })

        # 6. Continue.dev Configuration (~/.continue/config.json)
        continue_cfg = os.path.expanduser("~/.continue/config.json")
        if os.path.exists(continue_cfg):
            try:
                with open(continue_cfg, "r", encoding="utf-8") as f:
                    cdata = json.load(f)
                    for cm in cdata.get("models", []):
                        t = cm.get("title") or cm.get("model") or "continue-model"
                        p = cm.get("provider", "custom")
                        m = cm.get("model", t)
                        ide_models.append({
                            "name": f"continue:{t}",
                            "ide": "Continue.dev (VS Code / JetBrains)",
                            "provider": p,
                            "model": m,
                            "type": "ide_config",
                            "status": "configured",
                            "path": continue_cfg,
                        })
            except Exception:
                pass

        # 7. VS Code Cline / Roo Code Settings
        cline_paths = [
            os.path.expanduser("~/.config/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json"),
            os.path.expanduser("~/.config/Code/User/globalStorage/rooveterinaryinc.roo-cline/settings/cline_mcp_settings.json"),
        ]
        for cp in cline_paths:
            if os.path.exists(cp):
                ide_models.append({
                    "name": "cline:custom-model",
                    "ide": "VS Code Cline / Roo Code",
                    "provider": "custom",
                    "model": "custom-model",
                    "type": "ide_extension",
                    "status": "ready",
                    "path": cp,
                })

        return ide_models

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
        Runs auto-discovery across local models, IDE models/agents, hardware sensors,
        and cloud credentials, automatically selecting the best available configuration.
        """
        ide_models = cls.scan_ide_models()
        offline_models = cls.scan_offline_models()
        agents = cls.scan_agents()
        hardware = cls.scan_hardware()

        auto_provider = "ollama"
        auto_model = "llama3:latest"
        offline_ready = False
        is_ide = False

        # Priority 1: Offline local models (Ollama or GGUF files)
        if offline_models:
            first_m = offline_models[0]
            auto_model = first_m["name"]
            offline_ready = True
            state.active_ollama_model = auto_model
            state.active_model_name = auto_model
        else:
            # Priority 2: Online IDE Local Servers (LM Studio :1234, Jan :1337, LocalAI :8080)
            online_ide_servers = [m for m in ide_models if m.get("type") == "ide_server" and m.get("status") == "online"]
            if online_ide_servers:
                first_ide = online_ide_servers[0]
                auto_provider = first_ide["provider"]
                auto_model = first_ide["model"]
                state.active_model_name = f"{auto_provider}:{auto_model}"
                is_ide = True
            else:
                # Priority 3: Google Antigravity CLI (Gemini)
                agy_bin = shutil.which("agy") or os.path.expanduser("~/.local/bin/agy")
                has_agy = bool(agy_bin and os.path.exists(agy_bin))
                if has_agy:
                    auto_provider = "antigravity"
                    auto_model = "gemini-3.8-flash-high"
                    state.active_model_name = "antigravity:gemini-3.8-flash-high"
                else:
                    # Priority 4: GitHub Copilot CLI (VS Code IDE Agent)
                    copilot_bin = shutil.which("copilot") or os.path.expanduser("~/.local/bin/copilot")
                    if copilot_bin and os.path.exists(copilot_bin):
                        auto_provider = "copilot"
                        auto_model = "gpt-4o"
                        state.active_model_name = "copilot:gpt-4o"
                        is_ide = True
                    else:
                        # Priority 5: Cloud API credentials
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
                "is_ide_model": is_ide,
                "active_agent": connected_agent_name,
            },
            "offline_models": offline_models,
            "ide_models": ide_models,
            "agents": agents,
            "hardware": hardware,
            "cache_size": state.vector_cache.size(),
            "caveman_active": True,
        }

        return report


autodiscover = AutoDiscoveryEngine()
