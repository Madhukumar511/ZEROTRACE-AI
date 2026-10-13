import os
import time
import shutil
import subprocess
import requests
from typing import Dict, Any, Optional

from zerotrace.compression.tokenizer import estimate_token_count

# Supported cloud providers and their standard default models
CLOUD_PROVIDERS = {
    "antigravity": {
        "name": "Google Antigravity CLI (Gemini)",
        "default_model": "gemini-3.8-flash-high",
        "models": ["gemini-3.8-flash-high", "gemini-3.8-flash-medium", "claude-sonnet-4-6"],
        "env_key": None,
    },
    "gemini": {
        "name": "Google Gemini",
        "default_model": "gemini-1.5-flash",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-exp"],
        "env_key": "GEMINI_API_KEY",
    },
    "openai": {
        "name": "OpenAI",
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        "env_key": "OPENAI_API_KEY",
    },
    "groq": {
        "name": "Groq (Ultra-Fast LPU)",
        "default_model": "llama-3.3-70b-versatile",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "env_key": "GROQ_API_KEY",
    },
    "anthropic": {
        "name": "Anthropic Claude",
        "default_model": "claude-3-5-sonnet-20241022",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
        "env_key": "ANTHROPIC_API_KEY",
    },
    "copilot": {
        "name": "GitHub Copilot (VS Code CLI Agent)",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "claude-3.5-sonnet", "o1-preview", "gpt-4o-mini"],
        "env_key": None,
    },
    "cursor": {
        "name": "Cursor Agent (IDE CLI)",
        "default_model": "claude-3.5-sonnet",
        "models": ["claude-3.5-sonnet", "gpt-4o", "cursor-small"],
        "env_key": "CURSOR_API_KEY",
    },
    "lmstudio": {
        "name": "LM Studio (IDE Server :1234)",
        "default_model": "default",
        "models": ["default"],
        "env_key": None,
        "base_url": "http://127.0.0.1:1234/v1",
    },
    "jan": {
        "name": "Jan AI (IDE Server :1337)",
        "default_model": "default",
        "models": ["default"],
        "env_key": None,
        "base_url": "http://127.0.0.1:1337/v1",
    },
    "localai": {
        "name": "LocalAI / vLLM (:8080)",
        "default_model": "default",
        "models": ["default"],
        "env_key": None,
        "base_url": "http://127.0.0.1:8080/v1",
    },
    "custom": {
        "name": "Custom OpenAI-Compatible",
        "default_model": "default",
        "models": ["default"],
        "env_key": "CUSTOM_API_KEY",
    },
}


class CloudLLMClient:
    """
    Lightweight, unified client for web and cloud LLM providers.
    Zero heavy SDK dependencies; uses robust standard HTTP requests.
    Supports Google Gemini, OpenAI, Groq, Anthropic, and custom OpenAI-compatible endpoints.
    """

    def __init__(self, timeout: int = 45):
        self.timeout = timeout

    def resolve_key(self, provider: str, provided_key: Optional[str] = None) -> Optional[str]:
        if provided_key and provided_key.strip():
            return provided_key.strip()
        if provider in ["antigravity", "agy"]:
            agy_bin = shutil.which("agy") or os.path.expanduser("~/.local/bin/agy")
            return "antigravity_authenticated" if (agy_bin and os.path.exists(agy_bin)) else None
        if provider in ["copilot", "github-copilot"]:
            copilot_bin = shutil.which("copilot") or os.path.expanduser("~/.local/bin/copilot")
            return "copilot_authenticated" if (copilot_bin and os.path.exists(copilot_bin)) else None
        if provider in ["lmstudio", "jan", "localai"]:
            return "local_ide_endpoint"
        env_var = CLOUD_PROVIDERS.get(provider, {}).get("env_key")
        if env_var:
            return os.environ.get(env_var, "").strip() or None
        return None

    def call_gemini(self, prompt: str, model: str = "gemini-1.5-flash", api_key: Optional[str] = None) -> Dict[str, Any]:
        key = self.resolve_key("gemini", api_key)
        if not key:
            return {"success": False, "error": "Gemini API key missing. Set GEMINI_API_KEY in .env or provide in request."}

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }

        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            duration_s = time.time() - t0
            if resp.status_code != 200:
                return {"success": False, "error": f"Gemini error ({resp.status_code}): {resp.text}"}

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return {"success": False, "error": "Gemini returned no candidates"}

            text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            usage = data.get("usageMetadata", {})

            return {
                "success": True,
                "text": text,
                "provider": "gemini",
                "model": model,
                "prompt_tokens": usage.get("promptTokenCount", 0),
                "completion_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0),
                "duration_s": round(duration_s, 3),
            }
        except Exception as e:
            return {"success": False, "error": f"Gemini request failed: {str(e)}"}

    def call_openai_compatible(
        self,
        prompt: str,
        provider: str = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        key = self.resolve_key(provider, api_key)
        if not key and provider not in ["custom", "lmstudio", "jan", "localai"]:
            return {"success": False, "error": f"{provider.capitalize()} API key missing. Set {CLOUD_PROVIDERS.get(provider, {}).get('env_key')} in .env or provide in request."}

        if provider == "groq":
            url = base_url or "https://api.groq.com/openai/v1/chat/completions"
        elif provider == "lmstudio":
            url = base_url or "http://127.0.0.1:1234/v1/chat/completions"
        elif provider == "jan":
            url = base_url or "http://127.0.0.1:1337/v1/chat/completions"
        elif provider == "localai":
            url = base_url or "http://127.0.0.1:8080/v1/chat/completions"
        elif provider == "openai":
            url = base_url or "https://api.openai.com/v1/chat/completions"
        else:
            url = base_url or "https://api.openai.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key or ''}",
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
        }

        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            duration_s = time.time() - t0
            if resp.status_code != 200:
                return {"success": False, "error": f"{provider} error ({resp.status_code}): {resp.text}"}

            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                return {"success": False, "error": f"{provider} returned no choices"}

            text = choices[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})

            return {
                "success": True,
                "text": text,
                "provider": provider,
                "model": model,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "duration_s": round(duration_s, 3),
            }
        except Exception as e:
            return {"success": False, "error": f"{provider} request failed: {str(e)}"}

    def call_anthropic(self, prompt: str, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None) -> Dict[str, Any]:
        key = self.resolve_key("anthropic", api_key)
        if not key:
            return {"success": False, "error": "Anthropic API key missing. Set ANTHROPIC_API_KEY in .env or provide in request."}

        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        }
        payload = {
            "model": model,
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": prompt}],
        }

        t0 = time.time()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            duration_s = time.time() - t0
            if resp.status_code != 200:
                return {"success": False, "error": f"Anthropic error ({resp.status_code}): {resp.text}"}

            data = resp.json()
            content = data.get("content", [])
            text = content[0].get("text", "") if content else ""
            usage = data.get("usage", {})

            return {
                "success": True,
                "text": text,
                "provider": "anthropic",
                "model": model,
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
                "duration_s": round(duration_s, 3),
            }
        except Exception as e:
            return {"success": False, "error": f"Anthropic request failed: {str(e)}"}

    def call_antigravity(
        self,
        prompt: str,
        model: Optional[str] = None,
        conversation_id: Optional[str] = None,
        effort: str = "low",
    ) -> Dict[str, Any]:
        """Calls Google Antigravity CLI (agy) non-interactively using system Google session."""
        agy_bin = shutil.which("agy") or os.path.expanduser("~/.local/bin/agy")
        if not (agy_bin and os.path.exists(agy_bin)):
            return {"success": False, "error": "Google Antigravity CLI ('agy') not found in PATH or ~/.local/bin"}

        prompt = (prompt or "").strip() or "Hello"

        cmd = [agy_bin, "--disable-slash-commands"]
        if effort in ["low", "medium", "high"]:
            cmd.extend(["--effort", effort])
        if model and model != "default":
            cmd.extend(["--model", model])
        if conversation_id:
            cmd.extend(["--conversation", conversation_id])
        cmd.extend(["-p", prompt])

        t0 = time.time()
        timeout_sec = max(self.timeout, 90)
        try:
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_sec)
            duration_s = time.time() - t0
            if res.returncode != 0:
                err = res.stderr.strip() or f"agy exited with code {res.returncode}"
                return {"success": False, "error": err}

            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            text = ansi_escape.sub('', res.stdout).strip()
            prompt_toks = estimate_token_count(prompt)
            completion_toks = estimate_token_count(text)

            return {
                "success": True,
                "text": text,
                "provider": "antigravity",
                "model": model or "gemini-3.8-flash-high",
                "conversation_id": conversation_id,
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": prompt_toks + completion_toks,
                "duration_s": round(duration_s, 3),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Antigravity CLI timed out after {timeout_sec}s"}
        except Exception as e:
            return {"success": False, "error": f"Antigravity CLI execution error: {str(e)}"}

    def call_copilot(
        self,
        prompt: str,
        model: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Calls GitHub Copilot CLI non-interactively using system Copilot session."""
        copilot_bin = shutil.which("copilot") or os.path.expanduser("~/.local/bin/copilot")
        if not (copilot_bin and os.path.exists(copilot_bin)):
            return {"success": False, "error": "GitHub Copilot CLI ('copilot') not found in PATH or ~/.local/bin"}

        prompt = (prompt or "").strip() or "Hello"

        cmd = [copilot_bin, "-p", prompt, "--allow-all"]
        if model and model not in ["default", "copilot"]:
            cmd.extend(["--model", model])
        if conversation_id:
            cmd.extend([f"--resume={conversation_id}"])

        t0 = time.time()
        timeout_sec = max(self.timeout, 60)
        try:
            res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout_sec)
            duration_s = time.time() - t0
            if res.returncode != 0:
                err = res.stderr.strip() or f"copilot exited with code {res.returncode}"
                return {"success": False, "error": err}

            import re
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            raw_text = ansi_escape.sub('', res.stdout).strip()

            lines = []
            for line in raw_text.splitlines():
                if any(line.strip().startswith(prefix) for prefix in ["Changes ", "AI Credits ", "Tokens ", "Resume "]):
                    continue
                lines.append(line)
            clean_text = "\n".join(lines).strip() or raw_text

            prompt_toks = estimate_token_count(prompt)
            completion_toks = estimate_token_count(clean_text)

            return {
                "success": True,
                "text": clean_text,
                "provider": "copilot",
                "model": model or "gpt-4o",
                "conversation_id": conversation_id,
                "prompt_tokens": prompt_toks,
                "completion_tokens": completion_toks,
                "total_tokens": prompt_toks + completion_toks,
                "duration_s": round(duration_s, 3),
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Copilot CLI timed out after {timeout_sec}s"}
        except Exception as e:
            return {"success": False, "error": f"Copilot CLI execution error: {str(e)}"}

    def generate(
        self,
        provider: str,
        prompt: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        conversation_id: Optional[str] = None,
        effort: str = "low",
    ) -> Dict[str, Any]:
        """Dispatches request to specified cloud or IDE provider."""
        provider = provider.lower().strip()
        prov_info = CLOUD_PROVIDERS.get(provider, CLOUD_PROVIDERS.get("antigravity", CLOUD_PROVIDERS["openai"]))
        model = model or prov_info["default_model"]

        if provider in ["antigravity", "agy"]:
            return self.call_antigravity(prompt=prompt, model=model, conversation_id=conversation_id, effort=effort)
        elif provider in ["copilot", "github-copilot"]:
            return self.call_copilot(prompt=prompt, model=model, conversation_id=conversation_id)
        elif provider == "gemini":
            return self.call_gemini(prompt=prompt, model=model, api_key=api_key)
        elif provider == "anthropic":
            return self.call_anthropic(prompt=prompt, model=model, api_key=api_key)
        elif provider in ["openai", "groq", "custom", "lmstudio", "jan", "localai"]:
            return self.call_openai_compatible(
                prompt=prompt,
                provider=provider,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
        else:
            return {"success": False, "error": f"Unsupported cloud or IDE provider: '{provider}'"}


cloud_client = CloudLLMClient()
