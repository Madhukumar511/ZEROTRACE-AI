import requests
from typing import Dict, List, Any, Optional
from zerotrace.config import OLLAMA_BASE_URL


class OllamaClient:
    """Client for Ollama local inference daemon."""

    def __init__(self, base_url: str = OLLAMA_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def is_online(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            if r.status_code == 200:
                raw = r.json().get("models", [])
                models = []
                for m in raw:
                    models.append({
                        "name": m["name"],
                        "size_gb": round(m.get("size", 0) / 1e9, 2),
                        "family": m.get("details", {}).get("family", "unknown"),
                        "params": m.get("details", {}).get("parameter_size", "unknown"),
                        "quant": m.get("details", {}).get("quantization_level", "unknown"),
                    })
                return models
        except Exception:
            pass
        return []

    def generate(self, model: str, prompt: str, options: Optional[Dict[str, Any]] = None, timeout: int = 300) -> Dict[str, Any]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options or {},
        }
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def create_model(self, name: str, modelfile: str, timeout: int = 30) -> Dict[str, Any]:
        url = f"{self.base_url}/api/create"
        payload = {
            "name": name,
            "modelfile": modelfile,
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def get_quantization_variants(self, model_name: str) -> Dict[str, Any]:
        base = model_name.split(":")[0]
        variants = {
            "q2_k": {"bits": 2, "energy_factor": 0.35, "quality": "Low", "cmd": f"ollama pull {base}:q2_K"},
            "q4_0": {"bits": 4, "energy_factor": 0.50, "quality": "Medium", "cmd": f"ollama pull {base}:q4_0"},
            "q4_k_m": {"bits": 4, "energy_factor": 0.52, "quality": "Good", "cmd": f"ollama pull {base}:q4_K_M"},
            "q5_k_m": {"bits": 5, "energy_factor": 0.65, "quality": "Better", "cmd": f"ollama pull {base}:q5_K_M"},
            "q8_0": {"bits": 8, "energy_factor": 0.90, "quality": "Best", "cmd": f"ollama pull {base}:q8_0"},
        }
        try:
            available_models = [m["name"] for m in self.list_models()]
            for k, v in variants.items():
                tag = f"{base}:{k.replace('_', '-')}"
                v["downloaded"] = any(tag in d or k in d for d in available_models)
        except Exception:
            for v in variants.values():
                v["downloaded"] = False

        return {"base_model": base, "variants": variants}


ollama_client = OllamaClient()
