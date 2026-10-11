import threading
from collections import deque
from typing import Optional, Dict, Any, List


class StateManager:
    """Thread-safe centralized state manager for Zerotrace AI."""

    def __init__(self):
        self._lock = threading.RLock()
        self.phase: str = "idle"
        self.inference_active: bool = False

        # Ollama State
        self.active_ollama_model: Optional[str] = None
        self.ollama_options: Dict[str, Any] = {}
        self.ollama_profile: str = "none"

        # PyTorch State
        self.pytorch_model = None
        self.pytorch_tokenizer = None
        self.pytorch_model_name: str = "None"

        # Stress Test State
        self.is_stress_testing: bool = False
        self.stress_thread: Optional[threading.Thread] = None

        # Caveman Token Optimization State
        self.caveman_active: bool = False
        self.caveman_intensity: str = "full"
        self.total_tokens_saved: int = 0
        self.total_energy_saved_j: float = 0.0

        # Semantic Vector Cache & kNN Gating State
        from zerotrace.core.vector_datastore import VectorDatastore
        from zerotrace.core.knn_combiner import KNNCombiner
        self.vector_cache = VectorDatastore(dimension=384)
        self.knn_combiner = KNNCombiner(cache_threshold=0.88)
        self.total_cache_hits: int = 0
        self._seed_default_cache()

        # Baseline Snapshot
        self.baseline_snapshot: Dict[str, Any] = {
            "power_w": None,
            "tokens_per_sec": None,
            "latency_ms": None,
            "energy_j_per_tok": None,
            "energy_j": None,
            "tokens_generated": None,
            "cpu_pct": None,
            "gpu_util": None,
            "gpu_power_w": None,
            "model": None,
            "measured_at": None,
            "n_runs": None,
            "benchmark_prompt": None,
        }

        # Telemetry Rolling History (300 seconds)
        self.telemetry_history: deque = deque(maxlen=300)

    def set_active_model(self, model_name: str):
        with self._lock:
            self.active_ollama_model = model_name
            self.pytorch_model = None
            self.pytorch_tokenizer = None
            self.ollama_options = {}
            self.ollama_profile = "none"
            self.phase = "model_selected"
            for k in self.baseline_snapshot:
                self.baseline_snapshot[k] = None
            self.baseline_snapshot["model"] = model_name

    def update_baseline(self, baseline_metrics: Dict[str, Any]):
        with self._lock:
            self.baseline_snapshot.update(baseline_metrics)
            self.phase = "benchmarked"

    def reset_optimizations(self):
        with self._lock:
            self.ollama_options = {}
            self.ollama_profile = "none"
            self.caveman_active = False
            if self.phase == "optimized":
                self.phase = "benchmarked"

    def record_telemetry(self, snapshot: Dict[str, Any]):
        with self._lock:
            self.telemetry_history.append(snapshot)

    def reset_all(self):
        with self._lock:
            self.phase = "idle"
            self.inference_active = False
            self.active_ollama_model = None
            self.ollama_options = {}
            self.ollama_profile = "none"
            self.pytorch_model = None
            self.pytorch_tokenizer = None
            self.pytorch_model_name = "None"
            self.is_stress_testing = False
            self.stress_thread = None
            self.caveman_active = False
            self.caveman_intensity = "full"
            self.total_tokens_saved = 0
            self.total_energy_saved_j = 0.0
            for k in self.baseline_snapshot:
                self.baseline_snapshot[k] = None
            self.telemetry_history.clear()
            self.total_cache_hits = 0

    def _seed_default_cache(self):
        """Pre-seeds common developer queries into the semantic vector cache."""
        try:
            from zerotrace.core.embedder import embedder
            seeds = [
                (
                    "what is zerotrace ai",
                    "ZeroTrace AI is a green AI efficiency platform that slashes GPU electricity (Joules), token waste, and carbon emissions using Semantic Vector Caching, Caveman prompt compression, and live hardware power telemetry.",
                ),
                (
                    "how to save energy with llms",
                    "To minimize LLM electrical consumption: 1) Cache frequent queries with 384-D vector matching (0W response); 2) Compress prompt filler with Caveman (~60% token cut); 3) Use 4-bit quantization (Q4_0); 4) Cap generation token limits.",
                ),
                (
                    "how to reverse a string in python",
                    "In Python, reverse a string using slice syntax:\n```python\ntext = 'hello'\nreversed_text = text[::-1]\nprint(reversed_text)  # 'olleh'\n```",
                ),
                (
                    "what is caveman compression",
                    "Caveman compression strips conversational filler, pleasantries, greetings, and articles into dense, direct technical syntax while preserving code blocks and technical identifiers verbatim. It achieves ~65% token reduction.",
                ),
            ]
            for q, a in seeds:
                vec = embedder.embed(q)
                self.vector_cache.add(query=q, answer=a, vector=vec, metadata={"seed": True})
        except Exception:
            pass


state = StateManager()
