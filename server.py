"""
Zerotrace AI — Real-Time Hardware Telemetry & Green AI Profiler
Entry point and backward-compatible interface.
Modularized architecture powered by the `zerotrace` core engine.
"""

from zerotrace.api.app import app
from zerotrace.telemetry.hardware import (
    get_cpu_power_estimate,
    apply_gpu_power_cap,
    reset_gpu_power_cap,
    get_gpu_telemetry,
)
from zerotrace.telemetry.process import get_isolated_ai_cpu
from zerotrace.telemetry.aggregator import get_total_system_power
from zerotrace.core.emissions import get_carbon, set_carbon
from zerotrace.core.optimizer import (
    apply_token_limit,
    apply_context_reduction,
    apply_gpu_routing,
    apply_caveman_compression_optimization,
)
from zerotrace.core.benchmark import measure_ollama_inference, run_benchmark_suite
from zerotrace.compression.caveman import CavemanCompressor
from zerotrace.compression.tokenizer import estimate_token_count, calculate_savings
from zerotrace.config import HOST, PORT

__all__ = [
    "app",
    "get_cpu_power_estimate",
    "get_total_system_power",
    "get_carbon",
    "set_carbon",
    "get_isolated_ai_cpu",
    "get_gpu_telemetry",
    "apply_token_limit",
    "apply_context_reduction",
    "apply_gpu_routing",
    "apply_caveman_compression_optimization",
    "apply_gpu_power_cap",
    "reset_gpu_power_cap",
    "measure_ollama_inference",
    "run_benchmark_suite",
    "CavemanCompressor",
    "estimate_token_count",
    "calculate_savings",
]

if __name__ == "__main__":
    import uvicorn
    print("\n[OK] Zerotrace AI - Global Telemetry & Caveman Token Optimizer Active.")
    print("     Start Ollama first: ollama serve")
    print(f"     Then open http://localhost:{PORT}\n")
    uvicorn.run("server:app", host=HOST, port=PORT, reload=True)