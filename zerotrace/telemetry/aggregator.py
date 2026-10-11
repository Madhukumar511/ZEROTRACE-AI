from typing import Dict, Any
from .hardware import get_gpu_telemetry, get_cpu_power_estimate


def get_total_system_power(cpu_pct: float) -> Dict[str, Any]:
    """Aggregates CPU package estimate and real NVML GPU power consumption."""
    gpu = get_gpu_telemetry()
    cpu_w = get_cpu_power_estimate(cpu_pct)
    gpu_w = gpu["gpu_power_w"]
    total = round(cpu_w + gpu_w, 2)
    return {
        "cpu_w": cpu_w,
        "gpu_w": gpu_w,
        "total_w": total,
        "source": "NVML (GPU) + TDP×util estimate (CPU)",
        "gpu_info": gpu,
    }
