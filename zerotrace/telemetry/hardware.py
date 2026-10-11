import os
from typing import Dict, Any, Optional
from zerotrace.config import RAPL_ENERGY_PATH, DEFAULT_CPU_TDP_W

# NVIDIA NVML Initialization
NVML_OK = False
pynvml = None

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    pynvml = _pynvml
    NVML_OK = True
except Exception:
    pynvml = None
    NVML_OK = False


def get_gpu_telemetry() -> Dict[str, Any]:
    """Returns real GPU metrics via NVIDIA Management Library (NVML) if present."""
    if NVML_OK and pynvml is not None:
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            gpu_name = name if isinstance(name, str) else name.decode()
            return {
                "gpu_util_pct": round(float(util.gpu), 1),
                "gpu_mem_pct": round((mem.used / mem.total) * 100.0, 1) if mem.total > 0 else 0.0,
                "gpu_mem_used_gb": round(mem.used / 1e9, 2),
                "gpu_mem_total_gb": round(mem.total / 1e9, 2),
                "gpu_power_w": round(power_mw / 1000.0, 2),
                "gpu_name": gpu_name,
                "available": True,
            }
        except Exception:
            pass

    return {
        "gpu_util_pct": 0.0,
        "gpu_mem_pct": 0.0,
        "gpu_mem_used_gb": 0.0,
        "gpu_mem_total_gb": 0.0,
        "gpu_power_w": 0.0,
        "gpu_name": "N/A",
        "available": False,
    }


def get_cpu_base_tdp() -> float:
    """Estimates CPU TDP in watts from /proc/cpuinfo architecture tags."""
    tdp_w = DEFAULT_CPU_TDP_W
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "model name" in line.lower():
                        name = line.lower()
                        if "threadripper" in name or "epyc" in name:
                            tdp_w = 180.0
                        elif "xeon" in name:
                            tdp_w = 140.0
                        elif "i9" in name or "ryzen 9" in name:
                            tdp_w = 125.0
                        elif "i7" in name or "ryzen 7" in name:
                            tdp_w = 95.0
                        elif "i5" in name or "ryzen 5" in name:
                            tdp_w = 65.0
                        elif "i3" in name or "ryzen 3" in name:
                            tdp_w = 35.0
                        break
    except Exception:
        pass
    return tdp_w


def get_cpu_power_estimate(cpu_pct: float) -> float:
    """Estimates CPU package power in Watts using: CPU_Utilization_Ratio × TDP."""
    if cpu_pct <= 0.0:
        return 0.0
    tdp = get_cpu_base_tdp()
    return round(tdp * (cpu_pct / 100.0), 2)


def read_rapl_energy_uj() -> Optional[int]:
    """Reads hardware energy counter in microjoules from Intel RAPL if accessible."""
    try:
        if RAPL_ENERGY_PATH.exists():
            with open(RAPL_ENERGY_PATH, "r") as f:
                return int(f.read().strip())
    except Exception:
        return None
    return None


def apply_gpu_power_cap(power_limit_w: int) -> Dict[str, Any]:
    """Sets NVIDIA GPU power management limit in Watts."""
    if not NVML_OK or pynvml is None:
        return {"success": False, "error": "pynvml not available"}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        min_mw, max_mw = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
        target_mw = max(min_mw, min(int(power_limit_w * 1000), max_mw))
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, target_mw)
        actual_w = target_mw / 1000.0
        return {
            "success": True,
            "requested_w": power_limit_w,
            "actual_w": actual_w,
            "min_w": min_mw / 1000.0,
            "max_w": max_mw / 1000.0,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "note": "Requires root or administrator privileges"}


def reset_gpu_power_cap() -> Dict[str, Any]:
    """Restores NVIDIA GPU default power limit."""
    if not NVML_OK or pynvml is None:
        return {"success": False, "error": "pynvml not available"}
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        default_mw = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle)
        pynvml.nvmlDeviceSetPowerManagementLimit(handle, default_mw)
        return {"success": True, "restored_w": default_mw / 1000.0}
    except Exception as e:
        return {"success": False, "error": str(e)}
