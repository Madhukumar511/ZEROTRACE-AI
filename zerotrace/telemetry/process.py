import psutil
from typing import Dict

_ollama_procs: Dict[int, psutil.Process] = {}


def get_isolated_ai_cpu() -> float:
    """
    Sweeps system process trees for processes related to Ollama or Llama.
    Sums CPU usage and divides by total CPU cores to compute true
    system-wide percentage consumed solely by the AI, stripping OS noise.
    """
    global _ollama_procs
    current_pids = []
    total_cpu = 0.0

    for p in psutil.process_iter(['name']):
        try:
            name = p.info['name'].lower()
            if 'ollama' in name or 'llama' in name:
                pid = p.pid
                current_pids.append(pid)
                if pid not in _ollama_procs:
                    _ollama_procs[pid] = p
                    p.cpu_percent(interval=None)  # Prime baseline
                else:
                    total_cpu += _ollama_procs[pid].cpu_percent(interval=None)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    # Clean dead process handles
    keys_to_remove = [pid for pid in _ollama_procs if pid not in current_pids]
    for pid in keys_to_remove:
        del _ollama_procs[pid]

    core_count = psutil.cpu_count() or 1
    sys_cpu = total_cpu / core_count
    return round(sys_cpu, 1)
