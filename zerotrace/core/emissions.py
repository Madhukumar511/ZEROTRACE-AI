import time
import threading
import requests
from typing import Optional
from zerotrace.config import CARBON_API_URL, DEFAULT_CARBON_GCO2_KWH, CARBON_REFRESH_INTERVAL_SEC

_carbon_gco2_kwh: float = DEFAULT_CARBON_GCO2_KWH
_carbon_lock = threading.Lock()
_bg_thread_started = False


def _carbon_refresh_loop():
    global _carbon_gco2_kwh
    while True:
        try:
            r = requests.get(
                CARBON_API_URL,
                headers={"Accept": "application/json"},
                timeout=5,
            )
            if r.status_code == 200:
                data = r.json()
                val = data.get("data", [{}])[0].get("intensity", {}).get("actual")
                if val is not None:
                    with _carbon_lock:
                        _carbon_gco2_kwh = float(val)
        except Exception:
            pass
        time.sleep(CARBON_REFRESH_INTERVAL_SEC)


def start_carbon_monitor():
    """Starts the background carbon refresh loop daemon."""
    global _bg_thread_started
    if not _bg_thread_started:
        t = threading.Thread(target=_carbon_refresh_loop, daemon=True, name="carbon-monitor")
        t.start()
        _bg_thread_started = True


def get_carbon() -> float:
    """Returns the current carbon intensity in gCO2/kWh."""
    with _carbon_lock:
        return _carbon_gco2_kwh


def set_carbon(intensity: float):
    """Overrides carbon intensity value (useful for testing or manual regional configuration)."""
    global _carbon_gco2_kwh
    with _carbon_lock:
        _carbon_gco2_kwh = float(intensity)


# Auto-start carbon monitor on module load
start_carbon_monitor()
