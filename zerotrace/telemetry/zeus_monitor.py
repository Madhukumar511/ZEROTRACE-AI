from typing import Optional

ZEUS_OK = False
_zeus_monitor = None

try:
    from zeus.monitor import ZeusMonitor
    _zeus_monitor = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)
    ZEUS_OK = True
except Exception:
    _zeus_monitor = None
    ZEUS_OK = False


class ZeusManager:
    """Manages energy measurement windows using Zeus ML if available."""

    def __init__(self, monitor=None):
        self.monitor = monitor

    @property
    def is_available(self) -> bool:
        return ZEUS_OK and self.monitor is not None

    def begin_window(self, window_key: str):
        if self.is_available:
            try:
                self.monitor.begin_window(window_key, sync_execution=False)
            except Exception:
                pass

    def end_window(self, window_key: str) -> Optional[float]:
        """Returns total energy in Joules if Zeus is active."""
        if self.is_available:
            try:
                result = self.monitor.end_window(window_key, sync_execution=False)
                return getattr(result, "total_energy", None)
            except Exception:
                pass
        return None


zeus_manager = ZeusManager(_zeus_monitor)
