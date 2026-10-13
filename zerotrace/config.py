import os
from pathlib import Path

# Paths
PACKAGE_DIR = Path(__file__).resolve().parent
BASE_DIR = PACKAGE_DIR.parent
INDEX_HTML_PATH = (PACKAGE_DIR / "index.html") if (PACKAGE_DIR / "index.html").exists() else (BASE_DIR / "index.html")

# Networking & Service URLs
HOST = os.getenv("ZEROTRACE_HOST", "0.0.0.0")
PORT = int(os.getenv("ZEROTRACE_PORT", "8000"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Carbon Telemetry
CARBON_API_URL = os.getenv("CARBON_API_URL", "https://api.carbonintensity.org.uk/intensity")
DEFAULT_CARBON_GCO2_KWH = float(os.getenv("DEFAULT_CARBON_INTENSITY", "214.0"))
CARBON_REFRESH_INTERVAL_SEC = 300

# Benchmark defaults
BENCHMARK_PROMPT = os.getenv(
    "ZEROTRACE_BENCHMARK_PROMPT",
    "Explain how transformer neural networks work. "
    "Cover self-attention, positional encoding, feed-forward layers, and training. "
    "Give a detailed explanation with at least 250 words."
)
DEFAULT_BENCHMARK_RUNS = 3

# Hardware Settings
RAPL_ENERGY_PATH = Path("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj")
DEFAULT_CPU_TDP_W = 65.0
