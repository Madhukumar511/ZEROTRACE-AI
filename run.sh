#!/usr/bin/env bash
# ==============================================================================
#  🌱 ZeroTrace AI — 1-Click Laptop Runner (Linux & macOS)
#  Slashes AI electrical consumption, token waste, and carbon emissions.
# ==============================================================================

set -e

# Colors
GREEN='\033[0;32m'
TEAL='\033[0;36m'
GOLD='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${TEAL}"
cat << "EOF"
  ______               _______                      ___    ____
 /_  __/___  _________/_  __/_________ _________   /   |  /  _/
  / / / _ \/ ___/ __ \ / / / ___/ __ `/ ___/ _ \  / /| |  / /  
 / / /  __/ /  / /_/ // / / /  / /_/ / /__/  __/ / ___ |_/ /   
/_/  \___/_/   \____//_/ /_/   \__,_/\___/\___/ /_/  |_/___/   
                 Green AI Inference Gateway
EOF
echo -e "${NC}"

# 1. Check Python installation
if command -v python3 &>/dev/null; then
    PY_CMD="python3"
elif command -v python &>/dev/null; then
    PY_CMD="python"
else
    echo -e "${RED}[ERROR] Python 3 not found on your system.${NC}"
    echo "Please install Python 3.9+ from https://www.python.org/ or your package manager."
    exit 1
fi

PY_VER=$($PY_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}✓ Found Python $PY_VER ($PY_CMD)${NC}"

# 2. Virtual Environment Setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo -e "${GOLD}⚙ Setting up isolated virtual environment (.venv)...${NC}"
    $PY_CMD -m venv .venv
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✓ Virtual environment activated (.venv)${NC}"
fi

# 3. Check and install dependencies
echo -e "${TEAL}📦 Verifying dependencies...${NC}"
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo -e "${GREEN}✓ Dependencies verified and ready.${NC}"

# 4. Check Environment Configuration (.env)
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    echo -e "${GOLD}ℹ Created .env template from .env.example${NC}"
fi

# 5. Hardware Sensor Inspection (Intel RAPL / NVIDIA NVML)
echo -e "${TEAL}🔍 Checking hardware energy telemetry sensors...${NC}"
if [ -r "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj" ]; then
    echo -e "${GREEN}✓ Intel RAPL hardware energy counters accessible (Zero-overhead Joules tracking)${NC}"
else
    echo -e "${GOLD}ℹ Standard CPU power estimator active. (To enable microsecond Intel RAPL hardware counters: sudo chmod -R a+r /sys/class/powercap/intel-rapl/)${NC}"
fi

# 6. Check Local Ollama Status
if curl -s http://127.0.0.1:11434/api/tags &>/dev/null; then
    echo -e "${GREEN}✓ Local Ollama detected and running on port 11434${NC}"
else
    echo -e "${GOLD}ℹ Local Ollama is not running. (You can still chat using Cloud models like Gemini, Groq, or OpenAI!)${NC}"
fi

PORT=8000
HOST="127.0.0.1"
URL="http://$HOST:$PORT"

echo -e "\n${GREEN}========================================================================${NC}"
echo -e "${GREEN}🚀 ZeroTrace AI is launching at: ${TEAL}$URL${NC}"
echo -e "${GREEN}   - Web Dashboard:     ${TEAL}$URL${NC}"
echo -e "${GREEN}   - Green Chat:        ${TEAL}$URL#chat-sec${NC}"
echo -e "${GREEN}   - API Documentation: ${TEAL}$URL/docs${NC}"
echo -e "${GREEN}========================================================================${NC}\n"

# 7. Auto-open browser in background
(
    sleep 1.5
    if command -v xdg-open &>/dev/null; then
        xdg-open "$URL" &>/dev/null || true
    elif command -v open &>/dev/null; then
        open "$URL" &>/dev/null || true
    fi
) &

# 8. Start ZeroTrace Server
exec python server.py
