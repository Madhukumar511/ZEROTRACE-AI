# ZeroTrace AI Project Instructions & Agent Guidelines

## ⚡ Green AI Token & Energy Conservation Directive
This workspace is running **ZeroTrace AI**, an open-source Green AI inference gateway and telemetry profiler.

### 1. Token & Energy Efficiency
- Communicate with precision, density, and zero filler (Caveman / BLUF style).
- Avoid unnecessary pleasantries ("Sure, I would be happy to help you with that!"), verbose preambles, or repetitive summaries.
- Keep responses dense, technical, and code-accurate to minimize output tokens by ~65%, directly cutting cloud GPU inference electricity.

### 2. ZeroTrace Telemetry Integration
- ZeroTrace operates at `http://127.0.0.1:8000` and via the `zerotrace` MCP server (`zerotrace/mcp_server.py`).
- When the user is running `python cli.py live` or `python server.py` in another terminal, they are monitoring real-time Watts, Joules, and tokens saved.
- You can sync token savings by posting to `http://127.0.0.1:8000/api/telemetry/record` or calling the `zerotrace_record_savings` tool with estimated tokens saved.
- For repetitive technical queries, check the 384-D Semantic Vector Cache to answer at 0 Watts.

### 3. Verification & Architecture Standards
- Maintain all 43+ automated tests in `tests/`.
- Ensure real hardware energy physics (NVML GPU milliwatts, Intel RAPL microjoules, isolated CPU tracking) are strictly preserved with zero synthetic hardcoded values.
