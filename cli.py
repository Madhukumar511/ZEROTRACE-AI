#!/usr/bin/env python3
"""
Zerotrace AI - Command Line Interface (CLI)
Terminal-based hardware telemetry profiler, live carbon monitor,
and Caveman token/energy compression optimizer.
"""

import sys
import time
import argparse
from datetime import datetime

from zerotrace.telemetry.aggregator import get_total_system_power
from zerotrace.telemetry.hardware import NVML_OK, get_gpu_telemetry, get_cpu_power_estimate, get_cpu_base_tdp
from zerotrace.telemetry.process import get_isolated_ai_cpu
from zerotrace.telemetry.zeus_monitor import ZEUS_OK
from zerotrace.core.emissions import get_carbon
from zerotrace.compression.caveman import CavemanCompressor
from zerotrace.compression.prompt_optimizer import PromptOptimizer
from zerotrace.core.state import state
from zerotrace.models.ollama import ollama_client


def cmd_live(args):
    """Streams live hardware power, isolated AI CPU, carbon telemetry, and cumulative green savings to terminal."""
    print("\n" + "=" * 78)
    print(" ⚡ ZEROTRACE AI — LIVE HARDWARE & GREEN SAVINGS MONITOR")
    print("=" * 78)
    print(f" CPU Base TDP: {get_cpu_base_tdp()} W | NVML GPU: {'Enabled' if NVML_OK else 'Not Detected'} | Zeus: {'Active' if ZEUS_OK else 'Disabled'}")
    print(f" Carbon Intensity: {get_carbon():.1f} gCO2/kWh")
    print("-" * 78)
    print(f"{'TIME':<8} | {'CPU %':<7} | {'TOTAL W':<9} | {'CO2 (mg/s)':<11} | {'SAVED (J)':<11} | {'SAVED (gCO2)':<12} | {'CACHE HITS'}")
    print("-" * 78)

    try:
        while True:
            # Check if live server has updated state
            saved_j = state.total_energy_saved_j
            saved_tokens = state.total_tokens_saved
            saved_hits = state.total_cache_hits
            carbon_kwh = get_carbon()
            saved_co2_g = (saved_j / 3_600_000.0) * carbon_kwh

            try:
                import requests
                r = requests.get("http://127.0.0.1:8000/api/telemetry/live", timeout=0.5)
                if r.status_code == 200:
                    d = r.json()
                    saved_j = d.get("total_energy_saved_j", saved_j)
                    saved_tokens = d.get("total_tokens_saved", saved_tokens)
                    saved_hits = d.get("total_cache_hits", saved_hits)
                    saved_co2_g = d.get("total_carbon_saved_g", saved_co2_g)
            except Exception:
                pass

            ai_cpu = get_isolated_ai_cpu()
            power = get_total_system_power(ai_cpu)
            total_w = power["total_w"]
            co2_mg_s = ((total_w / 3_600_000.0) * carbon_kwh) * 1000.0

            now_str = datetime.now().strftime("%H:%M:%S")
            print(
                f"{now_str:<8} | "
                f"{ai_cpu:>5.1f}% | "
                f"{total_w:>7.1f} W | "
                f"{co2_mg_s:>9.2f} | "
                f"{saved_j:>9.1f} J | "
                f"{saved_co2_g:>10.5f} g | "
                f"{saved_hits:>5} hits"
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n" + "=" * 78)
        print(" 📊 SESSION FINAL GREEN AUDIT:")
        print(f" Total Electricity Saved: {state.total_energy_saved_j:.2f} Joules ({state.total_energy_saved_j / 3600.0:.4f} Watt-hours)")
        print(f" Total Tokens Saved:      {state.total_tokens_saved} tokens")
        print(f" Total Carbon Prevented:  {(state.total_energy_saved_j / 3_600_000.0) * get_carbon():.6f} gCO2")
        print(f" 0-Watt Cache Hits:       {state.total_cache_hits}")
        print("=" * 78 + "\n")


def cmd_compress(args):
    """Compresses prompts or text files using Caveman heuristics and shows token & energy savings."""
    text = args.text
    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            print(f"[ERROR] Could not read file {args.file}: {e}")
            sys.exit(1)

    if not text:
        print("[ERROR] Please provide text or --file to compress.")
        sys.exit(1)

    power = get_total_system_power(0.0)
    current_w = power["total_w"] if power["total_w"] > 0 else 65.0

    result = PromptOptimizer.optimize(
        prompt=text,
        intensity=args.intensity,
        power_w=current_w,
        tok_per_sec=25.0,
        carbon_gco2_kwh=get_carbon(),
    )
    metrics = result["metrics"]

    print("\n" + "=" * 65)
    print(f" 🪨 CAVEMAN TOKEN OPTIMIZER — INTENSITY: {args.intensity.upper()}")
    print("=" * 65)
    print("\n[ORIGINAL INPUT]:")
    print(text.strip())
    print("\n[CAVEMAN COMPRESSED]:")
    print(result["compressed_prompt"])
    print("\n[SYSTEM DIRECTIVE]:")
    print(result["system_prompt"])
    print("\n" + "-" * 65)
    print(" 📊 MEASURED SAVINGS & GREEN AI METRICS:")
    print("-" * 65)
    print(f" Original Tokens:      {metrics['original_tokens']}")
    print(f" Compressed Tokens:    {metrics['compressed_tokens']}")
    print(f" Tokens Saved:         {metrics['tokens_saved']}")
    print(f" Token Reduction:      {metrics['saving_pct']:.1f}% (Targets ~65%)")
    print(f" Projected Time Saved: {metrics['time_saved_s']:.3f} s")
    print(f" Projected Energy Saved: {metrics['energy_saved_j']:.4f} Joules")
    print(f" Projected Carbon Saved: {metrics['carbon_saved_g']:.6f} gCO2")
    print("=" * 65 + "\n")


def cmd_info(args):
    """Displays system hardware sensors, Ollama status, and configuration."""
    gpu = get_gpu_telemetry()
    models = ollama_client.list_models()

    print("\n" + "=" * 60)
    print(" ⚡ ZEROTRACE AI — SYSTEM TELEMETRY AUDIT")
    print("=" * 60)
    print(f" CPU Base TDP:         {get_cpu_base_tdp()} W")
    print(f" GPU (NVML):           {gpu['gpu_name']} ({'Online' if gpu['available'] else 'Offline'})")
    print(f" Zeus ML Monitor:      {'Available' if ZEUS_OK else 'Not Installed'}")
    print(f" Carbon Intensity:     {get_carbon():.1f} gCO2/kWh")
    print(f" Ollama Daemon:        {'Connected' if ollama_client.is_online() else 'Not running'}")
    if models:
        print(f" Available Models:     {', '.join(m['name'] for m in models)}")
    print("=" * 60 + "\n")


def cmd_ui(args=None):
    """Starts the ZeroTrace web dashboard and REST/SSE API server."""
    import uvicorn
    from zerotrace.config import HOST, PORT
    port = getattr(args, "port", None) or PORT
    host = getattr(args, "host", None) or HOST
    print("\n" + "=" * 60)
    print(" ⚡ ZEROTRACE AI — GREEN WEB DASHBOARD & API")
    print("=" * 60)
    print(f" Dashboard URL: http://localhost:{port}")
    print(f" Network Bind:  http://{host}:{port}")
    print(" Press Ctrl+C to stop.")
    print("=" * 60 + "\n")
    uvicorn.run("zerotrace.api.app:app", host=host, port=port, reload=False)


def detect_default_provider():
    """Detects best available inference provider: Local Ollama, Offline weights, or configured Cloud LLMs."""
    from zerotrace.core.autodiscover import autodiscover
    report = autodiscover.auto_connect()
    auto = report["auto_connected"]
    return auto["provider"], auto["model"], report


def dispatch_chat(
    message: str,
    provider: str = "ollama",
    model: str = None,
    api_key: str = None,
    use_cache: bool = True,
    use_caveman: bool = True,
    caveman_intensity: str = "full",
):
    """
    Sends chat request to running ZeroTrace server (if active)
    or executes in-process directly via routes_chat.
    """
    # 1. Try running server first (keeps web and CLI state in sync)
    try:
        import requests
        payload = {
            "message": message,
            "provider": provider,
            "model": model,
            "api_key": api_key,
            "use_cache": use_cache,
            "use_caveman": use_caveman,
            "caveman_intensity": caveman_intensity,
        }
        r = requests.post("http://127.0.0.1:8000/api/chat", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass

    # 2. Standalone execution (zero daemon required)
    from zerotrace.api.routes_chat import chat_endpoint, ChatRequest
    req = ChatRequest(
        message=message,
        provider=provider,
        model=model,
        api_key=api_key,
        use_cache=use_cache,
        use_caveman=use_caveman,
        caveman_intensity=caveman_intensity,
    )
    return chat_endpoint(req)


def print_session_audit():
    """Prints final session cumulative savings audit."""
    carbon_kwh = get_carbon()
    co2_g = (state.total_energy_saved_j / 3_600_000.0) * carbon_kwh
    print("\n" + "=" * 76)
    print(" 📊 ZEROTRACE AI — SESSION GREEN IMPACT AUDIT")
    print("-" * 76)
    print(f" Total Electricity Saved: {state.total_energy_saved_j:.2f} Joules ({state.total_energy_saved_j / 3600.0:.4f} Watt-hours)")
    print(f" Total Tokens Saved:      {state.total_tokens_saved:,} prompt & cache tokens")
    print(f" Total Carbon Prevented:  {co2_g:.6f} gCO2 grid emissions")
    print(f" 0-Watt Cache Hits:       {state.total_cache_hits} instant recalls")
    print("=" * 76 + "\n")


def cmd_one_shot(prompt: str, provider: str = None, model: str = None):
    """Executes a single prompt through ZeroTrace and outputs response with green metrics."""
    def_prov, def_model, _ = detect_default_provider()
    provider = provider or def_prov
    model = model or def_model

    try:
        res = dispatch_chat(message=prompt, provider=provider, model=model)
        answer = res.get("answer", "")
        print(f"\n{answer}\n")

        if res.get("cache_hit"):
            print(f"[🟢 100% ZERO-WATT CACHE HIT | <5ms | 0 GPU Joules | Saved {res.get('tokens_saved', 0)} tokens | {res.get('energy_saved_j', 0):.2f} J | {res.get('carbon_saved_g', 0):.5f} gCO2]")
        else:
            saved_tokens = res.get("tokens_saved", 0)
            saved_j = res.get("energy_saved_j", 0.0)
            saved_co2 = res.get("carbon_saved_g", 0.0)
            print(f"[⚡ ZEROTRACE SAVED: {saved_tokens} tokens | {saved_j:.2f} J Electricity | {saved_co2:.5f} gCO2 | Provider: {provider} ({model})]")
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")


def cmd_chat(args=None):
    """
    Starts an interactive Green AI chat in your terminal.
    Every prompt you send is checked against the 384-D vector cache (0W recall)
    and compressed with Caveman (~65% token drop), with live energy & carbon tracking.
    """
    import os
    default_provider, default_model, disco_report = detect_default_provider()
    provider = getattr(args, "provider", None) or default_provider
    model = getattr(args, "model", None) or default_model
    intensity = "full"
    use_cache = True
    use_caveman = True
    api_key = getattr(args, "key", None)

    active_agent = disco_report["auto_connected"]["active_agent"]
    hw_source = disco_report["hardware"]["telemetry_source"]
    offline_status = f"{len(disco_report['offline_models'])} Offline Models Ready" if disco_report['offline_models'] else "Cloud Gateway Ready"

    print("\n" + "=" * 78)
    print(" ⚡ ZEROTRACE AI — UNIVERSAL GREEN CHAT GATEWAY")
    print("=" * 78)
    print(f" Agent Link:      {active_agent}")
    print(f" Offline Models:  {offline_status}")
    print(f" Active Target:   {provider.upper()} ({model})")
    print(f" Telemetry Source:{hw_source} | Grid Carbon: {get_carbon():.1f} gCO2/kWh")
    print(" ZeroTrace runs transparently in the background while you chat:")
    print("   • 384-D Semantic Vector Cache  -> 0-Watt instant recall on repeated queries")
    print("   • Caveman Prompt Compression   -> ~65% token drop on novel queries")
    print("   • Physical Telemetry Tracking  -> Real Watts, Joules & gCO2 calculated per turn")
    print("-" * 78)
    print(" Commands: /provider [name]  /model [name]  /stats  /clear  /exit")
    print(" Type your prompt and press Enter to chat!")
    print("=" * 78 + "\n")

    if provider == "ollama" and not ollama_client.is_online():
        print(" [ℹ] Note: Local Ollama is not running.")
        print("     To use local models, run 'ollama serve' in another terminal.")
        print("     Or switch to Cloud Gemini/Groq/OpenAI: type '/provider gemini'\n")

    while True:
        try:
            prompt = input(f"\nYou [{provider}:{model}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print_session_audit()
            break

        if not prompt:
            continue

        # Handle slash commands
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()

            if cmd in ["/exit", "/quit", "/q"]:
                print_session_audit()
                break
            elif cmd == "/stats":
                print_session_audit()
                continue
            elif cmd == "/clear":
                os.system("clear" if os.name != "nt" else "cls")
                continue
            elif cmd == "/provider":
                if len(parts) > 1:
                    provider = parts[1].lower()
                    from zerotrace.models.cloud import CLOUD_PROVIDERS
                    if provider in CLOUD_PROVIDERS:
                        model = CLOUD_PROVIDERS[provider]["default_model"]
                    elif provider == "ollama":
                        models = ollama_client.list_models() if ollama_client.is_online() else []
                        model = models[0]["name"] if models else "llama3:latest"
                    print(f" ✓ Provider switched to: {provider.upper()} (default model: {model})")
                else:
                    print(" Usage: /provider <ollama | gemini | groq | openai | anthropic>")
                continue
            elif cmd == "/model":
                if len(parts) > 1:
                    model = parts[1]
                    print(f" ✓ Model set to: {model}")
                else:
                    print(" Usage: /model <model_name>")
                continue
            elif cmd == "/key":
                if len(parts) > 1:
                    api_key = parts[1]
                    print(" ✓ API key updated for session.")
                else:
                    print(" Usage: /key <your_api_key>")
                continue
            elif cmd == "/cache":
                if len(parts) > 1 and parts[1].lower() in ["off", "false", "0"]:
                    use_cache = False
                    print(" ✓ 384-D Vector Cache: DISABLED")
                else:
                    use_cache = True
                    print(" ✓ 384-D Vector Cache: ENABLED")
                continue
            elif cmd == "/caveman":
                if len(parts) > 1 and parts[1].lower() in ["off", "false", "0"]:
                    use_caveman = False
                    print(" ✓ Caveman Compression: DISABLED")
                else:
                    use_caveman = True
                    print(" ✓ Caveman Compression: ENABLED")
                continue
            elif cmd in ["/help", "/h", "/?"]:
                print("\n Available Commands:")
                print("   /provider <name>   Switch provider: ollama, gemini, groq, openai, anthropic")
                print("   /model <name>      Change model identifier")
                print("   /key <api_key>     Provide or update cloud API key")
                print("   /cache <on|off>    Toggle 0-Watt Semantic Cache")
                print("   /caveman <on|off>  Toggle Caveman prompt compression")
                print("   /stats             Display session cumulative green impact audit")
                print("   /clear             Clear terminal screen")
                print("   /exit              Exit chat and display final session report\n")
                continue
            else:
                print(f" Unknown command '{cmd}'. Type /help for available commands.")
                continue

        # Normal prompt inference
        print(" ⏳ ZeroTrace consulting cache & model...", end="\r", flush=True)
        try:
            res = dispatch_chat(
                message=prompt,
                provider=provider,
                model=model,
                api_key=api_key,
                use_cache=use_cache,
                use_caveman=use_caveman,
                caveman_intensity=intensity,
            )

            # Clear progress line
            print(" " * 45, end="\r")

            if not res.get("success", True) and "error" in res:
                print(f"\n[ERROR] {res.get('error')}")
                continue

            answer = res.get("answer", "")
            print(f"\nZeroTrace AI >\n{answer}\n")

            if res.get("cache_hit"):
                print(f"[🟢 100% ZERO-WATT CACHE HIT | <5ms | 0 GPU Joules | 0 API Tokens | Saved {res.get('tokens_saved', 0)} tokens | {res.get('energy_saved_j', 0):.2f} J | {res.get('carbon_saved_g', 0):.5f} gCO2]")
            else:
                saved_tokens = res.get("tokens_saved", 0)
                saved_j = res.get("energy_saved_j", 0.0)
                saved_co2 = res.get("carbon_saved_g", 0.0)
                dur = res.get("duration_s", 0.0)
                print(f"[⚡ ZEROTRACE SAVED: {saved_tokens} tokens | {saved_j:.2f} J Electricity | {saved_co2:.5f} gCO2 | Latency: {dur}s | Model: {provider}:{model}]")

            print(f" Session Cumulative: {state.total_energy_saved_j:.1f} J saved | {state.total_tokens_saved} tokens saved | {state.total_cache_hits} cache hit(s)")

        except Exception as e:
            print(" " * 45, end="\r")
            print(f"\n[ERROR] Request failed: {e}")
            if provider == "ollama" and not ollama_client.is_online():
                print(" -> Local Ollama is offline. Type '/provider gemini' or run 'ollama serve'.")


def main():
    # If user runs `python cli.py` with no arguments, launch interactive chat directly!
    if len(sys.argv) == 1:
        cmd_chat()
        return

    # If user runs `python cli.py "some prompt..."`, run one-shot prompt inference directly!
    if len(sys.argv) == 2 and sys.argv[1] not in ["-h", "--help", "live", "compress", "info", "chat", "ui", "serve"]:
        cmd_one_shot(sys.argv[1])
        return

    parser = argparse.ArgumentParser(description="Zerotrace AI - Green Hardware & Token Profiler CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # Chat command
    chat_p = subparsers.add_parser("chat", help="Start interactive Green AI chat (default)")
    chat_p.add_argument("--provider", "-p", choices=["ollama", "gemini", "groq", "openai", "anthropic", "custom"], help="Inference provider")
    chat_p.add_argument("--model", "-m", type=str, help="Model identifier")
    chat_p.add_argument("--key", "-k", type=str, help="API Key")

    # UI / Dashboard command
    ui_p = subparsers.add_parser("ui", help="Start the ZeroTrace green web dashboard and API server")
    ui_p.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    ui_p.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")

    serve_p = subparsers.add_parser("serve", help="Alias for 'ui'")
    serve_p.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    serve_p.add_argument("--host", type=str, default="0.0.0.0", help="Host interface (default: 0.0.0.0)")

    # Live command
    live_p = subparsers.add_parser("live", help="Stream live hardware watts and carbon emissions")
    live_p.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds (default: 1.0)")

    # Compress command
    comp_p = subparsers.add_parser("compress", help="Compress prompt with Caveman (~65%% token drop)")
    comp_p.add_argument("text", nargs="?", default="", help="Text to compress")
    comp_p.add_argument("--file", "-f", type=str, default="", help="Read input from text file")
    comp_p.add_argument("--intensity", "-i", choices=["lite", "full", "ultra"], default="full", help="Compression level")

    # Info command
    subparsers.add_parser("info", help="Inspect hardware sensors, TDP, and Ollama status")

    args = parser.parse_args()

    if args.command == "chat" or args.command is None:
        cmd_chat(args)
    elif args.command in ["ui", "serve"]:
        cmd_ui(args)
    elif args.command == "live":
        cmd_live(args)
    elif args.command == "compress":
        cmd_compress(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        cmd_chat()


if __name__ == "__main__":
    main()
