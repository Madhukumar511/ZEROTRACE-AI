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


def main():
    parser = argparse.ArgumentParser(description="Zerotrace AI - Green Hardware & Token Profiler CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

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

    if args.command == "live":
        cmd_live(args)
    elif args.command == "compress":
        cmd_compress(args)
    elif args.command == "info":
        cmd_info(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
