from typing import Dict, Any, Tuple
from zerotrace.telemetry.hardware import apply_gpu_power_cap, reset_gpu_power_cap
from zerotrace.compression.caveman import CavemanCompressor
from zerotrace.compression.tokenizer import calculate_savings
from zerotrace.core.state import state
from zerotrace.models.ollama import ollama_client


def apply_token_limit(current_options: dict, limit: int = 80) -> dict:
    """Clamps maximum output tokens to prevent runaway repetitive generations."""
    new_opts = dict(current_options)
    new_opts["num_predict"] = limit
    return new_opts


def apply_context_reduction(current_options: dict, ctx: int = 512) -> dict:
    """Reduces context window size (num_ctx), drastically cutting KV cache power."""
    new_opts = dict(current_options)
    new_opts["num_ctx"] = ctx
    new_opts["num_predict"] = min(current_options.get("num_predict", 200), max(16, ctx // 4))
    return new_opts


def apply_gpu_routing(current_options: dict, mode: str = "gpu_lean") -> dict:
    """Controls layer offload between GPU VRAM and CPU threads."""
    new_opts = dict(current_options)
    if mode == "gpu_lean":
        new_opts["num_gpu"] = 99
        new_opts["num_thread"] = 2
    elif mode == "cpu_lean":
        new_opts["num_gpu"] = 0
        new_opts["num_thread"] = 4
    return new_opts


def apply_caveman_compression_optimization(
    current_options: dict,
    intensity: str = "full",
) -> Tuple[dict, str, str]:
    """
    Applies Caveman ~65% token compression optimization to Ollama inference.
    Configures options and system prompt to prune ~65% of generation tokens.
    """
    new_opts = dict(current_options)
    # With caveman compression, generation is naturally ~65% shorter, so clamp num_predict proportionally
    curr_predict = current_options.get("num_predict", 250)
    new_opts["num_predict"] = max(40, int(curr_predict * 0.45))

    system_prompt = CavemanCompressor.get_system_prompt(intensity=intensity)
    profile = f"caveman_{intensity}"
    description = (
        f"Caveman {intensity.upper()} compression active: strips filler, pleasantries, and articles. "
        "Targets ~65% token consumption reduction while preserving 100% technical accuracy."
    )
    return new_opts, profile, description


def bake_ollama_optimization(model_name: str, options: dict, system_prompt: str = None) -> bool:
    """Bakes current optimization parameters into the local Ollama Modelfile."""
    try:
        modelfile_lines = [f"FROM {model_name}"]
        if system_prompt:
            modelfile_lines.append(f'SYSTEM """{system_prompt}"""')
        for key, value in options.items():
            modelfile_lines.append(f"PARAMETER {key} {value}")

        ollama_client.create_model(name=model_name, modelfile="\n".join(modelfile_lines))
        return True
    except Exception as e:
        print(f"[WARNING] Failed to bake model into Ollama: {e}")
        return False
