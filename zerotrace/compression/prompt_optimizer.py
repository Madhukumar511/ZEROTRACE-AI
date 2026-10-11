from typing import Dict, Any, Optional
from .caveman import CavemanCompressor
from .tokenizer import estimate_token_count, calculate_savings


class PromptOptimizer:
    """
    High-level prompt compression and token optimization coordinator.
    Optimizes user prompts and system messages to slash inference tokens and energy.
    """

    @classmethod
    def optimize(
        cls,
        prompt: str,
        intensity: str = "full",
        power_w: float = 65.0,
        tok_per_sec: float = 25.0,
        carbon_gco2_kwh: float = 214.0,
    ) -> Dict[str, Any]:
        compressed = CavemanCompressor.compress(prompt, intensity=intensity)
        savings = calculate_savings(
            original_text=prompt,
            compressed_text=compressed,
            power_w=power_w,
            tok_per_sec=tok_per_sec,
            carbon_gco2_kwh=carbon_gco2_kwh,
        )
        system_prompt = CavemanCompressor.get_system_prompt(intensity=intensity)

        return {
            "original_prompt": prompt,
            "compressed_prompt": compressed,
            "intensity": intensity,
            "system_prompt": system_prompt,
            "metrics": savings,
        }
