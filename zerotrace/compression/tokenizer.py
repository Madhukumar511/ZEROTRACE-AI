import re
from typing import Dict, Any, Optional

# Regex pattern for subword / BPE-style token splitting in standard Python re
_TOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)|[a-zA-Z0-9_]+|[^\s\w]+|\s+"""
)


def estimate_token_count(text: str) -> int:
    """
    Estimates subword token count for a given text string.
    Accurately accounts for words, subwords, whitespace chunks, and punctuation symbols.
    """
    if not text:
        return 0

    try:
        tokens = _TOKEN_PATTERN.findall(text)
    except Exception:
        tokens = text.split()

    # Subword heuristic: Words longer than 5 characters typically decompose into ~1 token per 4 chars
    total_tokens = 0
    for tok in tokens:
        stripped = tok.strip()
        if not stripped:
            total_tokens += 1
        elif len(stripped) <= 5:
            total_tokens += 1
        else:
            total_tokens += max(1, (len(stripped) + 3) // 4)

    return total_tokens


def calculate_savings(
    original_text: str,
    compressed_text: str,
    power_w: float = 65.0,
    tok_per_sec: float = 25.0,
    carbon_gco2_kwh: float = 214.0,
) -> Dict[str, Any]:
    """
    Computes token reduction percentage and projected hardware energy and carbon savings.
    """
    orig_tokens = estimate_token_count(original_text)
    comp_tokens = estimate_token_count(compressed_text)
    saved_tokens = max(0, orig_tokens - comp_tokens)

    saving_pct = round((saved_tokens / max(orig_tokens, 1)) * 100.0, 1) if orig_tokens > 0 else 0.0

    # Physical telemetry math:
    # Time (sec) saved = tokens_saved / tokens_per_second
    # Energy (Joules) saved = Power (Watts) * Time (sec)
    # Carbon (gCO2) saved = (Energy (J) / 3,600,000) * Carbon_Intensity (gCO2/kWh)
    eff_tok_sec = max(tok_per_sec, 1.0)
    time_saved_s = saved_tokens / eff_tok_sec
    energy_saved_j = power_w * time_saved_s
    carbon_saved_g = (energy_saved_j / 3_600_000.0) * carbon_gco2_kwh

    return {
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "tokens_saved": saved_tokens,
        "saving_pct": saving_pct,
        "time_saved_s": round(time_saved_s, 3),
        "energy_saved_j": round(energy_saved_j, 4),
        "carbon_saved_g": round(carbon_saved_g, 6),
    }
