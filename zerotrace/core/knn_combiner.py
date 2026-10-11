import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class KNNCombiner:
    """
    Adaptive kNN Combiner and Gating Gate.
    Inspired by Stanford/Meta kNN-LM probability interpolation.
    Evaluates nearest neighbors and determines if query can be answered directly
    from cached token memory (λ -> 1.0) or must fall back to the main LLM (λ -> 0.0).
    """
    def __init__(
        self,
        cache_threshold: float = 0.88,
        fallback_threshold: float = 0.70,
        temperature: float = 0.05,
    ):
        self.cache_threshold = cache_threshold
        self.fallback_threshold = fallback_threshold
        self.temperature = temperature

    def evaluate(
        self,
        neighbors: List[Tuple[float, Dict[str, Any]]],
        default_power_w: float = 65.0,
        est_llm_time_s: float = 2.5,
    ) -> Dict[str, Any]:
        """
        Evaluates nearest neighbor matches.
        
        Args:
            neighbors: List of (similarity_score, record) tuples from VectorDatastore.search()
            default_power_w: Baseline GPU/CPU wattage for estimating energy saved.
            est_llm_time_s: Average inference time avoided on cache hit.
        """
        if not neighbors:
            return {
                "hit": False,
                "lambda_gate": 0.0,
                "best_match": None,
                "similarity_score": 0.0,
                "confidence": 0.0,
                "energy_saved_j": 0.0,
            }

        scores = np.array([n[0] for n in neighbors], dtype=np.float32)
        best_score = float(scores[0])
        best_record = neighbors[0][1]

        # Temperature-scaled softmax over candidate similarities
        # P_i ∝ exp(similarity_i / T)
        exp_scores = np.exp((scores - np.max(scores)) / max(self.temperature, 1e-4))
        probs = exp_scores / np.sum(exp_scores)
        top_prob = float(probs[0])

        # Adaptive Lambda Gating Decision:
        # λ = 1.0 -> 100% cache prediction (Zero LLM compute)
        # λ = 0.0 -> 100% full LLM fallback
        if best_score >= self.cache_threshold:
            lambda_gate = 1.0
            hit = True
        elif best_score <= self.fallback_threshold:
            lambda_gate = 0.0
            hit = False
        else:
            # Linear ramp between fallback and cache threshold
            lambda_gate = float((best_score - self.fallback_threshold) / (self.cache_threshold - self.fallback_threshold))
            hit = lambda_gate >= 0.75

        # Energy saved if hit: Power (Watts) * Time (Seconds) avoided
        energy_saved_j = round(default_power_w * est_llm_time_s, 3) if hit else 0.0

        return {
            "hit": hit,
            "lambda_gate": round(lambda_gate, 3),
            "best_match": best_record if hit else None,
            "similarity_score": round(best_score, 4),
            "confidence": round(top_prob * best_score, 4),
            "energy_saved_j": energy_saved_j,
            "candidates_count": len(neighbors),
        }
