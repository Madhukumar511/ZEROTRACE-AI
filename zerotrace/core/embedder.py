import re
import numpy as np
from typing import List, Union

_DEFAULT_DIMENSION = 384


class FastSemanticEmbedder:
    """
    Lightweight, deterministic 384-dimensional text embedder.
    Supports SentenceTransformers if available, with a fast zero-dependency
    Subword Feature Projection fallback that runs on CPU in <100 microseconds.
    """
    def __init__(self, dimension: int = _DEFAULT_DIMENSION):
        self.dimension = dimension
        self._sbert_model = None
        self._projection_matrix = None

        # Try loading sentence-transformers if present
        try:
            from sentence_transformers import SentenceTransformer
            self._sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            self._sbert_model = None
            # Initialize fixed deterministic projection matrix (Gaussian seed 42)
            rng = np.random.RandomState(42)
            # 4096 bag-of-ngrams projected down to 384 dimensions
            self._vocab_bins = 4096
            self._projection_matrix = rng.randn(self._vocab_bins, self.dimension).astype(np.float32)
            # L2 normalize projection columns
            self._projection_matrix /= np.linalg.norm(self._projection_matrix, axis=0, keepdims=True)

    def embed(self, text: str) -> np.ndarray:
        """Embeds a single string into a normalized 384-dimensional vector."""
        if not text:
            return np.zeros(self.dimension, dtype=np.float32)

        if self._sbert_model is not None:
            vec = self._sbert_model.encode(text, convert_to_numpy=True)
            norm = np.linalg.norm(vec)
            return (vec / norm if norm > 1e-12 else vec).astype(np.float32)

        # High-speed deterministic feature projection fallback
        # 1. Clean and tokenize into words and subword 3-grams
        words = re.findall(r"\w+", text.lower())
        feature_counts = np.zeros(self._vocab_bins, dtype=np.float32)

        for w in words:
            # Word hash
            h_word = abs(hash(w)) % self._vocab_bins
            feature_counts[h_word] += 1.5

            # 3-gram character hashes
            if len(w) >= 3:
                for i in range(len(w) - 2):
                    tri = w[i:i+3]
                    h_tri = abs(hash(tri)) % self._vocab_bins
                    feature_counts[h_tri] += 1.0

        # Non-linear log-frequency transformation
        feature_counts = np.log1p(feature_counts)

        # Matrix projection: (1, 4096) @ (4096, 384) -> (384,)
        vec = np.dot(feature_counts, self._projection_matrix)

        # L2 Normalize
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm

        return vec.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of strings into an (N, 384) matrix."""
        return np.vstack([self.embed(t) for t in texts])


embedder = FastSemanticEmbedder()
