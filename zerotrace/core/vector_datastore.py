import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class VectorDatastore:
    """
    In-Memory Vector Datastore for Green Semantic Caching.
    Inspired by GPTCache and kNN-LM.
    Maintains 384-dimensional normalized vectors alongside text records.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.embeddings: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self.records: List[Dict[str, Any]] = []

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """
        Normalizes vector to unit length (L2 norm = 1.0).
        When vectors have length 1.0, Cosine Similarity is simply the Dot Product!
        """
        vec = np.asarray(vec, dtype=np.float32).flatten()
        if vec.shape[0] != self.dimension:
            raise ValueError(f"Vector dimension mismatch: expected {self.dimension}, got {vec.shape[0]}")
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            return np.zeros_like(vec)
        return vec / norm

    def add(self, query: str, answer: str, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Appends a verified query, answer, and its semantic vector to the datastore.
        Returns the index of the newly added record.
        """
        norm_vec = self._normalize(vector).reshape(1, self.dimension)
        if self.embeddings.shape[0] == 0:
            self.embeddings = norm_vec
        else:
            self.embeddings = np.vstack([self.embeddings, norm_vec])

        record = {
            "query": query,
            "answer": answer,
            "metadata": metadata or {},
        }
        self.records.append(record)
        return len(self.records) - 1

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """
        Computes cosine similarities against all cached vectors in one matrix dot product.
        Returns list of (similarity_score, record) tuples sorted descending by score.
        """
        if self.embeddings.shape[0] == 0:
            return []

        norm_q = self._normalize(query_vector)  # Shape (384,)

        # Dot product against all rows: (N, 384) @ (384,) -> (N,) similarities in [-1.0, 1.0]
        similarities = np.dot(self.embeddings, norm_q)

        # Clip values to [-1.0, 1.0] to prevent floating point inaccuracies
        similarities = np.clip(similarities, -1.0, 1.0)

        # Get top-k indices
        k = min(top_k, len(self.records))
        # Use argpartition for fast top-k or argsort
        sorted_indices = np.argsort(-similarities)[:k]

        results = []
        for idx in sorted_indices:
            results.append((float(similarities[idx]), self.records[idx]))

        return results

    def size(self) -> int:
        return len(self.records)

    def clear(self):
        self.embeddings = np.empty((0, self.dimension), dtype=np.float32)
        self.records = []

    def save(self, filepath: str):
        """Persists datastore vectors and records to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            embeddings=self.embeddings,
            records=np.array(json.dumps(self.records)),
            dimension=np.array(self.dimension),
        )

    def load(self, filepath: str) -> bool:
        """Loads persisted datastore from disk."""
        if not os.path.exists(filepath):
            return False
        try:
            data = np.load(filepath, allow_pickle=True)
            self.dimension = int(data["dimension"])
            self.embeddings = data["embeddings"].astype(np.float32)
            self.records = json.loads(str(data["records"]))
            return True
        except Exception:
            return False
