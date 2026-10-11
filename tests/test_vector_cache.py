import tempfile
import numpy as np
import pytest

from zerotrace.core.vector_datastore import VectorDatastore
from zerotrace.core.embedder import FastSemanticEmbedder
from zerotrace.core.knn_combiner import KNNCombiner


def test_vector_datastore_normalization():
    store = VectorDatastore(dimension=4)
    raw = np.array([3.0, 4.0, 0.0, 0.0])  # Length is 5.0
    norm = store._normalize(raw)
    assert np.linalg.norm(norm) == pytest.approx(1.0, rel=1e-5)
    assert norm[0] == pytest.approx(0.6, rel=1e-5)
    assert norm[1] == pytest.approx(0.8, rel=1e-5)


def test_vector_datastore_add_and_search():
    store = VectorDatastore(dimension=384)
    embedder = FastSemanticEmbedder(dimension=384)

    # Add records
    v1 = embedder.embed("How do I connect to PostgreSQL database in Python?")
    store.add(
        query="How do I connect to PostgreSQL database in Python?",
        answer="Use psycopg2: `conn = psycopg2.connect(...)`",
        vector=v1,
    )

    v2 = embedder.embed("What is the capital city of France?")
    store.add(
        query="What is the capital city of France?",
        answer="Paris is the capital of France.",
        vector=v2,
    )

    assert store.size() == 2

    # Search with similar query
    query_vec = embedder.embed("How to connect PostgreSQL with Python script?")
    results = store.search(query_vec, top_k=2)

    assert len(results) == 2
    top_score, top_record = results[0]
    second_score = results[1][0]
    assert "psycopg2" in top_record["answer"]
    assert top_score > 0.65  # High semantic similarity
    assert top_score > second_score + 0.30  # Strongly discriminates from unrelated query


def test_knn_combiner_decision():
    combiner = KNNCombiner(cache_threshold=0.85, fallback_threshold=0.60)

    # 1. High confidence match
    high_match = [(0.92, {"query": "Q1", "answer": "Answer 1"})]
    eval_hit = combiner.evaluate(high_match, default_power_w=100.0, est_llm_time_s=2.0)
    assert eval_hit["hit"] is True
    assert eval_hit["lambda_gate"] == 1.0
    assert eval_hit["energy_saved_j"] == 200.0

    # 2. Low confidence match (Miss -> Fallback)
    low_match = [(0.45, {"query": "Q2", "answer": "Answer 2"})]
    eval_miss = combiner.evaluate(low_match)
    assert eval_miss["hit"] is False
    assert eval_miss["lambda_gate"] == 0.0
    assert eval_miss["energy_saved_j"] == 0.0


def test_datastore_persistence():
    store = VectorDatastore(dimension=384)
    rng = np.random.RandomState(42)
    fake_vec = rng.randn(384).astype(np.float32)

    store.add("Sample query", "Sample answer", fake_vec)

    with tempfile.NamedTemporaryFile(suffix=".npz") as f:
        store.save(f.name)
        new_store = VectorDatastore(dimension=384)
        success = new_store.load(f.name)
        assert success is True
        assert new_store.size() == 1
        assert new_store.records[0]["query"] == "Sample query"
