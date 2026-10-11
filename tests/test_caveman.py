import pytest
from fastapi.testclient import TestClient

from server import app
from zerotrace.compression.caveman import CavemanCompressor
from zerotrace.compression.tokenizer import estimate_token_count, calculate_savings
from zerotrace.compression.prompt_optimizer import PromptOptimizer

client = TestClient(app)


def test_caveman_strips_pleasantries_and_filler():
    verbose_text = (
        "Sure! I'd be happy to help you with that. "
        "Basically, in order to fix this problem, you need to make use of connection pooling. "
        "In conclusion, hope this helps!"
    )
    compressed = CavemanCompressor.compress(verbose_text, intensity="full")
    assert "happy to help" not in compressed.lower()
    assert "basically" not in compressed.lower()
    assert "in order to" not in compressed.lower()
    assert "in conclusion" not in compressed.lower()
    assert "pool" in compressed.lower()


def test_caveman_drops_articles_in_full_mode():
    text = "The server requires a restart after an update."
    compressed = CavemanCompressor.compress(text, intensity="full")
    # In full mode, standalone 'the', 'a', 'an' are pruned
    words = compressed.lower().split()
    assert "the" not in words
    assert "a" not in words
    assert "an" not in words
    assert "server" in compressed.lower()
    assert "restart" in compressed.lower()


def test_caveman_preserves_code_blocks_verbatim():
    code_content = "def calculate_loss(y_true, y_pred):\n    return ((y_true - y_pred) ** 2).mean()"
    text = f"Sure! Here is the python code for your loss function:\n```python\n{code_content}\n```\nLet me know if you need anything else!"

    compressed = CavemanCompressor.compress(text, intensity="full")
    assert code_content in compressed
    assert "```python" in compressed
    assert "```" in compressed
    assert "happy" not in compressed.lower()
    assert "anything else" not in compressed.lower()


def test_caveman_preserves_urls_and_technical_terms():
    text = "Please check the official documentation at https://pytorch.org/docs/stable/index.html for details on CUDA_VISIBLE_DEVICES."
    compressed = CavemanCompressor.compress(text, intensity="full")
    assert "https://pytorch.org/docs/stable/index.html" in compressed
    assert "CUDA_VISIBLE_DEVICES" in compressed


def test_caveman_measured_token_reduction():
    verbose_prompt = (
        "Hello assistant! Could you please give a detailed explanation with at least 250 words "
        "on how transformer neural networks work? As an AI model, please cover the self-attention mechanism, "
        "the positional encoding, the feed-forward layers, and the training phase. "
        "It is really important that you make use of clear technical terms and basically explain everything thoroughly. "
        "Thank you so much, I look forward to your helpful answer!"
    )
    result = PromptOptimizer.optimize(verbose_prompt, intensity="full")
    metrics = result["metrics"]

    assert metrics["original_tokens"] > metrics["compressed_tokens"]
    assert metrics["tokens_saved"] > 0
    # Verifies dramatic token savings (>40% to ~65% on typical verbose inputs)
    assert metrics["saving_pct"] >= 45.0
    assert metrics["energy_saved_j"] > 0


def test_caveman_intensity_levels():
    text = "The database connection failed due to the fact that the timeout was exceeded."

    lite = CavemanCompressor.compress(text, intensity="lite")
    full = CavemanCompressor.compress(text, intensity="full")
    ultra = CavemanCompressor.compress(text, intensity="ultra")

    # Lite keeps articles, full drops articles
    assert len(full) <= len(lite)
    assert len(ultra) <= len(full)


def test_caveman_system_prompts():
    prompt_lite = CavemanCompressor.get_system_prompt("lite")
    prompt_full = CavemanCompressor.get_system_prompt("full")
    prompt_ultra = CavemanCompressor.get_system_prompt("ultra")

    assert "token-saver" in prompt_lite.lower()
    assert "caveman" in prompt_full.lower()
    assert "telegraphic" in prompt_ultra.lower()


def test_compress_api_endpoint():
    payload = {
        "text": "Certainly, here is the explanation. In order to optimize memory, you should decrease the batch size.",
        "intensity": "full",
    }
    response = client.post("/api/compress", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "compressed_text" in data
    assert "metrics" in data
    assert data["metrics"]["saving_pct"] > 0
    assert "system_prompt" in data


def test_caveman_stats_api_endpoint():
    response = client.get("/api/caveman/stats")
    assert response.status_code == 200
    data = response.json()
    assert "total_tokens_saved" in data
    assert "total_energy_saved_j" in data
