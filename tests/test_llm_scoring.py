"""
Unit tests for py/llm_scoring.py

Run with the Python interpreter managed by reticulate (py_require handles
package installation). To find the path from R:
  reticulate::py_config()$python

Or run directly:
  /path/to/reticulate/python tests/test_llm_scoring.py

Validates MLM and AR surprisal against reference values computed
with direct transformers calls (verified correct on CPU).

Reference values (bert-base-uncased MLM, "The cat eats the fish"):
  The:     0.0399
  cat:     6.9309
  eats:    7.0374
  the:     1.9208
  fish:   14.6581

Reference values (pagnol-small AR, "Le chat mange le poisson"):
  Le:      first token (convention-dependent)
  chat:   12.2677
  mange:   9.3294
  le:      4.0631
  poisson: 6.5276
"""

import sys
import math
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "py"))
import llm_scoring


# ── Helpers ──────────────────────────────────────────────────────────

def assert_close(actual, expected, tol, label=""):
    diff = abs(actual - expected)
    assert diff <= tol, (
        f"{label}: expected {expected:.4f}, got {actual:.4f}, diff={diff:.4f} > tol={tol}"
    )


# ── MLM Tests ────────────────────────────────────────────────────────

def test_mlm_surprisal():
    """Test MLM surprisal with bert-base-uncased on English sentence."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model("google-bert/bert-base-uncased", mode="mlm")

    tokens = ["The", "cat", "eats", "the", "fish"]
    result = llm_scoring.score_masked_lm_tokens(tokens, temperature=1.0, batch_size=0)

    expected = [0.0399, 6.9309, 7.0374, 1.9208, 14.6581]

    assert len(result["word_surprisals"]) == len(tokens)
    assert len(result["word_entropies"]) == len(tokens)
    assert result["word_token_counts"] == [1, 1, 1, 1, 1]

    for i, tok in enumerate(tokens):
        assert_close(result["word_surprisals"][i], expected[i], tol=0.01, label=tok)
        assert result["word_entropies"][i] > 0, f"{tok}: entropy should be > 0"
        assert math.isfinite(result["word_entropies"][i]), f"{tok}: entropy should be finite"


def test_mlm_empty_input():
    """Empty input should return empty lists."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model("google-bert/bert-base-uncased", mode="mlm")

    result = llm_scoring.score_masked_lm_tokens([])
    assert result["word_surprisals"] == []
    assert result["word_entropies"] == []
    assert result["word_token_counts"] == []


def test_mlm_short_sentence():
    """Single-word sentence should work."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model("google-bert/bert-base-uncased", mode="mlm")

    result = llm_scoring.score_masked_lm_tokens(["Hello"])
    assert len(result["word_surprisals"]) == 1
    assert math.isfinite(result["word_surprisals"][0])
    assert result["word_surprisals"][0] > 0


def test_mlm_subword_alignment():
    """Words split into subwords should still produce one surprisal per word."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model("google-bert/bert-base-uncased", mode="mlm")

    # "methylenedioxy" will be split into multiple subwords by BERT
    tokens = ["This", "is", "methylenedioxy"]
    result = llm_scoring.score_masked_lm_tokens(tokens)

    assert len(result["word_surprisals"]) == 3
    assert len(result["word_entropies"]) == 3
    assert result["word_token_counts"][2] >= 2, (
        f"Expected 'methylenedioxy' to have >=2 subwords, got {result['word_token_counts'][2]}"
    )
    assert result["word_surprisals"][2] > 0
    assert math.isfinite(result["word_surprisals"][2])


def test_mlm_word_tokens_returned():
    """word_tokens and subword_tokens should be present in result."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model("google-bert/bert-base-uncased", mode="mlm")

    tokens = ["The", "cat", "eats"]
    result = llm_scoring.score_masked_lm_tokens(tokens)

    assert "word_tokens" in result, "word_tokens missing from result"
    assert "subword_tokens" in result, "subword_tokens missing from result"
    assert len(result["word_tokens"]) == 3
    assert result["word_tokens"] == tokens


def test_mlm_pll_within_word_l2r():
    """within_word_l2r mode should give different scores than original for multi-subword words."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model("google-bert/bert-base-uncased", mode="mlm")

    # "methylenedioxy" splits into multiple subwords with BERT (verified in test_mlm_subword_alignment)
    tokens = ["This", "is", "methylenedioxy"]

    result_orig = llm_scoring.score_masked_lm_tokens(tokens, pll_mode="original")
    result_l2r  = llm_scoring.score_masked_lm_tokens(tokens, pll_mode="within_word_l2r")

    assert len(result_orig["word_surprisals"]) == 3
    assert len(result_l2r["word_surprisals"]) == 3

    # Single-subword words should be identical across modes
    assert_close(result_orig["word_surprisals"][0], result_l2r["word_surprisals"][0],
                 tol=1e-6, label="This (single subword)")
    assert_close(result_orig["word_surprisals"][1], result_l2r["word_surprisals"][1],
                 tol=1e-6, label="is (single subword)")

    # Multi-subword word should differ between modes
    assert result_orig["word_token_counts"][2] >= 2, (
        "'methylenedioxy' should have >= 2 subwords"
    )
    assert result_orig["word_surprisals"][2] != result_l2r["word_surprisals"][2], (
        "within_word_l2r should give different surprisal for multi-subword words"
    )


def test_mlm_pll_invalid_mode():
    """Invalid pll_mode should raise ValueError."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model("google-bert/bert-base-uncased", mode="mlm")

    try:
        llm_scoring.score_masked_lm_tokens(["test"], pll_mode="invalid")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_mlm_raw_sentence():
    """is_split_into_words=False should accept a raw string and return word-level scores."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model("google-bert/bert-base-uncased", mode="mlm")

    sentence = "The cat eats the fish"
    result = llm_scoring.score_masked_lm_tokens(sentence, is_split_into_words=False)

    # BERT tokenizer treats "The cat eats the fish" as 5 words
    assert len(result["word_surprisals"]) == 5
    assert len(result["word_entropies"]) == 5
    assert len(result["word_tokens"]) == 5

    for i, surp in enumerate(result["word_surprisals"]):
        assert math.isfinite(surp), f"word {i}: surprisal should be finite"
        assert surp >= 0, f"word {i}: surprisal should be >= 0"


def test_mlm_raw_empty_string():
    """Empty string with is_split_into_words=False should return empty result."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model("google-bert/bert-base-uncased", mode="mlm")

    result = llm_scoring.score_masked_lm_tokens("   ", is_split_into_words=False)
    assert result["word_surprisals"] == []
    assert result["word_tokens"] == []


def test_mlm_batching_consistency():
    """Batched and single-item scoring should give identical results."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model("google-bert/bert-base-uncased", mode="mlm")

    tokens = ["The", "cat", "eats", "the", "fish"]

    result_batch1 = llm_scoring.score_masked_lm_tokens(tokens, batch_size=1)
    result_batch0 = llm_scoring.score_masked_lm_tokens(tokens, batch_size=0)

    for i, tok in enumerate(tokens):
        assert_close(result_batch1["word_surprisals"][i],
                     result_batch0["word_surprisals"][i],
                     tol=1e-4, label=f"{tok} surprisal")
        assert_close(result_batch1["word_entropies"][i],
                     result_batch0["word_entropies"][i],
                     tol=1e-4, label=f"{tok} entropy")


# ── AR Tests ─────────────────────────────────────────────────────────

def test_ar_surprisal():
    """Test AR surprisal with pagnol-small on French sentence."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model(
        "lightonai/pagnol-small", mode="ar", add_prefix_space=True
    )

    tokens = ["Le", "chat", "mange", "le", "poisson"]
    result = llm_scoring.score_autoregressive_tokens(tokens, temperature=1.0)

    # Reference values (excluding first token — convention-dependent)
    expected_from_2nd = [12.2677, 9.3294, 4.0631, 6.5276]

    assert len(result["word_surprisals"]) == 5
    assert len(result["word_entropies"]) == 5

    for i, (tok, exp) in enumerate(
        zip(tokens[1:], expected_from_2nd), start=1
    ):
        assert_close(result["word_surprisals"][i], exp, tol=0.5, label=tok)

    for i, tok in enumerate(tokens):
        assert result["word_surprisals"][i] >= 0, f"{tok}: surprisal should be >= 0"
        assert math.isfinite(result["word_surprisals"][i]), f"{tok}: surprisal should be finite"


def test_ar_empty_input():
    """Empty input should return empty lists."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model(
        "lightonai/pagnol-small", mode="ar", add_prefix_space=True
    )

    result = llm_scoring.score_autoregressive_tokens([])
    assert result["word_surprisals"] == []
    assert result["word_entropies"] == []


def test_ar_entropy_positive():
    """Entropy should be positive for all non-first tokens."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model(
        "lightonai/pagnol-small", mode="ar", add_prefix_space=True
    )

    tokens = ["Elle", "parle", "bien"]
    result = llm_scoring.score_autoregressive_tokens(tokens)

    for i in range(1, len(tokens)):
        assert result["word_entropies"][i] > 0, (
            f"{tokens[i]}: entropy should be > 0, got {result['word_entropies'][i]}"
        )


def test_ar_word_tokens_returned():
    """word_tokens and subword_tokens should be present in AR result."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model(
        "lightonai/pagnol-small", mode="ar", add_prefix_space=True
    )

    tokens = ["Le", "chat", "mange"]
    result = llm_scoring.score_autoregressive_tokens(tokens)

    assert "word_tokens" in result, "word_tokens missing from AR result"
    assert "subword_tokens" in result, "subword_tokens missing from AR result"
    assert len(result["word_tokens"]) == 3


def test_ar_raw_sentence():
    """is_split_into_words=False should work for AR models."""
    os.environ["ALSIO_FORCE_CPU"] = "1"
    llm_scoring.load_llm_model(
        "lightonai/pagnol-small", mode="ar", add_prefix_space=True
    )

    sentence = "Le chat mange le poisson"
    result = llm_scoring.score_autoregressive_tokens(sentence, is_split_into_words=False)

    assert len(result["word_surprisals"]) == 5
    assert len(result["word_tokens"]) == 5

    for i, surp in enumerate(result["word_surprisals"]):
        assert math.isfinite(surp), f"word {i}: surprisal should be finite"
        assert surp >= 0, f"word {i}: surprisal should be >= 0"


# ── Run all tests ────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_mlm_surprisal,
        test_mlm_empty_input,
        test_mlm_short_sentence,
        test_mlm_subword_alignment,
        test_mlm_word_tokens_returned,
        test_mlm_pll_within_word_l2r,
        test_mlm_pll_invalid_mode,
        test_mlm_raw_sentence,
        test_mlm_raw_empty_string,
        test_mlm_batching_consistency,
        test_ar_surprisal,
        test_ar_empty_input,
        test_ar_entropy_positive,
        test_ar_word_tokens_returned,
        test_ar_raw_sentence,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
