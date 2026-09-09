"""Tests for scripts/sparse_encoder.py — no external dependencies."""
import pytest

from scripts.sparse_encoder import (
    VOCAB_SIZE,
    _hash_token,
    _tokenize,
    encode_document,
    encode_query,
)


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


def test_tokenize_lowercases():
    assert _tokenize("Marketing") == ["marketing"]


def test_tokenize_removes_stopwords():
    tokens = _tokenize("the quick brown fox")
    assert "the" not in tokens
    assert "brown" in tokens
    assert "fox" in tokens


def test_tokenize_min_length_two():
    tokens = _tokenize("a ab abc")
    assert "a" not in tokens
    assert "ab" in tokens
    assert "abc" in tokens


def test_tokenize_numbers_survive():
    # Marketing Research uses numeric tokens: "n=200", "95% CI", "1-7" scale.
    tokens = _tokenize("n=200 Likert 1-7 scale 95% CI")
    assert "200" in tokens
    assert "95" in tokens


def test_tokenize_empty():
    assert _tokenize("") == []


def test_tokenize_whitespace_only():
    assert _tokenize("   ") == []


# ---------------------------------------------------------------------------
# _hash_token
# ---------------------------------------------------------------------------


def test_hash_deterministic():
    # FNV-1a must be stable across calls (no random salt like Python's hash()).
    assert _hash_token("marketing") == _hash_token("marketing")


def test_hash_different_tokens_differ():
    assert _hash_token("marketing") != _hash_token("research")


def test_hash_in_range():
    for token in ["alpha", "beta", "conjoint", "cluster", "99", "ab"]:
        h = _hash_token(token)
        assert 0 <= h < VOCAB_SIZE, f"Hash for {token!r} out of range: {h}"


# ---------------------------------------------------------------------------
# encode_document
# ---------------------------------------------------------------------------


def test_encode_document_empty_string():
    indices, values = encode_document("")
    assert indices == [] and values == []


def test_encode_document_stopwords_only():
    indices, values = encode_document("the and or but")
    assert indices == [] and values == []


def test_encode_document_returns_sorted_indices():
    indices, _ = encode_document("conjoint analysis technique market research")
    assert indices == sorted(indices)


def test_encode_document_values_positive():
    _, values = encode_document("survey design survey method")
    assert all(v > 0 for v in values)


def test_encode_document_unique_indices():
    indices, _ = encode_document("word word word another another")
    assert len(indices) == len(set(indices))


def test_encode_document_lengths_match():
    indices, values = encode_document("market research brand loyalty")
    assert len(indices) == len(values)


def test_encode_document_tf_reflected_in_values():
    # "survey" appears 3×; "design" 1×. Their bucket values should reflect TF.
    # Guard against the astronomically unlikely hash collision.
    survey_idx = _hash_token("survey")
    design_idx = _hash_token("design")
    if survey_idx == design_idx:
        pytest.skip("Hash collision between 'survey' and 'design'")
    indices, values = encode_document("survey survey survey design")
    bucket = dict(zip(indices, values))
    assert bucket.get(survey_idx, 0) > bucket.get(design_idx, 0)


# ---------------------------------------------------------------------------
# encode_query
# ---------------------------------------------------------------------------


def test_encode_query_empty_string():
    indices, values = encode_query("")
    assert indices == [] and values == []


def test_encode_query_values_are_one():
    _, values = encode_query("what is conjoint analysis")
    assert all(v == 1.0 for v in values)


def test_encode_query_deduplicates_repeated_token():
    # A term repeated should still occupy exactly one bucket.
    indices_once, _ = encode_query("survey")
    indices_thrice, _ = encode_query("survey survey survey")
    assert indices_once == indices_thrice


def test_encode_query_sorted_indices():
    indices, _ = encode_query("cluster analysis market segmentation")
    assert indices == sorted(indices)


def test_encode_query_no_duplicate_indices():
    indices, _ = encode_query("research research research method method")
    assert len(indices) == len(set(indices))


def test_encode_query_lengths_match():
    indices, values = encode_query("validity reliability measurement")
    assert len(indices) == len(values)
