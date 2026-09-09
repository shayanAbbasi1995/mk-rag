"""Tests for scripts/chunker.py regex constants and pure-Python helpers.

The entire module is skipped when `unstructured` is not installed — which is
the case in CI where we install only lightweight test dependencies.
Install the full `requirements.txt` to run these locally.
"""
import pytest

chunker = pytest.importorskip("scripts.chunker", reason="unstructured not installed")

PAGE_FOOTER_RE = chunker.PAGE_FOOTER_RE
REFERENCE_TITLE_RE = chunker.REFERENCE_TITLE_RE
_is_pure_footer_chunk = chunker._is_pure_footer_chunk


# ---------------------------------------------------------------------------
# PAGE_FOOTER_RE — should match footer noise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Page 12",
    "page 3",
    "PAGE 100",
    "19 of 35",
    "42",
    "7",
    "Sage Publications",
    "Wiley Publications",
    "Routledge Publications",
    "Essentials of Marketing Research",
    "The Essentials of Marketing Research",
    "The Absolute Essentials of Marketing Research",
    "Essentials of Marketing Research: Putting Research into Practice",
    "Research: Putting Research into Practice",
    "Putting Research into Practice",
])
def test_page_footer_re_matches_noise(text):
    assert PAGE_FOOTER_RE.match(text.strip()), f"Expected match: {text!r}"


@pytest.mark.parametrize("text", [
    "Market segmentation groups customers by shared characteristics.",
    "Cronbach's alpha measures internal consistency of a scale.",
    "Factor analysis reduces many variables to a smaller set of factors.",
    "The researcher administered a 7-point Likert scale.",
    "Page references should be verified before submission.",
])
def test_page_footer_re_no_false_positives(text):
    assert not PAGE_FOOTER_RE.match(text.strip()), f"Unexpected match: {text!r}"


# ---------------------------------------------------------------------------
# REFERENCE_TITLE_RE — should match section title variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "References",
    "REFERENCES",
    "References:",
    "Bibliography",
    "BIBLIOGRAPHY",
    "Works Cited",
    "works cited",
    "Literature Cited",
    "  References  ",
])
def test_reference_title_re_matches(text):
    assert REFERENCE_TITLE_RE.match(text.strip()), f"Expected match: {text!r}"


@pytest.mark.parametrize("text", [
    "Factor Analysis",
    "Survey References Overview",
    "See references on page 45",
    "Chapter 3: Measurement",
    "Introduction",
    "References and Further Reading",
])
def test_reference_title_re_no_false_positives(text):
    assert not REFERENCE_TITLE_RE.match(text.strip()), f"Unexpected match: {text!r}"


# ---------------------------------------------------------------------------
# _is_pure_footer_chunk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Page 12",
    "Putting Research into Practice",
    "42",
    "Essentials of Marketing Research\nPutting Research into Practice",
])
def test_is_pure_footer_chunk_true(text):
    assert _is_pure_footer_chunk(text) is True, f"Expected True for: {text!r}"


@pytest.mark.parametrize("text", [
    "Cronbach's alpha measures the internal consistency of a scale across items.",
    "Market segmentation divides a market into distinct groups with shared needs.",
    "Conjoint analysis determines how consumers value different product attributes.",
])
def test_is_pure_footer_chunk_false(text):
    assert _is_pure_footer_chunk(text) is False, f"Expected False for: {text!r}"
