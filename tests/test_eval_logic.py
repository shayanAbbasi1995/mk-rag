"""Tests for pure-logic functions in scripts/eval.py.

No API calls are made. Client-construction functions are never invoked —
only module-level constants and stateless helpers are tested here.
Requires: qdrant-client, openai, python-dotenv (to import the module).
"""
import pytest

from scripts.eval import (
    ADMIN_QUERY_RE,
    TERM_PINNING,
    _reciprocal_rank_fusion,
    build_context,
    detect_doc_type_filter,
    is_refusal,
    source_coverage,
)


# ---------------------------------------------------------------------------
# Minimal stand-in for a Qdrant ScoredPoint
# ---------------------------------------------------------------------------


class FakePoint:
    def __init__(self, source_file: str, chunk_index: int | None = None, score: float = 1.0):
        self.payload = {"source_file": source_file, "chunk_index": chunk_index}
        self.score = score


# ---------------------------------------------------------------------------
# ADMIN_QUERY_RE — true positives
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "When is the midterm?",
    "What day is the final exam?",
    "office hours on Thursday?",
    "when is assignment 2 due",
    "what is the grading scheme",
    "syllabus for this course",
    "what is the schedule",
    "when are assignments due",
    "What is the due date for the report?",
    "when does class start",
    "what time does the exam begin",
])
def test_admin_re_matches_admin_queries(query):
    assert ADMIN_QUERY_RE.search(query), f"Expected match for: {query!r}"


# ---------------------------------------------------------------------------
# ADMIN_QUERY_RE — false positive guard (content questions must not match)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query", [
    "What is Cronbach's alpha and when do I use it?",
    "when do I apply factor analysis?",
    "How do I interpret cluster analysis results?",
    "What is a Likert scale?",
    "explain conjoint analysis",
    "What are part-worth utilities?",
])
def test_admin_re_does_not_match_content_queries(query):
    assert not ADMIN_QUERY_RE.search(query), f"Unexpected match for: {query!r}"


# ---------------------------------------------------------------------------
# detect_doc_type_filter
# ---------------------------------------------------------------------------


def test_detect_doc_type_filter_admin_returns_syllabus():
    assert detect_doc_type_filter("When is the midterm?") == "syllabus_outline"


def test_detect_doc_type_filter_content_returns_none():
    assert detect_doc_type_filter("What is Cronbach's alpha?") is None


def test_detect_doc_type_filter_cronbachs_with_when_do():
    # Regression: "when do I use it?" must NOT route to syllabus filter.
    result = detect_doc_type_filter("What is Cronbach's alpha and when do I use it?")
    assert result is None


# ---------------------------------------------------------------------------
# TERM_PINNING
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("query,idx", [
    ("How does cluster analysis work?", 0),
    ("explain clustering techniques", 0),
    ("difference between clustering and factor analysis", 0),
    ("what are part-worth utilities?", 1),
    ("how is part worth calculated in conjoint", 1),
    ("explain part-worth scores", 1),
])
def test_term_pinning_pattern_matches(query, idx):
    pattern, _, _ = TERM_PINNING[idx]
    assert pattern.search(query), f"TERM_PINNING[{idx}] should match: {query!r}"


@pytest.mark.parametrize("query", [
    "What is factor analysis?",
    "explain conjoint analysis",
    "how do I interpret regression coefficients",
])
def test_cluster_pinning_no_false_positives(query):
    cluster_pattern, _, _ = TERM_PINNING[0]
    assert not cluster_pattern.search(query), f"Unexpected cluster match: {query!r}"


@pytest.mark.parametrize("query", [
    "What is factor analysis?",
    "explain survey design",
    "how is internal consistency measured",
])
def test_partworth_pinning_no_false_positives(query):
    partworth_pattern, _, _ = TERM_PINNING[1]
    assert not partworth_pattern.search(query), f"Unexpected part-worth match: {query!r}"


def test_term_pinning_has_probe_and_pin_n():
    for pattern, probe, pin_n in TERM_PINNING:
        assert isinstance(probe, str) and len(probe) > 0
        assert isinstance(pin_n, int) and pin_n >= 1


# ---------------------------------------------------------------------------
# _reciprocal_rank_fusion
# ---------------------------------------------------------------------------


def test_rrf_empty_input():
    assert _reciprocal_rank_fusion([]) == []


def test_rrf_single_point_score():
    p = FakePoint("file.pdf", 0)
    result = _reciprocal_rank_fusion([[p]])
    assert len(result) == 1
    point, score = result[0]
    assert point is p
    # rank=0, k=60 → 1/(60+0+1) = 1/61
    assert abs(score - 1.0 / 61) < 1e-12


def test_rrf_scores_decrease_with_rank():
    points = [FakePoint(f"doc{i}.pdf", i) for i in range(5)]
    result = _reciprocal_rank_fusion([points])
    scores = [s for _, s in result]
    assert scores == sorted(scores, reverse=True)


def test_rrf_point_in_both_lists_scores_higher():
    p_both = FakePoint("both.pdf", 0)
    p_once = FakePoint("once.pdf", 1)
    result = _reciprocal_rank_fusion([[p_both, p_once], [p_both]])
    top_point, _ = result[0]
    assert top_point is p_both


def test_rrf_accumulates_score_across_lists():
    # Same point at rank 0 in two lists → 2 × 1/61
    p = FakePoint("file.pdf", 0)
    result = _reciprocal_rank_fusion([[p], [p]])
    _, score = result[0]
    assert abs(score - 2.0 / 61) < 1e-12


def test_rrf_deduplicates_same_chunk_identity():
    # Two lists containing the same (source_file, chunk_index) → one result entry.
    p = FakePoint("file.pdf", 0)
    result = _reciprocal_rank_fusion([[p], [p]])
    assert len(result) == 1


def test_rrf_custom_k():
    p = FakePoint("x.pdf", 0)
    result = _reciprocal_rank_fusion([[p]], k=0)
    _, score = result[0]
    assert abs(score - 1.0) < 1e-12  # 1/(0+0+1) = 1.0


def test_rrf_two_separate_points_both_present():
    p1 = FakePoint("a.pdf", 0)
    p2 = FakePoint("b.pdf", 1)
    result = _reciprocal_rank_fusion([[p1, p2]])
    assert len(result) == 2


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


def _doc(text, source="file.pdf", page=None, ctx_hdr=""):
    return {"source_file": source, "page_number": page, "text": text, "context_header": ctx_hdr}


def test_build_context_single_no_page():
    result = build_context([_doc("Survey design basics.")])
    assert "[1] Source: file.pdf" in result
    assert "Survey design basics." in result


def test_build_context_page_number_included():
    result = build_context([_doc("Factor analysis.", page=42)])
    assert "(p. 42)" in result


def test_build_context_no_page_no_parens():
    result = build_context([_doc("Body.", page=None)])
    assert "(p." not in result


def test_build_context_context_header_prepended():
    result = build_context([_doc("Body text.", ctx_hdr="Chapter on reliability.")])
    # header should appear before body in the string
    assert result.index("Chapter on reliability") < result.index("Body text")


def test_build_context_no_context_header_no_leading_blank():
    result = build_context([_doc("Body.", ctx_hdr="")])
    # The line after "Source: ..." should be the body, not a blank line.
    assert "[1] Source: file.pdf\nBody." in result


def test_build_context_multiple_docs_numbered_sequentially():
    docs = [_doc("First.", source="a.pdf"), _doc("Second.", source="b.pdf")]
    result = build_context(docs)
    assert "[1] Source: a.pdf" in result
    assert "[2] Source: b.pdf" in result


def test_build_context_multiple_docs_separated_by_double_newline():
    docs = [_doc("First."), _doc("Second.", source="b.pdf")]
    result = build_context(docs)
    assert "\n\n" in result


def test_build_context_empty_list():
    assert build_context([]) == ""


# ---------------------------------------------------------------------------
# source_coverage + is_refusal
# ---------------------------------------------------------------------------


def test_source_coverage_hit_on_source_file():
    docs = [{"source_file": "lecture_conjoint.pdf", "source_path": ""}]
    assert source_coverage(docs, "conjoint") is True


def test_source_coverage_miss():
    docs = [{"source_file": "lecture_conjoint.pdf", "source_path": ""}]
    assert source_coverage(docs, "reliability") is False


def test_source_coverage_case_insensitive():
    docs = [{"source_file": "Lecture_Conjoint.PDF", "source_path": ""}]
    assert source_coverage(docs, "conjoint") is True


def test_source_coverage_off_syllabus_always_true():
    assert source_coverage([], "OFF_SYLLABUS") is True


def test_source_coverage_hit_on_source_path():
    docs = [{"source_file": "unknown.pdf", "source_path": "/data/syllabus_outline.pdf"}]
    assert source_coverage(docs, "syllabus") is True


def test_is_refusal_canonical_phrase():
    answer = "Based on the provided course materials, I cannot find information about this."
    assert is_refusal(answer) is True


def test_is_refusal_variant_phrase():
    assert is_refusal("Could not find anything relevant to your question.") is True


def test_is_refusal_normal_answer():
    assert is_refusal("Cronbach's alpha measures internal consistency of a scale.") is False


def test_is_refusal_empty_string():
    assert is_refusal("") is False
