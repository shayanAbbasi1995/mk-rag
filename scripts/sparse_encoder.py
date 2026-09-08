"""Lightweight in-process BM25-style sparse encoder for Qdrant hybrid search.

Qdrant natively stores sparse vectors as (indices, values) pairs. We hash each
token to an integer bucket and emit weights, letting Qdrant's server-side
sparse index handle inverted-list lookups. This keeps the whole hybrid
pipeline dependency-free: no `rank_bm25`, no `pyserini`, no tokenizer models.

Design notes
------------
* Vocabulary size is fixed at ``VOCAB_SIZE`` (2**17 = 131_072). Hash collisions
  at this scale are acceptable for retrieval — BM25's IDF weighting is
  effectively unavailable for collided tokens, but recall still benefits from
  exact-token overlap because Qdrant's sparse similarity is a dot product of
  matching indices.
* At **index time** we emit raw term-frequency weights per token: each unique
  token in the chunk contributes its TF (count) as the value at its hashed
  index. We deliberately skip IDF at index time because computing corpus IDF
  requires a full pass over the corpus first, and we prefer a single-pass
  ingest.
* At **query time** we emit ``1.0`` per unique query token. This mirrors the
  common short-query BM25 approximation (the ``qtf`` term stays 1 for
  ~single-occurrence query terms) and keeps the retrieval side stateless.
* Stopword list is deliberately short — just the highest-frequency English
  function words. Marketing Research vocabulary (e.g. ``sample``, ``brand``,
  ``data``, ``research``) must survive filtering.
"""

from __future__ import annotations

import re
from collections import Counter

# Vocabulary bucket size. 2**17 = 131_072 gives ~0.01% collision rate on
# typical academic-English vocabularies and stays well under Qdrant's
# sparse-vector size limits.
VOCAB_SIZE: int = 131_072

# Cheap English stopwords. Kept short: heavier lists strip domain-relevant
# tokens (e.g. "same" occurs in "same-store sales", "which" in survey
# design examples). Extend only if a specific query surfaces noise.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then else of in on at to for from by with
    is are was were be been being am
    this that these those it its they them their there here
    as so than which who whom whose what when where why how
    not no nor do does did doing done
    have has had having
    i you he she we me us my your his her our
    """.split()
)

# Tokenizer: lowercase words of 2+ alphanumerics. Splits on whitespace and
# punctuation. Numbers survive (Marketing Research talks about "95% CI",
# "n=200", "Likert 1-7") because the leading digit tokens are meaningful.
_TOKEN_RE = re.compile(r"[a-z0-9]{2,}")


def _tokenize(text: str) -> list[str]:
    """Lowercase, extract 2+ char alphanumeric tokens, drop stopwords."""
    if not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS]


def _hash_token(token: str) -> int:
    """Stable, portable hash -> vocabulary index.

    We intentionally do NOT use Python's built-in ``hash()`` because it is
    salted per-process and would produce different bucket assignments on
    every run — breaking retrieval since index-time and query-time
    encoders would disagree.
    """
    # FNV-1a 64-bit hash. Deterministic across processes and Python
    # versions, no external dependencies.
    h = 0xCBF29CE484222325
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % VOCAB_SIZE


def encode_document(text: str) -> tuple[list[int], list[float]]:
    """Return (indices, values) for a document to be stored in Qdrant.

    Values are raw term frequencies (integer counts cast to float). This is
    the ``tf`` component of BM25 without the length-normalisation term (we
    do not have corpus-average document length at index time in a single
    pass), which is fine because Qdrant's dot-product sparse similarity
    still surfaces documents whose vocabulary overlaps the query most.
    """
    tokens = _tokenize(text)
    if not tokens:
        return [], []
    counts = Counter(tokens)
    # Aggregate by hashed index so collisions from distinct tokens still
    # produce a valid (indices, values) pair (Qdrant requires unique
    # indices). We sum the collided TFs — a mild noise source but harmless.
    bucket: dict[int, float] = {}
    for tok, tf in counts.items():
        idx = _hash_token(tok)
        bucket[idx] = bucket.get(idx, 0.0) + float(tf)
    indices = sorted(bucket.keys())
    values = [bucket[i] for i in indices]
    return indices, values


def encode_query(text: str) -> tuple[list[int], list[float]]:
    """Return (indices, values) for a query.

    Weight per unique query token is 1.0. Rationale: BM25's query-term
    frequency ``qtf`` is almost always 1 for short natural-language
    questions, and dropping the IDF weighting (which we don't have) leaves
    ``1.0`` as the natural default. Documents matching *more* query terms
    will still rank higher because the sparse dot product accumulates
    per-term contributions.
    """
    tokens = _tokenize(text)
    if not tokens:
        return [], []
    # Dedup while preserving hash-bucket uniqueness (Qdrant requires it).
    buckets: dict[int, float] = {}
    for tok in tokens:
        idx = _hash_token(tok)
        buckets.setdefault(idx, 1.0)
    indices = sorted(buckets.keys())
    values = [buckets[i] for i in indices]
    return indices, values
