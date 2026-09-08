"""Diagnose why session-01 chunks are not returned for the query
'What are the topics taught in session 1?'.

Runs three probes:

(a) The exact hybrid RRF pipeline used by ``app.py`` (dense + sparse
    prefetch, RRF fusion, dedup, per-source cap) and prints every kept
    result with source_file, chunk_index, and score.

(b) A raw Qdrant scroll for all points whose ``source_file`` contains
    ``session-01`` — so we know what session-01 chunks even exist in the
    collection, and can spot-check whether any of them plausibly match
    the query.

(c) For the session-01 chunks from (b), computes cosine similarity
    against the query's dense vector *manually* (client-side dot product
    on the normalised vectors already stored in Qdrant). This tells us
    whether session-01 is ranking low because it's genuinely dissimilar,
    or whether it's being filtered out downstream by the diversity cap
    or the dedup logic.
"""

from __future__ import annotations

import io
import os
import pathlib
import sys
from typing import Any

# Force UTF-8 stdout so ligatures, smart quotes, and other non-cp1252
# characters embedded in course materials (e.g. "fi" ligature at U+FB01
# in the R textbook) can be printed on Windows without a UnicodeEncodeError.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Allow ``python scripts/debug_session1.py`` from the repo root by putting
# the parent dir on sys.path so ``scripts.sparse_encoder`` resolves.
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
from langchain_openai import OpenAIEmbeddings  # noqa: E402
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.models import (  # noqa: E402
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    Prefetch,
    SparseVector,
)

from scripts.sparse_encoder import encode_query as encode_sparse_query  # noqa: E402

load_dotenv(override=False)

QUERY = "What are the topics taught in session 1?"
EMBEDDING_MODEL_ID = "text-embedding-3-small"
QDRANT_COLLECTION = "course-docs"
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
RETRIEVER_FETCH_K = 12
RETRIEVER_K = 6
MAX_HITS_PER_SOURCE = 2

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPEN_AI_EMBEDDINGS_API_KEY")

if not (QDRANT_URL and QDRANT_API_KEY and OPENAI_API_KEY):
    raise SystemExit(
        "Missing one of QDRANT_URL / QDRANT_API_KEY / "
        "OPEN_AI_EMBEDDINGS_API_KEY. Populate .env first."
    )


def _preview(text: str, n: int = 200) -> str:
    """Return a single-line preview of ``text``, whitespace-collapsed."""
    if not text:
        return ""
    flat = " ".join(text.split())
    return flat[:n] + ("..." if len(flat) > n else "")


def _cosine(a: list[float], b: list[float]) -> float:
    """Client-side cosine similarity. Qdrant stores normalised vectors
    for the Cosine metric, but we do not rely on that here — we compute
    the full cosine so this remains correct regardless of internal
    storage format."""
    if not a or not b or len(a) != len(b):
        return float("nan")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def main() -> None:
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL_ID, api_key=OPENAI_API_KEY)

    print("=" * 78)
    print(f"QUERY: {QUERY!r}")
    print("=" * 78)

    # ---- (a) Reproduce the app's hybrid pipeline ---------------------------
    query_vector = embedder.embed_query(QUERY)
    sp_indices, sp_values = encode_sparse_query(QUERY)

    prefetch: list[Prefetch] = [
        Prefetch(
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            limit=RETRIEVER_FETCH_K,
        ),
    ]
    if sp_indices:
        prefetch.append(
            Prefetch(
                query=SparseVector(indices=sp_indices, values=sp_values),
                using=SPARSE_VECTOR_NAME,
                limit=RETRIEVER_FETCH_K,
            )
        )
    else:
        print("[WARN] sparse encoder returned no tokens; dense-only branch")

    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=RETRIEVER_FETCH_K,
        with_payload=True,
    )
    fused = response.points

    print(f"\n(a) Hybrid RRF top-{RETRIEVER_FETCH_K} BEFORE dedup/cap "
          f"(as returned by Qdrant):")
    print("-" * 78)
    for rank, p in enumerate(fused, start=1):
        payload: dict[str, Any] = p.payload or {}
        src = payload.get("source_file", "?")
        cidx = payload.get("chunk_index")
        ch = payload.get("context_header", "")
        text = payload.get("text", "") or ""
        print(f"  #{rank:>2}  rrf={p.score:.5f}  {src}  chunk_idx={cidx}")
        if ch:
            print(f"        header: {_preview(ch, 120)}")
        print(f"        text:   {_preview(text, 180)}")

    # Now apply the app.py dedup + cap logic so we can see what actually
    # reaches the LLM.
    seen_chunks: set[tuple[str, Any]] = set()
    per_source: dict[str, int] = {}
    kept: list[Any] = []
    for p in fused:
        payload = p.payload or {}
        src = payload.get("source_file", "?")
        cidx = payload.get("chunk_index")
        key = (src, cidx)
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        if per_source.get(src, 0) >= MAX_HITS_PER_SOURCE:
            continue
        per_source[src] = per_source.get(src, 0) + 1
        kept.append(p)
        if len(kept) >= RETRIEVER_K:
            break

    print(f"\n(a') AFTER dedup + MAX_HITS_PER_SOURCE={MAX_HITS_PER_SOURCE} "
          f"(these are the {RETRIEVER_K} chunks the LLM actually sees):")
    print("-" * 78)
    for rank, p in enumerate(kept, start=1):
        payload = p.payload or {}
        src = payload.get("source_file", "?")
        cidx = payload.get("chunk_index")
        ch = payload.get("context_header", "")
        text = payload.get("text", "") or ""
        print(f"  #{rank}  rrf={p.score:.5f}  {src}  chunk_idx={cidx}")
        if ch:
            print(f"      header: {_preview(ch, 120)}")
        print(f"      text:   {_preview(text, 180)}")

    session01_in_kept = any(
        "session-01" in (p.payload or {}).get("source_file", "") for p in kept
    )
    print(f"\n  session-01 present in kept results? -> {session01_in_kept}")

    # ---- (b) Scroll ALL session-01 points ---------------------------------
    print("\n" + "=" * 78)
    print("(b) All points where source_file contains 'session-01':")
    print("=" * 78)

    # source_file lacks a text index (no MatchText), but it accepts exact
    # MatchAny lookups. Enumerate the four filenames session-01 can appear
    # under (Quarto .qmd and rendered .html; plus the "extended" variant).
    session01_filenames = [
        "session-01.qmd",
        "session-01.html",
        "session-01-extended.qmd",
        "session-01-extended.html",
    ]
    session01_points: list[Any] = []
    next_offset = None
    while True:
        page, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchAny(any=session01_filenames),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=[DENSE_VECTOR_NAME],
            limit=256,
            offset=next_offset,
        )
        session01_points.extend(page)
        if next_offset is None:
            break

    print(f"Found {len(session01_points)} session-01 point(s).\n")
    for p in session01_points:
        payload = p.payload or {}
        src = payload.get("source_file", "?")
        cidx = payload.get("chunk_index")
        ch = payload.get("context_header", "")
        text = payload.get("text", "") or ""
        print(f"  {src}  chunk_idx={cidx}")
        if ch:
            print(f"      header: {_preview(ch, 140)}")
        print(f"      text:   {_preview(text, 180)}")

    # ---- (c) Manual cosine similarity against session-01 chunks ------------
    print("\n" + "=" * 78)
    print("(c) Manual dense cosine similarity vs the query embedding, "
          "top 10 session-01 chunks:")
    print("=" * 78)

    scored: list[tuple[float, Any]] = []
    for p in session01_points:
        # p.vector is a dict when named vectors were requested.
        vec = None
        if isinstance(p.vector, dict):
            vec = p.vector.get(DENSE_VECTOR_NAME)
        if vec is None:
            continue
        scored.append((_cosine(query_vector, vec), p))

    scored.sort(key=lambda t: t[0], reverse=True)
    for rank, (score, p) in enumerate(scored[:10], start=1):
        payload = p.payload or {}
        src = payload.get("source_file", "?")
        cidx = payload.get("chunk_index")
        ch = payload.get("context_header", "")
        text = payload.get("text", "") or ""
        print(f"  #{rank}  cos={score:.4f}  {src}  chunk_idx={cidx}")
        if ch:
            print(f"      header: {_preview(ch, 140)}")
        print(f"      text:   {_preview(text, 180)}")

    # For comparison, print the top dense-only global hits (no fusion,
    # no sparse) so we can see what the pure dense retriever chose and at
    # what score session-01 would need to beat to enter top-12.
    print("\n(c') For comparison: TOP DENSE-ONLY GLOBAL hits (no fusion):")
    print("-" * 78)
    dense_only = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        using=DENSE_VECTOR_NAME,
        limit=RETRIEVER_FETCH_K,
        with_payload=True,
    ).points
    for rank, p in enumerate(dense_only, start=1):
        payload = p.payload or {}
        src = payload.get("source_file", "?")
        cidx = payload.get("chunk_index")
        print(f"  #{rank:>2}  cos={p.score:.4f}  {src}  chunk_idx={cidx}")


if __name__ == "__main__":
    main()
