"""
Marketing-Research course RAG: end-to-end ingestion orchestrator.

Pipeline (mirrors `Preprocessing Unstructured Data for LLM Applications.pdf`):

    docs/**/*  --> partitioner.partition_with_cache()   [Stage 1]
               --> chunker.clean_elements()             [Stage 2]
               --> chunker.chunk_elements()             [Stage 3, chunk_by_title]
               --> OpenAI text-embedding-3-small        [Stage 4a]
               --> Qdrant Cloud collection              [Stage 4b]

Runs incrementally via `index_tracker.json`: only files whose MD5 hash has
changed since the last run are re-parsed / re-embedded / re-upserted.

Usage:
    python -m scripts.ingest                 # ingest everything new/changed
    python -m scripts.ingest --limit 3       # smoke-test on 3 files
    python -m scripts.ingest --files a.pdf   # force-ingest one or more files
    python -m scripts.ingest --dry-run       # partition + chunk only; no API calls
    python -m scripts.ingest --reset         # drop the Qdrant collection first
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)
from tqdm import tqdm

from scripts.chunker import chunk_elements, clean_elements
from scripts.partitioner import (
    file_md5,
    iter_source_files,
    partition_with_cache,
)
from scripts.sparse_encoder import encode_document as encode_sparse_doc

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
CACHE_DIR = REPO_ROOT / "data" / "elements_cache"
TRACKER_FILE = REPO_ROOT / "index_tracker.json"

COLLECTION_NAME = "course-docs"        # kept in sync with app.py
EMBEDDING_DIM = 1536                   # text-embedding-3-small
EMBEDDING_BATCH_SIZE = 100             # OpenAI accepts up to 2048 inputs/req;
                                       # 100 keeps request bodies small and
                                       # backoff granularity useful.
UPSERT_BATCH_SIZE = 100

# Named-vector labels for the hybrid config. Storing dense and sparse
# vectors under distinct names lets Qdrant apply the correct index and
# similarity function to each; the retriever picks them by name in
# Prefetch(using=...). Keep these strings in sync with app.py.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

# Max Title elements to walk up when building a chunk's context header.
# Three levels covers "Chapter > Section > Subsection" style hierarchies
# without producing headers longer than the chunk body they annotate.
MAX_CONTEXT_HEADER_LEVELS = 3

# The `.env` variable OPEN_AI_EMBEDDINGS_NAME is a project label, not a
# model ID. If the user ever sets it to a real embeddings model id
# (starts with 'text-embedding-') we honour it; otherwise we default to
# text-embedding-3-small which is what the existing app.py expects.
_env_model = os.getenv("OPEN_AI_EMBEDDINGS_NAME", "")
EMBEDDING_MODEL = (
    _env_model if _env_model.startswith("text-embedding-") else "text-embedding-3-small"
)

# LLM-generated contextual headers (P2.2). We call DeepSeek-V3 via
# OpenRouter for a single-sentence topic description prepended to each
# chunk before embedding + indexing. This buys us contextual retrieval
# (Anthropic's "Contextual Retrieval" recipe): the embedding of a chunk
# that says "This section describes stratified sampling design decisions"
# ranks much better on queries like "how do I set up stratified sampling"
# than the raw chunk body alone.
#
# OPENROUTER_API_NAME in .env is a project label (mr-rag-open), NOT a
# model ID — match how app.py treats it. Pass the real ID here.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "deepseek/deepseek-chat"  # DeepSeek-V3

# Max character length of the doc-summary snippet passed to the context
# LLM. Bigger context helps the LLM disambiguate the chunk, but every
# extra character multiplies across ~8,875 calls. 800 chars is enough for
# 1-2 paragraphs of a document opening, which is what the LLM needs to
# know "this is Chapter 4 of a marketing-research textbook about X".
DOC_SUMMARY_MAX_CHARS = 800
# Max chunk-text characters shown to the context LLM. 500 chars is enough
# for the LLM to identify the topic without paying to embed the whole
# chunk into the prompt as well.
CHUNK_SNIPPET_MAX_CHARS = 500

# Concurrency for the LLM-context per-chunk calls within one file.
# DeepSeek-V3 via OpenRouter handles ~10 concurrent requests easily, and
# most chunks return in 0.5-2s. Serial calls make full-corpus re-ingest
# unbearably slow (~2 hours); with 10-way concurrency we're down to ~15
# minutes. Order is preserved because we submit-then-map over futures
# with the original chunk order.
LLM_CONTEXT_CONCURRENCY = 10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


# -----------------------------------------------------------------------------
# Tracker
# -----------------------------------------------------------------------------


def load_tracker() -> dict[str, Any]:
    if TRACKER_FILE.exists():
        return json.loads(TRACKER_FILE.read_text(encoding="utf-8"))
    return {"last_updated": None, "files": {}, "failures": {}}


def save_tracker(tracker: dict[str, Any]) -> None:
    tracker["last_updated"] = datetime.now(timezone.utc).isoformat()
    TRACKER_FILE.write_text(json.dumps(tracker, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Client factories (isolated so tests / repl sessions can call in with fakes)
# -----------------------------------------------------------------------------


def build_openai_client() -> OpenAI:
    api_key = os.getenv("OPEN_AI_EMBEDDINGS_API_KEY")
    if not api_key:
        raise RuntimeError("OPEN_AI_EMBEDDINGS_API_KEY missing from .env")
    return OpenAI(api_key=api_key)


def build_openrouter_client() -> OpenAI:
    """OpenAI-SDK-compatible client pointed at OpenRouter.

    Used only for the P2.2 contextual-header LLM calls. We isolate this
    from `build_openai_client` because the two use different API keys and
    different base URLs, and mixing them up is a silent auth error.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing from .env")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def build_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not (url and api_key):
        raise RuntimeError("QDRANT_URL and/or QDRANT_API_KEY missing from .env")
    # Timeout raised from the 5s default: Qdrant Cloud free tier can be
    # slow on cold start, and a single upsert of ~100 vectors can push
    # past 5s under network jitter.
    return QdrantClient(url=url, api_key=api_key, timeout=60)


def ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    """Create the hybrid (dense + sparse) collection if it does not exist.

    The collection uses **named vectors**: ``dense`` holds the OpenAI
    embedding for cosine similarity, ``sparse`` holds the BM25-style
    lexical vector for exact-token overlap. Both are queried per-request
    via `Prefetch(using=...)` and their result lists fused with RRF at
    the retriever.
    """
    existing = {c.name for c in client.get_collections().collections}
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=dim, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                # on_disk=False keeps the sparse inverted index in RAM.
                # Our corpus is small (< 10k points), so the memory cost
                # is negligible and query latency is noticeably better.
                SPARSE_VECTOR_NAME: SparseVectorParams(
                    index=SparseIndexParams(on_disk=False),
                ),
            },
        )
        logger.info(
            "Created Qdrant collection %r (dense dim=%d, sparse=%r)",
            name, dim, SPARSE_VECTOR_NAME,
        )
    else:
        logger.info("Qdrant collection %r already exists", name)
    ensure_payload_indexes(client, name)


# Payload fields we need indexed on the server side. Two reasons:
#   1. delete_stale_points() filters by (source_file, source_path) — Qdrant
#      rejects filter queries against unindexed keyword fields with a 400.
#      Without the index, our anti-duplicate delete step would silently
#      fail and we'd be right back to accumulating duplicate points.
#   2. Future UI features (source-type filter in the sidebar, page filter)
#      need indexes on element_category and page_number to be usable.
_INDEX_FIELDS: dict[str, PayloadSchemaType] = {
    "source_file": PayloadSchemaType.KEYWORD,
    "source_path": PayloadSchemaType.KEYWORD,
    "element_category": PayloadSchemaType.KEYWORD,
    "file_hash": PayloadSchemaType.KEYWORD,
    "page_number": PayloadSchemaType.INTEGER,
    # doc_type filter (app.py::_detect_doc_type_filter -> "syllabus_outline"
    # for admin queries). Must be indexed for MatchValue filters to work.
    "doc_type": PayloadSchemaType.KEYWORD,
}


def ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    """Create keyword/integer payload indexes if missing. Idempotent:
    Qdrant returns a benign 'already exists' error on re-create which
    we swallow so this is safe to call on every ingest run."""
    try:
        existing = set(client.get_collection(collection).payload_schema or {})
    except Exception:  # noqa: BLE001
        existing = set()
    for field, schema in _INDEX_FIELDS.items():
        if field in existing:
            continue
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=schema,
                wait=True,
            )
            logger.info("Created payload index on %r (%s)", field, schema)
        except Exception as e:  # noqa: BLE001
            logger.debug("Payload index %r not created: %s", field, e)


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------


def embed_texts(
    openai_client: OpenAI,
    texts: list[str],
    model: str = EMBEDDING_MODEL,
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Batch-embed `texts`, preserving order. Retries transient failures
    with exponential backoff (3 attempts) so a single blip mid-ingest
    doesn't waste the whole run's work."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        for attempt in range(3):
            try:
                resp = openai_client.embeddings.create(model=model, input=batch)
                vectors.extend(d.embedding for d in resp.data)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "Embedding batch failed (attempt %d/3): %s. Retrying in %ds.",
                    attempt + 1,
                    e,
                    wait,
                )
                time.sleep(wait)
    return vectors


# -----------------------------------------------------------------------------
# LLM-generated contextual headers (P2.2)
# -----------------------------------------------------------------------------


def build_doc_summary(elements: list[Any]) -> str:
    """Return an ~800-char summary snippet built from the first Title and
    NarrativeText elements of a document.

    Used as background context in the ``generate_llm_context`` prompt so
    the LLM can write chunk-topic sentences that reflect the *document's*
    subject rather than just what's visible in the chunk.

    Concatenation order preserves document order (elements are already
    document-ordered by the partitioner). We stop as soon as we've
    collected DOC_SUMMARY_MAX_CHARS characters — no need to walk the
    whole doc.
    """
    parts: list[str] = []
    used = 0
    for el in elements:
        category = getattr(el, "category", None) or el.to_dict().get("type", "")
        if category not in ("Title", "NarrativeText"):
            continue
        text = (getattr(el, "text", "") or "").strip()
        if not text:
            continue
        parts.append(text)
        used += len(text) + 1  # +1 for the joining space
        if used >= DOC_SUMMARY_MAX_CHARS:
            break
    summary = " ".join(parts)
    return summary[:DOC_SUMMARY_MAX_CHARS]


def generate_llm_context(
    chunk_text: str,
    doc_summary: str,
    openrouter_client: OpenAI | None,
    model: str = OPENROUTER_MODEL,
) -> str:
    """Return a one-sentence topic descriptor for the chunk, or "" on failure.

    Mirrors Anthropic's contextual-retrieval prompt: the LLM sees the doc
    summary and the chunk, then writes a self-contained sentence that
    describes what concept or topic the chunk covers. Prepending this
    sentence to the chunk before embedding materially improves recall on
    queries whose vocabulary differs from the chunk body.

    Backoff mirrors ``embed_texts`` (3 attempts, 1s/2s/4s) so a transient
    OpenRouter blip doesn't wipe a run's worth of context calls.

    Returns "" on any exception — callers should fall back to the
    rule-based breadcrumb header in that case.
    """
    if openrouter_client is None or not chunk_text.strip():
        return ""

    system = (
        "You are a document indexer. Given a document summary and a text "
        "chunk, write a single sentence (max 20 words) that describes "
        "what concept or topic this chunk covers. Return ONLY the "
        "sentence, no quotes or punctuation at start/end."
    )
    user = (
        f"Document summary: {doc_summary}\n\n"
        f"Chunk: {chunk_text[:CHUNK_SNIPPET_MAX_CHARS]}"
    )

    for attempt in range(3):
        try:
            resp = openrouter_client.chat.completions.create(
                model=model,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            sentence = (resp.choices[0].message.content or "").strip()
            # Strip stray wrapping quotes / trailing punctuation the LLM
            # sometimes adds despite the instruction. We DO keep the
            # sentence's internal punctuation; only strip the outer.
            sentence = sentence.strip().strip('"').strip("'").strip()
            # Kill trailing full stops so the prepend doesn't produce
            # "topic.\nBody" which reads awkwardly next to real prose.
            sentence = sentence.rstrip(".!?")
            return sentence
        except Exception as e:  # noqa: BLE001
            if attempt == 2:
                logger.warning(
                    "LLM context call failed after 3 attempts: %s. "
                    "Falling back to rule-based breadcrumb.", e,
                )
                return ""
            wait = 2 ** attempt
            logger.debug(
                "LLM context call attempt %d/3 failed: %s. Retrying in %ds.",
                attempt + 1, e, wait,
            )
            time.sleep(wait)
    return ""


# -----------------------------------------------------------------------------
# Per-file pipeline
# -----------------------------------------------------------------------------


def _build_element_index(elements: list[Any]) -> dict[str, Any]:
    """Return a lookup table from ``element_id`` -> element for the raw
    (pre-chunk) elements list. Used by ``build_context_header`` when the
    partitioner populates parent_id chains (PDFs sometimes do, HTML often
    does not). For flat element streams the caller falls back to the
    Title-depth stack (see build_context_headers_for_chunks)."""
    index: dict[str, Any] = {}
    for el in elements:
        el_id = getattr(el, "id", None) or el.to_dict().get("element_id")
        if el_id:
            index[el_id] = el
    return index


def _title_depth(el: Any) -> int:
    """Return the Title element's `category_depth` (0 = outermost heading).

    Titles emitted by partition_md/html carry `metadata.category_depth`
    reflecting the source heading level. The rule-based PDF partitioner
    (`_partition_pdf_fast`) does not, so we default to 0, which keeps all
    PDF titles at the same level (fine — the fallback still gives us the
    single most recent enclosing heading in the header)."""
    depth = getattr(el.metadata, "category_depth", None)
    try:
        return int(depth) if depth is not None else 0
    except (TypeError, ValueError):
        return 0


def _extract_chunk_own_title(chunk: Any) -> tuple[str, int] | None:
    """Return (title_text, depth) for the Title element inside a chunk's
    `orig_elements`, or None if the chunk has no leading Title.

    `chunk_by_title` merges elements upward under their governing Title,
    so the chunk's own governing Title (if any) is the *first* Title in
    orig_elements. This is the innermost breadcrumb component."""
    md_dict = chunk.metadata.to_dict() if hasattr(chunk.metadata, "to_dict") else {}
    oe = md_dict.get("orig_elements")
    if not oe:
        return None
    try:
        # Lazy import: unstructured is expensive to load, and this helper
        # gets called once per chunk during ingest only.
        from unstructured.staging.base import elements_from_base64_gzipped_json
        origs = elements_from_base64_gzipped_json(oe)
    except Exception:  # noqa: BLE001 — malformed cache shouldn't kill ingest
        return None
    for el in origs:
        cat = getattr(el, "category", None) or el.to_dict().get("type", "")
        text = (el.text or "").strip()
        if cat == "Title" and text:
            return text, _title_depth(el)
    return None


def build_context_headers_for_chunks(
    chunks: list[Any],
    max_levels: int = MAX_CONTEXT_HEADER_LEVELS,
) -> list[str]:
    """Return one breadcrumb string per chunk, in the same order as
    ``chunks``, using a depth-indexed running stack of Title elements.

    Algorithm:
      * Walk chunks in order (chunks are already in document order because
        `chunk_by_title` preserves it).
      * For each chunk, extract its own Title (from ``orig_elements``).
        If present, use it as the innermost breadcrumb element and update
        the running stack: any stack entry at depth >= this Title's depth
        is popped (deeper headings are no longer in scope), then this
        Title is pushed at its own depth.
      * The chunk's breadcrumb is the top-down stack snapshot (outermost
        Title first), joined with ``' > '`` and wrapped in ``[...]``.
      * When a chunk has no leading Title, its breadcrumb is the current
        stack snapshot (i.e. it inherits the most recent enclosing
        heading). If the stack is empty, the header is ``""`` — the
        caller then skips prepending.

    Returns an empty header for chunks with no resolvable enclosing title.
    """
    # stack[depth] = title text at that heading level, currently in scope.
    # We use a dict keyed by depth (int) so PDFs (all depth=0) still get
    # a single-level stack while qmd/html (depth 0..N) get proper nesting.
    stack: dict[int, str] = {}

    def _snapshot(exclude_depth: int | None = None) -> list[str]:
        """Return the stack contents ordered by depth ascending
        (outermost first). If ``exclude_depth`` is set, drop that level —
        used when a chunk's own Title is already the last breadcrumb
        element and we don't want to duplicate the enclosing entry at
        the same depth."""
        keys = sorted(stack.keys())
        return [stack[k] for k in keys if exclude_depth is None or k != exclude_depth]

    headers: list[str] = []
    for chunk in chunks:
        own = _extract_chunk_own_title(chunk)
        if own is not None:
            title_text, depth = own
            # Pop any deeper-or-equal entries: a new heading at the same
            # depth closes the previous one; a heading at a shallower
            # depth is not reachable via ordered chunks in unstructured
            # (they surface in a separate chunk), so the equality case is
            # what actually fires here for level-1 slide titles.
            for k in [k for k in stack.keys() if k >= depth]:
                stack.pop(k)
            stack[depth] = title_text
            crumbs = _snapshot()
        else:
            # Inherit the current enclosing headings.
            crumbs = _snapshot()

        # Truncate to the deepest max_levels entries so extreme nestings
        # ("Chapter 4 > Section 4.2 > Subsection 4.2.1 > Point (c)") don't
        # produce headers longer than the chunk body.
        if len(crumbs) > max_levels:
            crumbs = crumbs[-max_levels:]

        if crumbs:
            # Length guard: an accidentally very long slide title would
            # bloat the embedding input. 200 chars is generous — real
            # section titles are ~40 chars.
            joined = " > ".join(c.strip()[:200] for c in crumbs)
            headers.append(f"[{joined}]")
        else:
            headers.append("")
    return headers


def build_context_header(
    chunk: Any,
    element_index: dict[str, Any],
    max_levels: int = MAX_CONTEXT_HEADER_LEVELS,
) -> str:
    """Single-chunk header builder that walks the ``parent_id`` chain.

    Kept for cases where the partitioner does populate parent_id (some
    HTML/DOCX flows). Returns ``""`` when no chain resolves — the ingest
    loop uses ``build_context_headers_for_chunks`` as the primary path
    because it works even when parent_id is missing.
    """
    parent_id = getattr(chunk.metadata, "parent_id", None)
    if not parent_id:
        return ""

    titles: list[str] = []
    visited: set[str] = set()
    hops = 0
    while parent_id and parent_id not in visited and hops < max_levels * 4:
        visited.add(parent_id)
        hops += 1
        el = element_index.get(parent_id)
        if el is None:
            break
        category = getattr(el, "category", None) or el.to_dict().get("type", "")
        text = (el.text or "").strip()
        if category == "Title" and text:
            titles.append(text)
            if len(titles) >= max_levels:
                break
        parent_id = getattr(el.metadata, "parent_id", None)

    if not titles:
        return ""
    breadcrumb = " > ".join(reversed(titles))
    return f"[{breadcrumb}]"


def chunks_to_points(
    chunks: list[Any],
    source_path: Path,
    file_hash: str,
    vectors: list[list[float]],
    sparse_vectors: list[tuple[list[int], list[float]]],
    context_headers: list[str],
    breadcrumbs: list[str],
    prefixed_texts: list[str],
    doc_type: str,
) -> list[PointStruct]:
    """Convert (chunk, dense vec, sparse vec, header, prefixed-text) tuples
    into Qdrant PointStructs.

    Payload carries all the metadata a downstream retriever might filter or
    cite on: source filename, docs-relative path, element category, page
    number, parent_id, per-file chunk index, and — new for the contextual
    ingest — the LLM-or-breadcrumb-prefixed ``text`` alongside a standalone
    ``context_header`` (LLM sentence when available, breadcrumb otherwise)
    and a separate ``breadcrumb`` field kept for debugging so we can
    inspect the old rule-based header even when the LLM one is winning.
    ``doc_type`` is written per-point so ``app.py::_detect_doc_type_filter``
    can route admin queries at retrieval time.

    Each point stores **two** named vectors:
      * ``dense``  -> OpenAI text-embedding-3-small (1536-dim, cosine)
      * ``sparse`` -> hashed BM25-TF sparse vector for lexical recall
    Both are computed on the *prefixed* text so that queries mentioning
    section titles (e.g. "mail surveys") hit both the semantic embedding
    and the exact-token overlap of the header.
    """
    points: list[PointStruct] = []
    try:
        rel = source_path.relative_to(DOCS_DIR).as_posix()
    except ValueError:
        rel = source_path.as_posix()

    for i, (chunk, vector, (sp_idx, sp_vals), header, breadcrumb, prefixed_text) in enumerate(
        zip(chunks, vectors, sparse_vectors, context_headers, breadcrumbs, prefixed_texts, strict=True)
    ):
        md = chunk.metadata
        payload = {
            # `source_file` is the key app.py already displays as the
            # source citation — keep the name stable.
            "source_file": source_path.name,
            "source_path": rel,
            "page_number": getattr(md, "page_number", None),
            "element_category": getattr(chunk, "category", None) or chunk.to_dict().get("type"),
            "parent_id": getattr(md, "parent_id", None),
            "element_id": getattr(chunk, "id", None) or chunk.to_dict().get("element_id"),
            "chunk_index": i,
            "file_hash": file_hash,
            # `text` stores the *prefixed* form so retrieval, display, and
            # LLM context all see the contextualised chunk. `context_header`
            # is stored separately (LLM sentence if we got one, else the
            # breadcrumb) so we can audit which chunks got a header
            # without regexing the prefix back out of `text`. The raw
            # rule-based breadcrumb is kept in `breadcrumb` for debugging.
            "text": prefixed_text,
            "context_header": header,
            "breadcrumb": breadcrumb,
            "doc_type": doc_type,
        }
        # Qdrant point IDs must be int or UUID. We use uuid4 so re-ingesting
        # a file doesn't collide with a previously-indexed version.
        points.append(
            PointStruct(
                id=str(uuid4()),
                # Named-vector dict: {"dense": [...], "sparse": SparseVector(...)}
                vector={
                    DENSE_VECTOR_NAME: vector,
                    SPARSE_VECTOR_NAME: SparseVector(indices=sp_idx, values=sp_vals),
                },
                payload=payload,
            )
        )
    return points


def delete_stale_points(
    qdrant: QdrantClient,
    source_file: str,
    source_path: str,
    previous_hash: str | None,
) -> None:
    """Purge every existing point for this file before re-upserting.

    We key on BOTH source_file and source_path so that two distinct files
    that happen to share a basename (e.g. `README.md` under two different
    subfolders, `maritime_focus.txt` under two data folders) don't wipe
    each other's chunks.

    Runs on every ingest of a file (even a first-time ingest where
    previous_hash is None) so that a prior half-completed run or a
    collection whose tracker was reset can never leave orphan points
    behind. This is the fix for the duplication incident where 2,017
    duplicate points accumulated in the collection.
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue, FilterSelector
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchValue(value=source_file),
                    ),
                    FieldCondition(
                        key="source_path",
                        match=MatchValue(value=source_path),
                    ),
                ]
            )
        ),
    )


def ingest_file(
    path: Path,
    openai_client: OpenAI,
    qdrant: QdrantClient,
    file_hash: str,
    previous_hash: str | None,
    dry_run: bool = False,
    openrouter_client: OpenAI | None = None,
    use_llm_context: bool = True,
) -> dict[str, Any]:
    """Ingest one source file end-to-end. Returns a stats dict.

    On dry_run we do stages 1-3 (partition + clean + chunk) but skip
    embedding + Qdrant upsert. Useful for validating chunking quality
    on a laptop without spending API credit.

    When ``use_llm_context`` is True and ``openrouter_client`` is set, we
    replace the rule-based breadcrumb prefix with an LLM-generated topic
    sentence built from a summary of the whole document and the chunk
    body. This is Anthropic's contextual-retrieval recipe and materially
    improves recall on paraphrased queries. The rule-based breadcrumb is
    still stored in the payload as ``breadcrumb`` for debugging, and is
    used as a fallback prefix on any chunk where the LLM call fails.
    """
    elements = partition_with_cache(path, CACHE_DIR, file_hash=file_hash)
    if not elements:
        logger.warning("No elements extracted from %s — skipping", path.name)
        return {"elements": 0, "chunks": 0, "vectors": 0, "skipped": True}

    cleaned = clean_elements(elements)
    chunks = chunk_elements(cleaned)

    if not chunks:
        logger.warning("No chunks produced from %s — skipping", path.name)
        return {"elements": len(elements), "chunks": 0, "vectors": 0, "skipped": True}

    if dry_run:
        return {
            "elements": len(elements),
            "chunks": len(chunks),
            "vectors": 0,
            "skipped": False,
            "dry_run": True,
        }

    # Build a per-chunk breadcrumb using the running Title-depth stack
    # over the chunk sequence. This works for every partitioner backend
    # because it uses each chunk's own leading Title (visible in
    # `orig_elements`) rather than requiring the partitioner to populate
    # `parent_id`. As a secondary path, if a chunk has no own-Title but
    # the raw partitioner DID stamp parent_id, we fall back to the
    # parent-chain walker so we don't lose the header when both signals
    # are available.
    element_index = _build_element_index(elements)
    seq_headers = build_context_headers_for_chunks(chunks)

    # Compute a single document summary once per file; the LLM sees the
    # same doc-level context for every chunk of this file.
    doc_summary = build_doc_summary(elements) if use_llm_context else ""

    # Compute the docs-relative posix path once — used both for the
    # delete filter and by the doc_type classifier.
    try:
        rel_path = path.relative_to(DOCS_DIR).as_posix()
    except ValueError:
        rel_path = path.as_posix()

    # Classify doc_type from the on-disk path so retrieval-time filters
    # (admin -> syllabus_outline) work out of the box after ingest. Uses
    # the same rules as scripts/backfill_doc_type.py — kept in sync via
    # direct import so we don't drift.
    from scripts.backfill_doc_type import classify_doc_type
    doc_type = classify_doc_type(rel_path)

    # Precompute breadcrumbs + bodies so we can drive the LLM step in
    # parallel across a chunk's siblings.
    breadcrumbs: list[str] = [
        seq or build_context_header(chunk, element_index)
        for chunk, seq in zip(chunks, seq_headers, strict=True)
    ]
    bodies: list[str] = [chunk.text or "" for chunk in chunks]

    # Concurrent LLM-context calls: DeepSeek-V3 via OpenRouter is fine
    # with ~10-way parallelism, and serial calls make a full re-ingest
    # take hours. We preserve order by mapping (executor.map returns
    # results in submission order).
    llm_sentences: list[str] = [""] * len(chunks)
    if use_llm_context and openrouter_client is not None:
        with ThreadPoolExecutor(max_workers=LLM_CONTEXT_CONCURRENCY) as ex:
            llm_sentences = list(ex.map(
                lambda body: generate_llm_context(body, doc_summary, openrouter_client),
                bodies,
            ))

    context_headers: list[str] = []      # what we actually prepended
    prefixed_texts: list[str] = []
    header_hits = 0
    llm_hits = 0
    for body, breadcrumb, llm_sentence in zip(bodies, breadcrumbs, llm_sentences, strict=True):
        # Choose the header to prepend: LLM sentence first, breadcrumb
        # only when LLM is disabled or its call fails.
        if llm_sentence:
            header = llm_sentence
            llm_hits += 1
        elif breadcrumb:
            header = breadcrumb
        else:
            header = ""

        if header:
            header_hits += 1
            prefixed = f"{header}\n{body}"
        else:
            prefixed = body

        context_headers.append(header)
        prefixed_texts.append(prefixed)

    logger.info(
        "%s: %d/%d chunks got context headers (LLM=%d, breadcrumb=%d)",
        path.name, header_hits, len(chunks), llm_hits, header_hits - llm_hits,
    )

    vectors = embed_texts(openai_client, prefixed_texts)
    sparse_vectors = [encode_sparse_doc(t) for t in prefixed_texts]

    # Remove any prior version of this file's chunks so we don't leave
    # stale content behind on re-ingest. Also runs on first-time ingest
    # as a defensive belt-and-braces against orphan points from earlier
    # broken runs.
    delete_stale_points(qdrant, path.name, rel_path, previous_hash)

    points = chunks_to_points(
        chunks, path, file_hash, vectors, sparse_vectors,
        context_headers, breadcrumbs, prefixed_texts, doc_type,
    )
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start : start + UPSERT_BATCH_SIZE],
            wait=True,
        )

    return {
        "elements": len(elements),
        "chunks": len(chunks),
        "vectors": len(vectors),
        "skipped": False,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest course docs into Qdrant.")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ingest at most N files (after change-detection).",
    )
    p.add_argument(
        "--files",
        nargs="+",
        default=None,
        help="Force-ingest specific files (paths relative to docs/, or absolute).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Partition + chunk only; skip embeddings + Qdrant upsert.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate the Qdrant collection before ingesting.",
    )
    p.add_argument(
        "--no-llm-context",
        action="store_true",
        help=(
            "Skip the LLM-generated contextual header (P2.2). Falls back "
            "to the rule-based Title breadcrumb only. Cuts ingest cost by "
            "one OpenRouter call per chunk (~$0.50/full re-ingest at "
            "DeepSeek-V3 prices)."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="DEBUG-level logging.",
    )
    return p.parse_args()


def resolve_forced_files(names: list[str]) -> list[Path]:
    """--files may be absolute paths or paths relative to docs/. Reject
    anything that doesn't exist so a typo doesn't silently ingest nothing."""
    resolved: list[Path] = []
    for n in names:
        candidate = Path(n)
        if not candidate.is_absolute():
            candidate = (DOCS_DIR / n).resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"--files: no such file: {n}")
        resolved.append(candidate)
    return resolved


def main() -> int:
    args = parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not DOCS_DIR.exists():
        logger.error("docs/ directory not found at %s", DOCS_DIR)
        return 1

    tracker = load_tracker()

    # Decide which files to touch.
    if args.files:
        candidates = resolve_forced_files(args.files)
    else:
        candidates = list(iter_source_files(DOCS_DIR))

    # Change detection: keep only files whose MD5 changed since last run
    # (unless --files was given, which is an explicit override).
    to_process: list[tuple[Path, str, str | None]] = []
    for path in candidates:
        h = file_md5(path)
        prev = tracker["files"].get(str(path))
        if args.files or prev != h:
            to_process.append((path, h, prev))

    if args.limit:
        to_process = to_process[: args.limit]

    if not to_process:
        logger.info("Nothing new to ingest.")
        return 0

    logger.info(
        "%d file(s) to ingest (of %d discovered). Cache dir: %s",
        len(to_process),
        len(candidates),
        CACHE_DIR,
    )

    # Set up clients only if we're actually going to hit the network.
    openai_client: OpenAI | None = None
    openrouter_client: OpenAI | None = None
    qdrant: QdrantClient | None = None
    use_llm_context = not args.no_llm_context
    if not args.dry_run:
        openai_client = build_openai_client()
        qdrant = build_qdrant_client()
        # Build the OpenRouter client only when we actually need it —
        # skips the OPENROUTER_API_KEY check when the user has opted out
        # of LLM-generated headers.
        if use_llm_context:
            openrouter_client = build_openrouter_client()
            logger.info("LLM context headers ENABLED (model=%s)", OPENROUTER_MODEL)
        else:
            logger.info("LLM context headers DISABLED (--no-llm-context)")
        if args.reset:
            existing = {c.name for c in qdrant.get_collections().collections}
            if COLLECTION_NAME in existing:
                qdrant.delete_collection(COLLECTION_NAME)
                logger.info("Dropped collection %r", COLLECTION_NAME)
        ensure_collection(qdrant, COLLECTION_NAME, EMBEDDING_DIM)

    totals = {"files": 0, "elements": 0, "chunks": 0, "vectors": 0, "failed": 0}
    for path, file_hash, prev_hash in tqdm(to_process, unit="file"):
        try:
            stats = ingest_file(
                path=path,
                openai_client=openai_client,  # type: ignore[arg-type]
                qdrant=qdrant,                # type: ignore[arg-type]
                file_hash=file_hash,
                previous_hash=prev_hash,
                dry_run=args.dry_run,
                openrouter_client=openrouter_client,
                use_llm_context=use_llm_context,
            )
            totals["files"] += 1
            totals["elements"] += stats["elements"]
            totals["chunks"] += stats["chunks"]
            totals["vectors"] += stats["vectors"]

            # Only record success in the tracker if we actually indexed
            # (or dry-ran) something. This keeps the tracker honest.
            if not stats.get("skipped"):
                tracker["files"][str(path)] = file_hash
                # Clear any prior failure record for this file.
                tracker.get("failures", {}).pop(str(path), None)
            logger.info(
                "  %-60s  elements=%d  chunks=%d  vectors=%d%s",
                path.name[:60],
                stats["elements"],
                stats["chunks"],
                stats["vectors"],
                "  [DRY-RUN]" if stats.get("dry_run") else "",
            )
        except Exception as e:  # noqa: BLE001 — log and continue; don't lose the run
            totals["failed"] += 1
            logger.exception("FAILED: %s (%s)", path.name, e)
            tracker.setdefault("failures", {})[str(path)] = {
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # Only persist the tracker if we were actually writing to Qdrant.
    # Otherwise a --dry-run would mark files as "done" and the next real
    # run would skip them.
    if not args.dry_run:
        save_tracker(tracker)

    logger.info(
        "Done. files=%d  elements=%d  chunks=%d  vectors=%d  failed=%d",
        totals["files"] - totals["failed"],
        totals["elements"],
        totals["chunks"],
        totals["vectors"],
        totals["failed"],
    )
    return 0 if totals["failed"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
