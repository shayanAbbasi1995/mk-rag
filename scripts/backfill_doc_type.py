"""Backfill ``doc_type`` payload field on every point in the Qdrant course-docs
collection, then index it so it can be used in filter queries.

Why: ``app.py::_detect_doc_type_filter`` needs the field to route admin
queries (syllabus / midterm / office hours) to ``doc_type=syllabus_outline``
only. Prior to this script the field did not exist in any payload — the
filter would return zero hits and the app would silently fall back to
"no results".

Classification rules (case-insensitive substring on `source_path`):

    slides / session      -> lecture_slides
    TextBook / textbook   -> textbook_chapter
    Outline / outline / syllabus -> syllabus_outline
    readings              -> reading
    (anything else)       -> other

Order matters: syllabus overrides textbook (an "Outline" folder inside a
"TextBook" tree should still be classified as syllabus_outline). The
implementation runs an explicit if/elif chain in the priority order above.

Usage:
    .venv\\Scripts\\python.exe scripts\\backfill_doc_type.py
"""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

load_dotenv()

COLLECTION_NAME = "course-docs"
BATCH_SIZE = 100          # points per set_payload call
SCROLL_LIMIT = 256        # points per scroll page — Qdrant Cloud handles this comfortably

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("backfill_doc_type")


def classify_doc_type(source_path: str) -> str:
    """Map a ``source_path`` payload string to a ``doc_type`` category.

    Case-insensitive substring matching, with priority ordering: syllabus
    beats textbook so ``TextBooks/Outline/foo.md`` classifies as
    ``syllabus_outline`` rather than ``textbook_chapter``.
    """
    if not source_path:
        return "other"
    p = source_path.lower()

    # 1. Syllabus / outline first — outlines can live inside textbook trees.
    if "outline" in p or "syllabus" in p:
        return "syllabus_outline"
    # 2. Textbook chapters.
    if "textbook" in p:
        return "textbook_chapter"
    # 3. Lecture slides (also matches the on-disk `session-NN.qmd` files).
    if "slides" in p or "session" in p:
        return "lecture_slides"
    # 4. Assigned readings.
    if "readings" in p:
        return "reading"
    return "other"


def build_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not (url and api_key):
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY missing from .env")
    return QdrantClient(url=url, api_key=api_key, timeout=60)


def backfill(client: QdrantClient) -> Counter:
    """Scroll every point, classify, and issue batched ``set_payload`` calls.

    Batching by 100 point-ids per ``set_payload`` amortises the round-trip
    cost. We could group by target category and issue one call per category
    (~5 calls total) but that would require holding all ~8,875 point-ids in
    memory before the first write; the batched-100 approach streams.
    """
    counter: Counter[str] = Counter()
    total_scrolled = 0
    total_written = 0

    # We accumulate one small buffer per doc_type so a single set_payload
    # call updates only points that all share the same target payload —
    # Qdrant requires payload to be constant across the point-ids list.
    buffers: dict[str, list[str | int]] = {}

    def flush(doc_type: str) -> None:
        nonlocal total_written
        pts = buffers.get(doc_type) or []
        if not pts:
            return
        client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"doc_type": doc_type},
            points=pts,
            wait=True,
        )
        total_written += len(pts)
        buffers[doc_type] = []

    next_offset = None
    while True:
        page, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=True,
            with_vectors=False,
            limit=SCROLL_LIMIT,
            offset=next_offset,
        )
        if not page:
            break

        for point in page:
            payload = point.payload or {}
            source_path = payload.get("source_path", "") or ""
            doc_type = classify_doc_type(source_path)

            counter[doc_type] += 1
            buffers.setdefault(doc_type, []).append(point.id)
            if len(buffers[doc_type]) >= BATCH_SIZE:
                flush(doc_type)

        total_scrolled += len(page)
        logger.info(
            "Scrolled %d points (written so far: %d)",
            total_scrolled, total_written,
        )

        if next_offset is None:
            break

    # Drain remaining partial buffers.
    for doc_type in list(buffers.keys()):
        flush(doc_type)

    logger.info(
        "Backfill complete: scrolled=%d  written=%d",
        total_scrolled, total_written,
    )
    return counter


def ensure_doc_type_index(client: QdrantClient) -> None:
    """Create a keyword payload index on ``doc_type`` so it can be used in
    filter queries. Idempotent: Qdrant returns a benign error if it
    already exists, which we swallow."""
    try:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="doc_type",
            field_schema=PayloadSchemaType.KEYWORD,
            wait=True,
        )
        logger.info("Created payload index on 'doc_type' (keyword)")
    except Exception as e:  # noqa: BLE001
        # Most common failure mode is "index already exists" — safe to skip.
        logger.info("Payload index on 'doc_type' not created (may already exist): %s", e)


def main() -> int:
    client = build_qdrant_client()

    # Sanity: confirm the collection exists before scrolling.
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        logger.error("Collection %r does not exist in Qdrant", COLLECTION_NAME)
        return 1

    counter = backfill(client)
    ensure_doc_type_index(client)

    total = sum(counter.values())
    parts = ", ".join(f"{k}={v}" for k, v in sorted(counter.items()))
    print(f"\nBackfilled {total} points: {parts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
