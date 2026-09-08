"""Ingest a single Zoom / lecture video transcript into the course-docs
Qdrant collection.

Usage: .venv\\Scripts\\python.exe scripts\\ingest_video.py <path/to/transcript.srt> --video-name "Lecture 1"
Supports SRT and VTT formats.
Add zoom recordings or MP4 transcripts to docs/videos/ and run this script for each.
The ingested chunks will appear in the course-docs Qdrant collection with doc_type="video_transcript".

Design notes
------------
Kept as a standalone script (not folded into `ingest.py`) because video
transcripts want a fundamentally different chunking policy — fixed-length
time windows with overlap, not title-driven semantic segmentation. Mixing
the two flows in one file would make ingest.py's control flow hard to
follow. When we're ready to run this on 10+ videos in one go, we can
grow a `--transcripts-dir` loop in main().

Chunking uses ~5-minute windows with ~30 seconds of overlap, matching
common recipes for lecture-transcript RAG. Chunks are dense-embedded
only — we skip the sparse (BM25) branch because transcript vocabulary is
noisy (filler words, disfluencies) and the sparse-vector recall gain we
get on typed course materials mostly disappears on speech-to-text output.

Payload schema (per point):
    doc_type:       "video_transcript"
    video_name:     human-readable label passed via --video-name
    source_file:    transcript filename (used by app.py::_format_citation)
    start_seconds:  chunk start offset in the video
    end_seconds:    chunk end offset in the video
    start_time:     "MM:SS" or "H:MM:SS"
    end_time:       "MM:SS" or "H:MM:SS"
    text:           concatenated transcript text for this window
    chunk_index:    per-video ordinal for dedup
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

load_dotenv()

COLLECTION_NAME = "course-docs"
DENSE_VECTOR_NAME = "dense"
EMBEDDING_MODEL_ID = "text-embedding-3-small"

# Time windows: 5 minutes with 30 seconds of overlap. Big enough to fit a
# coherent explanation segment; small enough that a single chunk still
# fits comfortably in the embedding model's 8k-token context window.
DEFAULT_WINDOW_SECONDS = 300
DEFAULT_OVERLAP_SECONDS = 30

# OpenAI embedding batching. Same as scripts/ingest.py — keeps request
# bodies small and gives us tight retry granularity if a batch fails.
EMBEDDING_BATCH_SIZE = 100

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_video")


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


# SRT and VTT both use HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT). We accept
# either separator via a single regex with an alternation on `[,.]`.
_TIMESTAMP_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


def _timestamp_to_seconds(ts: str) -> int:
    """Convert an SRT/VTT timestamp string to whole seconds.

    We drop the millisecond component: the caller uses windows measured
    in seconds, so sub-second precision is discarded on either side.
    """
    m = _TIMESTAMP_RE.match(ts.strip())
    if not m:
        raise ValueError(f"Malformed timestamp: {ts!r}")
    h, mnt, s, _ms = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + int(s)


def format_timestamp(seconds: int) -> str:
    """Return `MM:SS` for <1h, `H:MM:SS` for >=1h.

    Used in citations so students can jump to the exact spot in the
    recording. Keeps citations short for the common case of a
    45-minute lecture while still displaying correctly for longer
    workshop recordings.
    """
    if seconds < 0:
        seconds = 0
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------


# SRT arrow separator: "00:00:12,300 --> 00:00:15,900"
_SRT_ARROW_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
)


def parse_srt(path: Path) -> list[dict[str, Any]]:
    """Parse an SRT file into `[{start_seconds, end_seconds, text}]`.

    SRT format (one cue):

        3
        00:00:12,300 --> 00:00:15,900
        Marketing research begins with a
        clear statement of the decision problem.

        4
        ...

    We ignore the numeric cue index and concatenate all lines under a
    cue as its text. Blank lines separate cues.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    segments: list[dict[str, Any]] = []
    # Split on runs of blank lines: robust to Windows/Unix line endings.
    blocks = re.split(r"\n\s*\n", raw.strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # Find the first line containing a timestamp arrow. This makes
        # the parser tolerant of missing/extra cue-index lines.
        ts_line_idx = next(
            (i for i, ln in enumerate(lines) if _SRT_ARROW_RE.search(ln)),
            None,
        )
        if ts_line_idx is None:
            continue
        m = _SRT_ARROW_RE.search(lines[ts_line_idx])
        if not m:
            continue
        start = _timestamp_to_seconds(m.group(1))
        end = _timestamp_to_seconds(m.group(2))
        text = " ".join(lines[ts_line_idx + 1:]).strip()
        if text:
            segments.append({
                "start_seconds": start,
                "end_seconds": end,
                "text": text,
            })
    return segments


# ---------------------------------------------------------------------------
# WebVTT parsing
# ---------------------------------------------------------------------------


# WebVTT has an optional "WEBVTT" header, allows optional cue identifiers,
# and uses `.` between seconds and milliseconds. Cue block otherwise
# looks like SRT. We reuse the same arrow regex since it accepts either
# separator, and just strip the header.
def parse_vtt(path: Path) -> list[dict[str, Any]]:
    """Parse a WebVTT file into `[{start_seconds, end_seconds, text}]`.

    WebVTT differs from SRT in three small ways: the file may open with
    "WEBVTT" and metadata lines; cue identifiers are optional (SRT
    requires them); timestamps use `.` instead of `,` for milliseconds.
    Our shared timestamp regex accepts both separators so the only real
    special case is dropping the WEBVTT header and any NOTE blocks.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    # Drop the WEBVTT header line and any metadata that follows it up to
    # the first blank line.
    raw = re.sub(r"^WEBVTT[^\n]*\n(?:[^\n]+\n)*?\s*\n", "", raw, count=1)
    segments: list[dict[str, Any]] = []
    blocks = re.split(r"\n\s*\n", raw.strip())
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        # Skip NOTE and STYLE blocks — WebVTT metadata, not caption cues.
        if lines[0].startswith(("NOTE", "STYLE", "REGION")):
            continue
        ts_line_idx = next(
            (i for i, ln in enumerate(lines) if _SRT_ARROW_RE.search(ln)),
            None,
        )
        if ts_line_idx is None:
            continue
        m = _SRT_ARROW_RE.search(lines[ts_line_idx])
        if not m:
            continue
        start = _timestamp_to_seconds(m.group(1))
        end = _timestamp_to_seconds(m.group(2))
        text = " ".join(lines[ts_line_idx + 1:]).strip()
        if text:
            segments.append({
                "start_seconds": start,
                "end_seconds": end,
                "text": text,
            })
    return segments


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_transcript(
    segments: list[dict[str, Any]],
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
) -> list[dict[str, Any]]:
    """Group per-caption segments into fixed-length overlapping windows.

    Windows start every `(window_seconds - overlap_seconds)` seconds so
    consecutive chunks share `overlap_seconds` of content. A concept that
    starts near a window boundary still appears whole in one of the two
    overlapping windows.

    Each returned chunk carries the actual start/end of the *segments it
    contains* (rather than the notional window bounds), which makes
    citations point to the real transcript span instead of the padded
    window.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if overlap_seconds < 0 or overlap_seconds >= window_seconds:
        raise ValueError("overlap_seconds must be in [0, window_seconds)")
    if not segments:
        return []

    stride = window_seconds - overlap_seconds
    end_time = segments[-1]["end_seconds"]

    chunks: list[dict[str, Any]] = []
    window_start = 0
    chunk_index = 0
    while window_start <= end_time:
        window_end = window_start + window_seconds
        # Include a segment if its start falls within [window_start, window_end).
        contained = [
            s for s in segments
            if window_start <= s["start_seconds"] < window_end
        ]
        if contained:
            text = " ".join(s["text"] for s in contained).strip()
            if text:
                chunks.append({
                    "chunk_index": chunk_index,
                    "start_seconds": contained[0]["start_seconds"],
                    "end_seconds": contained[-1]["end_seconds"],
                    "text": text,
                })
                chunk_index += 1
        window_start += stride
    return chunks


# ---------------------------------------------------------------------------
# Client factories
# ---------------------------------------------------------------------------


def build_openai_client() -> OpenAI:
    api_key = os.getenv("OPEN_AI_EMBEDDINGS_API_KEY")
    if not api_key:
        raise RuntimeError("OPEN_AI_EMBEDDINGS_API_KEY missing from .env")
    return OpenAI(api_key=api_key)


def build_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not (url and api_key):
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY missing from .env")
    return QdrantClient(url=url, api_key=api_key, timeout=60)


def embed_texts(
    openai_client: OpenAI,
    texts: list[str],
    batch_size: int = EMBEDDING_BATCH_SIZE,
) -> list[list[float]]:
    """Batch-embed texts, preserving order. No retry loop here because the
    call site is a single script run; if the network drops, re-run."""
    out: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = openai_client.embeddings.create(
            model=EMBEDDING_MODEL_ID,
            input=batch,
        )
        out.extend(d.embedding for d in resp.data)
    return out


# ---------------------------------------------------------------------------
# Ingest entry point
# ---------------------------------------------------------------------------


def ingest_video_file(
    transcript_path: Path,
    video_name: str,
    openai_client: OpenAI,
    qdrant_client: QdrantClient,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    overlap_seconds: int = DEFAULT_OVERLAP_SECONDS,
) -> int:
    """Parse -> chunk -> embed -> upsert. Returns number of points written.

    Dense-only: we skip the sparse (BM25) index because ASR transcripts
    have too much filler ("uh", "so", "right") to benefit from lexical
    matching, and the sparse index would waste storage.
    """
    ext = transcript_path.suffix.lower()
    if ext == ".srt":
        segments = parse_srt(transcript_path)
    elif ext == ".vtt":
        segments = parse_vtt(transcript_path)
    else:
        raise ValueError(f"Unsupported transcript extension: {ext}. Use .srt or .vtt.")

    if not segments:
        logger.warning("No cue segments parsed from %s", transcript_path.name)
        return 0

    chunks = chunk_transcript(
        segments,
        window_seconds=window_seconds,
        overlap_seconds=overlap_seconds,
    )
    if not chunks:
        logger.warning("No chunks produced from %s", transcript_path.name)
        return 0

    logger.info(
        "%s: %d segments -> %d chunks (window=%ds, overlap=%ds)",
        transcript_path.name, len(segments), len(chunks),
        window_seconds, overlap_seconds,
    )

    texts = [c["text"] for c in chunks]
    vectors = embed_texts(openai_client, texts)

    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        payload = {
            "doc_type": "video_transcript",
            "video_name": video_name,
            "source_file": transcript_path.name,
            "start_seconds": chunk["start_seconds"],
            "end_seconds": chunk["end_seconds"],
            "start_time": format_timestamp(chunk["start_seconds"]),
            "end_time": format_timestamp(chunk["end_seconds"]),
            "text": chunk["text"],
            "chunk_index": chunk["chunk_index"],
        }
        points.append(PointStruct(
            id=str(uuid4()),
            # Named-vector dict with ONLY the dense entry. Qdrant accepts
            # a subset of named vectors — points without a sparse entry
            # simply won't surface from sparse-branch queries, which is
            # exactly what we want for transcript content.
            vector={DENSE_VECTOR_NAME: vector},
            payload=payload,
        ))

    # Batch upserts so a single failed request doesn't lose the whole
    # video's worth of work.
    for i in range(0, len(points), 100):
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i + 100],
            wait=True,
        )
    logger.info("Upserted %d points for %s", len(points), video_name)
    return len(points)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingest a video transcript into the course-docs collection.")
    p.add_argument("transcript_path", type=Path, help="Path to .srt or .vtt file.")
    p.add_argument("--video-name", required=True, help="Human-readable video label (used in citations).")
    p.add_argument("--window-seconds", type=int, default=DEFAULT_WINDOW_SECONDS)
    p.add_argument("--overlap-seconds", type=int, default=DEFAULT_OVERLAP_SECONDS)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.transcript_path.exists():
        logger.error("Transcript not found: %s", args.transcript_path)
        return 1

    openai_client = build_openai_client()
    qdrant_client = build_qdrant_client()
    n = ingest_video_file(
        args.transcript_path,
        args.video_name,
        openai_client,
        qdrant_client,
        window_seconds=args.window_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    print(f"Done. wrote {n} points for {args.video_name!r}")
    return 0 if n > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
