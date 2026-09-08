"""Chunk-quality inspector for the RAG ingestion pipeline.

Runs partition -> clean -> chunk on one or more files (or a small sample
from the corpus) and prints chunk statistics + spot-check text. This is
the fast-feedback loop for iterating on the cleaner / chunker without
paying for embeddings, per the reference PDF's "Decouples Pipeline
Stages" caching motivation.

Usage:
    python -m scripts.inspect_chunks                     # sample 4 files
    python -m scripts.inspect_chunks --files a.pdf b.md  # specific files
    python -m scripts.inspect_chunks --sample 10         # random 10 files
    python -m scripts.inspect_chunks --show-small 100    # show chunks <100 chars
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from scripts.chunker import chunk_elements, clean_elements
from scripts.partitioner import (
    file_md5,
    iter_source_files,
    partition_with_cache,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
CACHE_DIR = REPO_ROOT / "data" / "elements_cache"

DEFAULT_SAMPLE = [
    "TextBooks/Essentials of Marketing/essentials-of-marketing-research - Chapter 1.pdf",
    "slides/Session 1/session-01.qmd",
    "slides/Session 1/session-01.html",
    "Outline/M731 - Marketing Research Outline - Shay Abbasi.docx",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--files", nargs="+", help="Files relative to docs/ or absolute.")
    p.add_argument("--sample", type=int, help="Randomly sample N files from docs/.")
    p.add_argument(
        "--show-small",
        type=int,
        default=100,
        help="Print chunks shorter than N chars (default 100).",
    )
    return p.parse_args()


def resolve(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        cand = Path(p)
        if not cand.is_absolute():
            cand = (DOCS_DIR / p).resolve()
        if not cand.exists():
            raise FileNotFoundError(p)
        out.append(cand)
    return out


def main() -> int:
    args = parse_args()

    if args.files:
        paths = resolve(args.files)
    elif args.sample:
        all_files = list(iter_source_files(DOCS_DIR))
        random.seed(0)  # deterministic sampling for reproducible checks
        paths = random.sample(all_files, min(args.sample, len(all_files)))
    else:
        paths = resolve(DEFAULT_SAMPLE)

    for p in paths:
        elements = partition_with_cache(p, CACHE_DIR, file_hash=file_md5(p))
        cleaned = clean_elements(elements)
        chunks = chunk_elements(cleaned)
        sizes = [len(c.text or "") for c in chunks]
        rel = p.relative_to(DOCS_DIR) if DOCS_DIR in p.parents else p
        print(f"=== {rel} ===")
        print(f"  elements={len(elements)}  cleaned={len(cleaned)}  chunks={len(chunks)}")
        if sizes:
            print(
                f"  chunk chars: min={min(sizes)}  median={sorted(sizes)[len(sizes)//2]}"
                f"  max={max(sizes)}  mean={sum(sizes)/len(sizes):.0f}"
            )
            small = [(i, c) for i, c in enumerate(chunks) if len(c.text or "") < args.show_small]
            if small:
                print(f"  {len(small)} chunk(s) <{args.show_small} chars:")
                for i, c in small[:5]:
                    body = (c.text or "")[:150].replace("\n", " | ")
                    print(f"    [{i}] {len(c.text)}c: {body}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
