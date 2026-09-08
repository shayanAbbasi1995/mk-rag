"""Audit Qdrant `course-docs` for duplicate chunks.

Two files or two ingest passes may have deposited the same (source_file,
chunk_index) pair as two Qdrant points with different UUIDs. This script
scrolls the whole collection and reports:

    * total points
    * unique (source_file, chunk_index) keys
    * top 15 duplicated keys (with counts and point-id samples)
    * per-source duplication share

Read-only — makes no changes.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, defaultdict

from dotenv import load_dotenv
from qdrant_client import QdrantClient

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

load_dotenv()

COLLECTION = "course-docs"


def main() -> int:
    qd = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=120,
    )

    info = qd.get_collection(COLLECTION)
    print(f"Reported points_count = {info.points_count}")

    # Scroll all points. Only pull the small identity fields we need to
    # bucket by; skip `text` to keep the payload small on the wire.
    key_counts: Counter[tuple[str, int]] = Counter()
    key_point_ids: dict[tuple[str, int], list[str]] = defaultdict(list)
    per_source_counts: Counter[str] = Counter()
    per_source_unique: dict[str, set[int]] = defaultdict(set)

    next_page = None
    scanned = 0
    while True:
        points, next_page = qd.scroll(
            collection_name=COLLECTION,
            limit=1000,
            with_payload=["source_file", "chunk_index"],
            with_vectors=False,
            offset=next_page,
        )
        if not points:
            break
        for pt in points:
            p = pt.payload or {}
            sf = p.get("source_file", "?")
            ci = p.get("chunk_index")
            if ci is None:
                continue
            key = (sf, int(ci))
            key_counts[key] += 1
            if len(key_point_ids[key]) < 3:
                key_point_ids[key].append(str(pt.id))
            per_source_counts[sf] += 1
            per_source_unique[sf].add(int(ci))
        scanned += len(points)
        if next_page is None:
            break

    print(f"Scanned {scanned} points")
    unique_keys = len(key_counts)
    dup_keys = sum(1 for v in key_counts.values() if v > 1)
    dup_extra_points = sum(v - 1 for v in key_counts.values() if v > 1)
    print(
        f"Unique (source_file, chunk_index) keys: {unique_keys}\n"
        f"Keys with duplicates: {dup_keys}\n"
        f"Extra (duplicate) points beyond one-per-key: {dup_extra_points}\n"
    )

    print("Top 15 duplicated keys:")
    for (sf, ci), n in key_counts.most_common(15):
        if n == 1:
            break
        sample_ids = key_point_ids[(sf, ci)]
        print(f"  {n:>3}x  {sf}  chunk_index={ci}  sample_ids={sample_ids}")

    print("\nPer-source duplication share (sources with any duplication):")
    rows = []
    for sf, total in per_source_counts.items():
        uniq = len(per_source_unique[sf])
        if total != uniq:
            rows.append((total - uniq, sf, total, uniq))
    for extra, sf, total, uniq in sorted(rows, reverse=True):
        print(f"  +{extra:>4} dup   {sf}   ({total} points, {uniq} unique chunk_index)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
