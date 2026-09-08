"""Remove duplicate points from Qdrant `course-docs`.

A point is considered a duplicate when another point in the collection
has the same `(source_file, source_path, chunk_index)` payload. The
duplicates were introduced when the ingester was run more than once
against the same file (either via a collection reset that left the
tracker stale, or because two files with the same basename live in
different subfolders and the stale-delete step only matched by
basename).

De-dup strategy:
  * Scroll the whole collection, bucketing point-ids by dedup key.
  * For each bucket with >1 point, keep the *first* point (sorted by
    point id string for determinism) and mark the rest for deletion.
  * Delete in batches of 500. Idempotent: running twice is a no-op.

Run in --dry-run first to see the counts. Then rerun without --dry-run
to actually delete. Read-modifies collection but does not re-embed or
touch source files.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

load_dotenv()

COLLECTION = "course-docs"
DELETE_BATCH = 500


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report duplicates but don't delete.")
    args = parser.parse_args()

    qd = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=300,
    )

    before = qd.get_collection(COLLECTION).points_count
    print(f"points before: {before}")

    buckets: dict[tuple[str, str, int], list[str]] = defaultdict(list)

    next_page = None
    scanned = 0
    while True:
        pts, next_page = qd.scroll(
            collection_name=COLLECTION,
            limit=1000,
            with_payload=["source_file", "source_path", "chunk_index"],
            with_vectors=False,
            offset=next_page,
        )
        if not pts:
            break
        for p in pts:
            pl = p.payload or {}
            sf = pl.get("source_file", "?")
            sp = pl.get("source_path", "?")
            ci = pl.get("chunk_index")
            if ci is None:
                # Un-keyable point: leave it in place rather than risk a
                # false-positive delete of legitimately unique content.
                continue
            buckets[(sf, sp, int(ci))].append(str(p.id))
        scanned += len(pts)
        if next_page is None:
            break

    print(f"scanned: {scanned}")

    to_delete: list[str] = []
    dup_buckets = 0
    for key, ids in buckets.items():
        if len(ids) > 1:
            dup_buckets += 1
            # Deterministic keep: lexicographically smallest UUID string.
            ids_sorted = sorted(ids)
            to_delete.extend(ids_sorted[1:])

    print(f"duplicate buckets: {dup_buckets}")
    print(f"points to delete: {len(to_delete)}")

    if args.dry_run:
        print("[dry-run] no deletes performed")
        return 0

    for start in range(0, len(to_delete), DELETE_BATCH):
        batch = to_delete[start : start + DELETE_BATCH]
        qd.delete(
            collection_name=COLLECTION,
            points_selector=PointIdsList(points=batch),  # type: ignore[arg-type]
            wait=True,
        )
        print(f"  deleted {start + len(batch)} / {len(to_delete)}")

    after = qd.get_collection(COLLECTION).points_count
    print(f"points after: {after}   (delta: {before - after})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
