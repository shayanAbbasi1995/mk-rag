"""Live retrieval diagnostic for the Marketing Research course RAG.

Embeds a set of representative questions, runs top-k dense search against
the Qdrant `course-docs` collection, and prints scores + payload snippets
so we can eyeball retrieval quality (score spread, source diversity,
chunk cleanliness, orphaned mid-sentence starts, etc.).

Usage:
    .venv\\Scripts\\python.exe -m scripts.eval_retrieval
"""

from __future__ import annotations

import os
import sys
from statistics import mean, pstdev

# Windows cp1252 default codepage chokes on ligatures (fi, fl, curly
# quotes, etc.) present in textbook PDFs. Reconfigure stdout to utf-8
# with a replacement fallback so a single glyph doesn't crash the run.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv()

EMBEDDING_MODEL: str = "text-embedding-3-small"
COLLECTION: str = "course-docs"
TOP_K: int = 8  # pull a few extra beyond the app's default of 4 so we can
                # see whether hits 5-8 are meaningfully worse or basically
                # tied with hits 1-4.

QUERIES: list[str] = [
    # Broad conceptual — should map well to lecture-level content.
    "What is conjoint analysis?",
    "How do you design a survey questionnaire?",
    "What is the difference between qualitative and quantitative research?",
    "Explain focus groups in marketing research",
    "What are the steps in the marketing research process?",
    # Jargon / acronym — tests whether dense search catches exact terms.
    "What is a Likert scale?",
    "Explain NPS (Net Promoter Score)",
    "How does ANOVA work in marketing research?",
    # Plain-English rephrase of a technical concept — tests vocabulary
    # mismatch: does the retriever find the perceptual-mapping section
    # when the student asks it colloquially?
    "How do I show where brands sit relative to each other on a chart?",
    # Course-specific meta question — probes whether the outline/syllabus
    # was ingested cleanly.
    "What topics are covered in the course?",
]


def main() -> int:
    oai = OpenAI(api_key=os.getenv("OPEN_AI_EMBEDDINGS_API_KEY"))
    qd = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        timeout=60,
    )

    # Sanity: how many points does the collection actually hold?
    info = qd.get_collection(COLLECTION)
    print(f"Collection `{COLLECTION}` points_count={info.points_count}\n")

    for q in QUERIES:
        vec = (
            oai.embeddings.create(model=EMBEDDING_MODEL, input=q)
            .data[0]
            .embedding
        )
        hits = qd.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=TOP_K,
            with_payload=True,
        )
        scores = [h.score for h in hits]
        sources = [(h.payload or {}).get("source_file", "?") for h in hits]
        n_unique_sources = len(set(sources))

        print("=" * 100)
        print(f"Q: {q}")
        if scores:
            print(
                f"   score stats: max={max(scores):.3f}  min={min(scores):.3f}  "
                f"mean={mean(scores):.3f}  spread={max(scores) - min(scores):.3f}  "
                f"stdev={pstdev(scores):.3f}"
            )
            print(f"   unique sources in top-{TOP_K}: {n_unique_sources}")
        for rank, h in enumerate(hits, start=1):
            p = h.payload or {}
            text = (p.get("text") or "").replace("\n", " ")
            snippet = text[:160]
            print(
                f"  {rank:>2}. [{h.score:.3f}] "
                f"{p.get('source_file','?')} "
                f"p={p.get('page_number','?')} "
                f"cat={p.get('element_category','?')} "
                f"chunk_idx={p.get('chunk_index','?')} "
                f"len={len(text)}"
            )
            print(f"       {snippet}{'...' if len(text) > 160 else ''}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
