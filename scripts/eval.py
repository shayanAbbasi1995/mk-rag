"""Lightweight retrieval + generation evaluation harness for the course RAG.

Reads `evals/qa_pairs.jsonl`, runs each question through the same hybrid
retrieval pipeline the Streamlit app uses (multi-query RRF fusion,
optional doc_type filter, cross-encoder rerank), calls DeepSeek-V3 via
OpenRouter for generation, and prints a summary table.

We intentionally do NOT use `ragas` here:
  - `ragas` adds ~50 dependencies and defaults to OpenAI as its judge LLM.
  - For a 25-question set the interpretability of manual, dependency-free
    metrics beats the automation of a black-box scorer.
  - If you later want RAGAS-style faithfulness/answer-relevance scoring:
        pip install ragas==0.2.6 datasets
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision
        dataset = Dataset.from_list([
            {"question": row["question"], "answer": row["answer"],
             "contexts": row["retrieved_texts"], "ground_truth": row["ground_truth"]}
            for row in results
        ])
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    ...and swap the print block below for a RAGAS DataFrame dump.

Usage:
    .venv\\Scripts\\python.exe -m scripts.eval                # run all
    .venv\\Scripts\\python.exe -m scripts.eval --questions 5  # first 5
    .venv\\Scripts\\python.exe -m scripts.eval --skip-generation  # retrieval only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SparseVector,
)

# We reuse the sparse encoder directly. The rest of the retrieval pipeline
# is reimplemented here without the Streamlit `st.cache_resource` decorators
# so the script can run headlessly from a terminal.
from scripts.sparse_encoder import encode_query as encode_sparse_query


# ---------------------------------------------------------------------------
# Config (must match app.py)
# ---------------------------------------------------------------------------

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
QA_PATH = REPO_ROOT / "evals" / "qa_pairs.jsonl"
RESULTS_PATH = REPO_ROOT / "evals" / "eval_results.jsonl"

QDRANT_COLLECTION = "course-docs"
EMBEDDING_MODEL_ID = "text-embedding-3-small"
OPENROUTER_CHAT_MODEL_ID = "deepseek/deepseek-chat"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

RETRIEVER_FETCH_K = 20
RETRIEVER_K = 6
MAX_HITS_PER_SOURCE = 2
MULTI_QUERY_VARIANTS = 3

ADMIN_QUERY_RE = re.compile(
    r"\b("
    r"when\s+(?:is|are|does|do|will|can\s+i|should\s+i)"
    r"|what\s+(?:day|date|time)"
    r"|office\s+hours"
    r"|midterm|final\s+exam"
    r"|assignment\s+(?:is|due)|due\s+date"
    r"|grading\s+scheme|grade\s+(?:breakdown|weight)"
    r"|syllabus|course\s+outline|schedule|deadline"
    r")\b",
    re.IGNORECASE,
)

# Refusal detection for off-syllabus questions. The system prompt in app.py
# instructs the model to prefix refusals with this exact phrase.
REFUSAL_PATTERN = re.compile(
    r"cannot find information about this|"
    r"could not find (anything|any) relevant|"
    r"based on the provided course materials, i cannot",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------


def build_openai_client() -> OpenAI:
    api_key = os.getenv("OPEN_AI_EMBEDDINGS_API_KEY")
    if not api_key:
        raise RuntimeError("OPEN_AI_EMBEDDINGS_API_KEY missing from .env")
    return OpenAI(api_key=api_key)


def build_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing from .env")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def build_qdrant_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")
    if not (url and api_key):
        raise RuntimeError("QDRANT_URL / QDRANT_API_KEY missing from .env")
    return QdrantClient(url=url, api_key=api_key, timeout=60)


# Lazy-loaded cross-encoder — first eval takes ~20s to load the model.
_reranker = None


def get_reranker():  # type: ignore[no-untyped-def]
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker


# ---------------------------------------------------------------------------
# Query understanding
# ---------------------------------------------------------------------------


def generate_query_variants(query: str, or_client: OpenAI) -> list[str]:
    """Return `[query] + up to N paraphrases`. Falls back to `[query]` on
    any LLM error so retrieval never breaks."""
    system = (
        "Generate 3 alternative phrasings of this search query for a "
        "university marketing research course. Return ONLY the 3 queries, "
        "one per line, no numbering."
    )
    try:
        resp = or_client.chat.completions.create(
            model=OPENROUTER_CHAT_MODEL_ID,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
        )
        raw = (resp.choices[0].message.content or "").strip()
        variants: list[str] = []
        for line in raw.splitlines():
            cleaned = re.sub(r"^\s*[-*\d.)\s]+", "", line).strip().strip('"').strip("'")
            if cleaned and cleaned.lower() != query.lower():
                variants.append(cleaned)
        return [query] + variants[:MULTI_QUERY_VARIANTS]
    except Exception:
        return [query]


def detect_doc_type_filter(query: str) -> str | None:
    return "syllabus_outline" if ADMIN_QUERY_RE.search(query) else None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Any]],
    k: int = 60,
) -> list[tuple[Any, float]]:
    scores: dict[tuple[Any, Any], float] = defaultdict(float)
    exemplars: dict[tuple[Any, Any], Any] = {}
    for lst in ranked_lists:
        for rank, point in enumerate(lst):
            payload = point.payload or {}
            key = (
                payload.get("source_file", "Unknown"),
                payload.get("chunk_index"),
            )
            scores[key] += 1.0 / (k + rank + 1)
            if key not in exemplars:
                exemplars[key] = point
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(exemplars[k], s) for k, s in fused]


def embed_batch(openai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    resp = openai_client.embeddings.create(model=EMBEDDING_MODEL_ID, input=texts)
    return [d.embedding for d in resp.data]


def search_docs(
    query: str,
    openai_client: OpenAI,
    or_client: OpenAI,
    qdrant: QdrantClient,
    doc_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Multi-query hybrid RRF retrieval. Returns list of dicts with
    payload + fused rank score + rerank score (populated later)."""
    variants = generate_query_variants(query, or_client)
    variant_vectors = embed_batch(openai_client, variants)
    variant_sparse = [encode_sparse_query(v) for v in variants]

    qdrant_filter: Filter | None = None
    if doc_type_filter:
        qdrant_filter = Filter(
            must=[FieldCondition(key="doc_type", match=MatchValue(value=doc_type_filter))]
        )

    branch_results: list[list[Any]] = []
    for vec, (sp_idx, sp_vals) in zip(variant_vectors, variant_sparse, strict=True):
        try:
            r = qdrant.query_points(
                collection_name=QDRANT_COLLECTION,
                query=vec,
                using=DENSE_VECTOR_NAME,
                limit=RETRIEVER_FETCH_K,
                with_payload=True,
                query_filter=qdrant_filter,
            )
            branch_results.append(list(r.points))
        except Exception:
            pass
        if sp_idx:
            try:
                r = qdrant.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=SparseVector(indices=sp_idx, values=sp_vals),
                    using=SPARSE_VECTOR_NAME,
                    limit=RETRIEVER_FETCH_K,
                    with_payload=True,
                    query_filter=qdrant_filter,
                )
                branch_results.append(list(r.points))
            except Exception:
                pass

    fused = _reciprocal_rank_fusion(branch_results)

    kept: list[dict[str, Any]] = []
    per_source: dict[str, int] = {}
    seen_chunks: set[tuple[Any, Any]] = set()
    for point, score in fused:
        payload = point.payload or {}
        source = payload.get("source_file", "Unknown")
        chunk_idx = payload.get("chunk_index")
        key = (source, chunk_idx)
        if key in seen_chunks:
            continue
        seen_chunks.add(key)
        if per_source.get(source, 0) >= MAX_HITS_PER_SOURCE:
            continue
        per_source[source] = per_source.get(source, 0) + 1
        kept.append({
            "text": payload.get("text", ""),
            "source_file": source,
            "source_path": payload.get("source_path", ""),
            "page_number": payload.get("page_number"),
            "doc_type": payload.get("doc_type"),
            "chunk_index": chunk_idx,
            "rrf_score": score,
        })
        if len(kept) >= RETRIEVER_FETCH_K:
            break

    return kept


def rerank(query: str, docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not docs:
        return docs
    try:
        model = get_reranker()
        pairs = [(query, d["text"]) for d in docs]
        scores = model.predict(pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = float(s)
        docs.sort(key=lambda d: d["rerank_score"], reverse=True)
        return docs[:RETRIEVER_K]
    except Exception:
        # Reranker unavailable — return the top-K by RRF instead.
        return docs[:RETRIEVER_K]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_context(docs: list[dict[str, Any]]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d["source_file"]
        page = d.get("page_number")
        header = f"{src} (p. {page})" if page is not None else src
        lines.append(f"[{i}] Source: {header}\n{d['text'].strip()}")
    return "\n\n".join(lines)


SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful teaching assistant for a Marketing Research course. "
    "Answer the student's question using ONLY the numbered context passages "
    "below.\n\n"
    "Grounding rules:\n"
    "1. Ground every factual claim by citing the supporting passage inline "
    "with [N] where N is the passage number.\n"
    "2. If multiple passages support a claim, cite together: [1][3].\n"
    "3. If the question cannot be answered from the retrieved passages, "
    "respond with EXACTLY: \"Based on the provided course materials, I "
    "cannot find information about this. [General knowledge, not from "
    "course materials:] ...\"\n\n"
    "Context:\n{context}"
)


def generate_answer(
    or_client: OpenAI,
    question: str,
    docs: list[dict[str, Any]],
) -> str:
    context = build_context(docs)
    resp = or_client.chat.completions.create(
        model=OPENROUTER_CHAT_MODEL_ID,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_TEMPLATE.format(context=context)},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def source_coverage(docs: list[dict[str, Any]], hint: str) -> bool:
    """Did any retrieved chunk come from a file matching `hint` (case-insensitive
    substring on source_file or source_path)? Returns True on OFF_SYLLABUS
    hints regardless of retrieval — those are scored via refusal_rate."""
    if hint == "OFF_SYLLABUS":
        return True
    hint_l = hint.lower()
    for d in docs:
        if hint_l in (d.get("source_file") or "").lower():
            return True
        if hint_l in (d.get("source_path") or "").lower():
            return True
    return False


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_PATTERN.search(answer or ""))


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def load_qa_pairs(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"QA fixture not found: {path}")
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--questions", type=int, default=None,
                   help="Evaluate only the first N questions.")
    p.add_argument("--skip-generation", action="store_true",
                   help="Retrieval-only run (no LLM call, no refusal metric).")
    p.add_argument("--out", type=Path, default=RESULTS_PATH)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    qa = load_qa_pairs(QA_PATH)
    if args.questions:
        qa = qa[: args.questions]

    openai_client = build_openai_client()
    or_client = build_openrouter_client()
    qdrant = build_qdrant_client()

    results: list[dict[str, Any]] = []
    for i, row in enumerate(qa, start=1):
        q = row["question"]
        hint = row.get("expected_source_hint", "")
        t0 = time.time()

        doc_type = detect_doc_type_filter(q)
        docs = search_docs(q, openai_client, or_client, qdrant, doc_type)
        docs = rerank(q, docs)

        answer = ""
        if not args.skip_generation:
            if not docs:
                # Mirror app.py's short-circuit: with no context, the model
                # cannot ground an answer. Emit the canonical refusal string
                # so the refusal metric counts this as a proper abstention.
                answer = (
                    "Based on the provided course materials, I cannot find "
                    "information about this."
                )
            else:
                try:
                    answer = generate_answer(or_client, q, docs)
                except Exception as e:
                    answer = f"[ERROR: {e}]"

        elapsed = time.time() - t0
        top_rrf = docs[0]["rrf_score"] if docs else 0.0

        result = {
            "question": q,
            "ground_truth": row.get("ground_truth", ""),
            "expected_source_hint": hint,
            "doc_type_filter": doc_type,
            "answer": answer,
            "answer_length": len(answer),
            "num_retrieved": len(docs),
            "source_coverage": source_coverage(docs, hint),
            "refused": is_refusal(answer),
            "top_rrf_score": round(top_rrf, 5),
            "retrieved_sources": [d["source_file"] for d in docs],
            "elapsed_seconds": round(elapsed, 2),
        }
        results.append(result)
        marker = "REFUSED" if result["refused"] else "answered"
        print(
            f"[{i:>2}/{len(qa)}] {marker:>8}  cov={result['source_coverage']}  "
            f"len={result['answer_length']:>4}  rrf={result['top_rrf_score']:.4f}  "
            f"t={elapsed:.1f}s  | {q[:70]}"
        )

    # Persist results.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Summary metrics.
    n = len(results)
    off_syllabus = [r for r in results if r["expected_source_hint"] == "OFF_SYLLABUS"]
    on_syllabus = [r for r in results if r["expected_source_hint"] != "OFF_SYLLABUS"]

    if on_syllabus:
        avg_len = sum(r["answer_length"] for r in on_syllabus) / len(on_syllabus)
        coverage = sum(1 for r in on_syllabus if r["source_coverage"]) / len(on_syllabus)
        avg_rrf = sum(r["top_rrf_score"] for r in on_syllabus) / len(on_syllabus)
    else:
        avg_len = coverage = avg_rrf = 0.0

    if off_syllabus:
        refusal_rate = sum(1 for r in off_syllabus if r["refused"]) / len(off_syllabus)
    else:
        refusal_rate = float("nan")

    print()
    print("=" * 72)
    print(f"EVAL SUMMARY   (n={n}, on-syllabus={len(on_syllabus)}, off-syllabus={len(off_syllabus)})")
    print("=" * 72)
    print(f"  Avg answer length (chars)      : {avg_len:>7.1f}")
    print(f"  Source coverage (on-syllabus)  : {coverage:>7.1%}")
    print(f"  Avg top-1 RRF score            : {avg_rrf:>7.4f}")
    print(f"  Refusal rate (off-syllabus)    : {refusal_rate:>7.1%}")
    print(f"  Results written to             : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
