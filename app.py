"""Streamlit chat UI for a course RAG assistant.

Set COURSE_NAME in .env (or Streamlit secrets) to brand the UI for your course.

Generation goes through OpenRouter (DeepSeek-V3) because it's ~10x cheaper
than calling OpenAI's chat models directly. Embeddings still hit OpenAI
because `text-embedding-3-small` is $0.02/1M tokens with no comparable
free-tier alternative, and swapping embedding models would require
rebuilding the Qdrant collection (dim is immutable).
"""

from __future__ import annotations

import os
import pathlib
import re
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchValue,
    Prefetch,
    SparseVector,
)

from scripts.sparse_encoder import encode_query as encode_sparse_query

# Load .env from the repo root. override=False so Streamlit Cloud's
# real secrets always win over any stale local .env values.
load_dotenv(override=False)

# --- Model IDs --------------------------------------------------------------
# These are the real provider-side model identifiers. The env vars
# OPEN_AI_EMBEDDINGS_NAME and OPENROUTER_API_NAME in .env are *project
# labels* (e.g. "mr-rag-open"), NOT model IDs — do not pass them to the
# provider clients. See .claude/agent-memory/.../project-infra.md.
EMBEDDING_MODEL_ID: str = "text-embedding-3-small"  # 1536-dim, cosine
OPENROUTER_CHAT_MODEL_ID: str = "deepseek/deepseek-chat"  # DeepSeek-V3
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
QDRANT_COLLECTION: str = "course-docs"

# Named-vector labels — must match scripts/ingest.py's DENSE_VECTOR_NAME
# and SPARSE_VECTOR_NAME. Qdrant routes each Prefetch to the matching
# index by name.
DENSE_VECTOR_NAME: str = "dense"
SPARSE_VECTOR_NAME: str = "sparse"

# Retrieval tuning ----------------------------------------------------------
# We over-fetch and then de-dup + threshold + truncate to RETRIEVER_K.
#   RETRIEVER_FETCH_K: raw hits pulled from EACH of the two prefetch
#     branches (dense and sparse) before RRF fusion. Set high enough
#     that after fusion, dedup, and diversity capping we still have
#     RETRIEVER_K quality hits to pass to the LLM.
#   RETRIEVER_K:      final number of chunks fed into the prompt.
#   MAX_HITS_PER_SOURCE: cap to force source-file diversity. Without this,
#     a well-covered topic ("questionnaire design") returns 6+ chunks
#     from the same textbook chapter and starves other sources.
#   CHAT_HISTORY_TURNS: how many prior user+assistant pairs to include
#     in the LLM context so follow-ups ("give me an example") resolve.
#
# NOTE: the previous cosine SCORE_FLOOR was removed when we moved to RRF
# fusion. RRF scores are reciprocal-rank sums bounded by ~2/(60+1) ≈ 0.033
# per matched branch — they carry no absolute similarity meaning, so a
# floor tuned on cosine (0.35) would drop every hit. Off-topic hits are
# now filtered by (a) the sparse branch producing zero score on queries
# with no vocabulary overlap and (b) MAX_HITS_PER_SOURCE + the diversity
# rules below.
RETRIEVER_FETCH_K: int = 20
RETRIEVER_K: int = 6
MAX_HITS_PER_SOURCE: int = 2
CHAT_HISTORY_TURNS: int = 10

# Session-aware retrieval augmentation ---------------------------------------
# When a student asks about a specific session (e.g. "what are the topics in
# session 1?"), the actual session-NN.qmd/.html chunks often do NOT rank at
# the top of pure semantic retrieval — the slides describe the content but
# don't self-label as "the topics of session N", so `README.md` (which has a
# per-session topic table) and `overview.qmd` (session-by-session plan) win
# the cosine race, and `session-03.qmd` even beats session-01 for a Session-1
# query because its opener happens to contain the literal word "topics".
#
# Fix: detect a session number in the query, do a Qdrant scroll for chunks
# whose `source_file` matches the corresponding session filenames, cosine-
# rank those chunks against the query, and pin the top few in front of the
# LLM context. This is deterministic — a student who asks about Session N
# always sees Session N content, no matter how the embedder ranks it.
SESSION_QUERY_RE = re.compile(
    r"\b(?:session|week|lecture|wk)\s*#?\s*(\d{1,2})\b",
    re.IGNORECASE,
)
# How many session-specific chunks to pin at the top of the context. Kept
# small so semantic hits still get most of the RETRIEVER_K budget.
SESSION_PIN_TOP_N: int = 3

# Term-pinning: jargon terms whose dense embeddings get buried when the
# query also contains a competing high-frequency term. Each entry is
# (detection_pattern, probe_query, pin_top_n). When the query matches the
# pattern, we run a dedicated dense search with the probe and prepend the
# top chunks unconditionally — exactly like session pinning. The probe query
# is semantically rich so it surfaces specific content even when the student
# phrases the question in a comparison form ("A vs B") that causes A to lose
# the cosine race to B.
TERM_PINNING: list[tuple[re.Pattern[str], str, int]] = [
    (
        re.compile(r"\b(?:cluster\s+analysis|clustering)\b", re.IGNORECASE),
        "cluster analysis market segmentation grouping respondents cases K-means hierarchical",
        2,
    ),
    (
        re.compile(r"\bpart.?worth\b", re.IGNORECASE),
        "part-worth utility conjoint analysis attribute level value contribution preference",
        3,
    ),
]

# --- Multi-query expansion + reranker knobs --------------------------------
# When multi-query is enabled we generate N alternative phrasings of the
# rewritten query and fuse per-branch Prefetch results via RRF. This trades
# 1 extra LLM call (variant generation) + N-1 extra embed calls (batched)
# for materially better recall on jargon-heavy or ambiguous queries.
MULTI_QUERY_VARIANTS: int = 3  # produces 4 total (original + 3 variants)

# Cross-encoder reranker settings (P2.3). We over-fetch RETRIEVER_FETCH_K
# candidates from Qdrant and let the cross-encoder pick the top RETRIEVER_K
# by (query, passage) relevance score. RETRIEVER_FETCH_K was bumped from 12
# to 20 so the reranker has a wider candidate pool to work with.
RERANKER_MODEL_ID: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Admin/syllabus query detection (P2.1). When a student asks about admin
# concerns (dates, deadlines, office hours, exam schedule), we filter
# retrieval to doc_type="syllabus_outline" so the model doesn't pull an
# example midterm question from a textbook chapter that happens to
# mention "midterm" and answer with textbook content.
ADMIN_QUERY_RE = re.compile(
    # Admin-query triggers. We anchor 'when' with a following auxiliary
    # ('when is/are/does/will...') so questions like "when would stratified
    # sampling be preferred" or "when do I use Cronbach's alpha" don't get
    # mis-routed to the syllabus filter. 'do' was removed because "when do
    # I use X?" is a methodology question, not a scheduling one.
    # Unconditional keywords (midterm, office hours, etc.) are strong
    # enough signals on their own.
    r"\b("
    r"when\s+(?:is|are|does|will|can\s+i|should\s+i)"
    r"|what\s+(?:day|date|time)"
    r"|office\s+hours"
    r"|midterm|final\s+exam"
    r"|assignment\s+(?:is|due)|due\s+date"
    r"|grading\s+scheme|grade\s+(?:breakdown|weight)"
    r"|syllabus|course\s+outline|schedule|deadline"
    r")\b",
    re.IGNORECASE,
)

# Heuristic for whether a follow-up needs conversational rewriting.
# Pronoun/reference tokens whose presence in a short query strongly
# suggests we need earlier context to resolve them.
_COREF_TOKENS = re.compile(
    r"\b(it|its|this|that|these|those|they|them|their|he|she|his|her|"
    r"one|ones|example|another|same|previous|above|earlier)\b",
    re.IGNORECASE,
)

# Course branding — set COURSE_NAME in .env or Streamlit secrets to customise
# the UI title without touching code.
COURSE_NAME: str = os.getenv("COURSE_NAME", "Course Assistant: Ask a question")


st.set_page_config(page_title=COURSE_NAME, layout="centered")
st.title(COURSE_NAME)


_SECRETS_PATHS = [
    pathlib.Path.home() / ".streamlit" / "secrets.toml",
    pathlib.Path(__file__).parent / ".streamlit" / "secrets.toml",
]
# We treat Streamlit Cloud as "has secrets" too. Cloud mounts dashboard
# secrets at ~/.streamlit/secrets.toml (so the file-existence check
# usually succeeds), but the STREAMLIT_RUNTIME_ENV / HOSTNAME env vars
# it sets are our fallback signal so a future Cloud deploy that skips
# the on-disk mount still routes through st.secrets.
_HAS_SECRETS_FILE = any(p.exists() for p in _SECRETS_PATHS) or (
    "STREAMLIT_RUNTIME_ENV" in os.environ
    or os.environ.get("HOSTNAME", "").startswith(("streamlit", "appuser"))
)


def _get_secret(key: str) -> str | None:
    """Read a secret from Streamlit Secrets first, then the environment.

    Only touches st.secrets when there's evidence we're on Streamlit Cloud
    (or a secrets.toml is present locally) — otherwise pure local dev via
    `.env` triggers a noisy Streamlit warning on every rerun even though
    the env-var fallback works.
    """
    if _HAS_SECRETS_FILE:
        try:
            value = st.secrets.get(key)  # type: ignore[union-attr]
            if value:
                return value
        except Exception:
            pass
    return os.getenv(key)


OPENROUTER_API_KEY = _get_secret("OPENROUTER_API_KEY")
OPENAI_API_KEY = _get_secret("OPEN_AI_EMBEDDINGS_API_KEY")
QDRANT_URL = _get_secret("QDRANT_URL")
QDRANT_API_KEY = _get_secret("QDRANT_API_KEY")


def _require(name: str, value: Any) -> None:
    """Fail loudly at app boot if a required secret is missing.

    Better to surface a clear error in the Streamlit UI than to let a
    downstream client crash with a cryptic auth error.
    """
    if not value:
        st.error(
            f"Missing required secret `{name}`.\n\n"
            "- **Local dev**: add it to the repo-root `.env` file "
            "(see `.streamlit/secrets.toml.example` for the full key list).\n"
            "- **Streamlit Community Cloud**: add it under "
            "**App settings -> Secrets** in the dashboard "
            "(https://share.streamlit.io)."
        )
        st.stop()


_require("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
_require("OPEN_AI_EMBEDDINGS_API_KEY", OPENAI_API_KEY)
_require("QDRANT_URL", QDRANT_URL)
_require("QDRANT_API_KEY", QDRANT_API_KEY)


@st.cache_resource
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


@st.cache_resource
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL_ID, api_key=OPENAI_API_KEY)


# Table used by _normalise_for_dedup to collapse HTML smart-quotes and
# a few other typographic variants back to their ASCII originals so that
# .qmd source and its rendered .html output produce the same dedup key.
_QUOTE_NORMALISE = str.maketrans({
    "‘": "'",  # left single quote
    "’": "'",  # right single quote / apostrophe
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "–": "-",  # en dash
    "—": "-",  # em dash
    "…": "...",  # ellipsis
    " ": " ",  # non-breaking space
})


def _normalise_for_dedup(text: str) -> str:
    """Return a whitespace-collapsed, quote-normalised, lowercase form of
    `text` used only as a de-dup key. Not for display."""
    return "".join(text.translate(_QUOTE_NORMALISE).split()).lower()


def _detect_session_number(query: str) -> int | None:
    """Return the session/week/lecture number mentioned in `query`, if any.

    Recognises the four aliases used in course conversation:
    'session N', 'week N', 'lecture N', 'wk N'. Case-insensitive. Returns
    the numeric session (1-13 for M731) or None if no match. If the query
    mentions more than one session number (rare — e.g. 'what changes
    between session 1 and session 2?'), we return the FIRST one; both will
    still surface via the normal hybrid search branch, and the pinned
    branch narrows the first one so at least it is guaranteed present.
    """
    m = SESSION_QUERY_RE.search(query)
    if not m:
        return None
    try:
        n = int(m.group(1))
    except ValueError:
        return None
    # M731 has 13 sessions. Don't over-match on incidental numbers like
    # "session 99".
    if 1 <= n <= 13:
        return n
    return None


def _session_source_filenames(session_number: int) -> list[str]:
    """Return the four `source_file` payload values a given session can
    have been indexed under: the primary `.qmd` and its rendered `.html`,
    plus the corresponding `-extended` variants where present. Zero-pads
    to two digits to match the on-disk filenames (`session-01`, not
    `session-1`)."""
    stem = f"session-{session_number:02d}"
    return [
        f"{stem}.qmd",
        f"{stem}.html",
        f"{stem}-extended.qmd",
        f"{stem}-extended.html",
    ]


def _fetch_session_chunks(
    session_number: int,
    query_vector: list[float],
    top_n: int,
) -> list[Document]:
    """Pull `top_n` chunks from the given session's source files, ranked
    by dense cosine against `query_vector`.

    Two-step approach because Qdrant's `source_file` payload field is not
    indexed for text/substring lookup (only exact `MatchAny` works), and
    we want cosine ranking:
      1. Scroll ALL points where `source_file` matches one of the
         session's filenames. Session decks are ~130 chunks max, so this
         is a single small request.
      2. Compute cosine similarity client-side against the query vector,
         sort desc, take top_n.

    Client-side ranking (rather than a filtered Qdrant `query_points`)
    keeps this dependency-free and behaves identically to the diagnostic
    that established this fix works.
    """
    filenames = _session_source_filenames(session_number)
    client = get_qdrant_client()

    session_points: list[Any] = []
    next_offset = None
    while True:
        page, next_offset = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="source_file",
                        match=MatchAny(any=filenames),
                    )
                ]
            ),
            with_payload=True,
            with_vectors=[DENSE_VECTOR_NAME],
            limit=256,
            offset=next_offset,
        )
        session_points.extend(page)
        if next_offset is None:
            break

    if not session_points:
        return []

    # Client-side cosine. Qdrant stores L2-normalised vectors for the
    # Cosine metric, but we compute the full formula so this stays
    # correct if the storage assumption ever changes.
    def _cos(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
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

    scored: list[tuple[float, Any]] = []
    for p in session_points:
        vec = p.vector.get(DENSE_VECTOR_NAME) if isinstance(p.vector, dict) else None
        if vec is None:
            continue
        scored.append((_cos(query_vector, vec), p))
    scored.sort(key=lambda t: t[0], reverse=True)

    docs: list[Document] = []
    for score, p in scored[:top_n]:
        payload = p.payload or {}
        docs.append(Document(
            page_content=payload.get("text", ""),
            metadata={
                "source_file": payload.get("source_file", "Unknown"),
                "context_header": payload.get("context_header", ""),
                "page_number": payload.get("page_number"),
                "element_category": payload.get("element_category", ""),
                "score": float(score),
            },
        ))
    return docs


def _fetch_term_pinned_chunks(probe_query: str, top_n: int) -> list[Document]:
    """Run a dedicated dense search for `probe_query` and return the top
    `top_n` chunks (max 1 per source file) as Documents to be pinned.

    Used by TERM_PINNING when a jargon term would otherwise lose the cosine
    race to a competing high-frequency term in the student's question (e.g.
    "cluster analysis" vs "factor analysis" in a comparison query). A direct
    probe guarantees the concept is represented regardless of how many RRF
    branches favour the competing term.
    """
    probe_vec = get_embeddings().embed_query(probe_query)
    client = get_qdrant_client()
    try:
        hits = client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=probe_vec,
            using=DENSE_VECTOR_NAME,
            limit=top_n * 6,
            with_payload=True,
        ).points
    except Exception:
        return []

    seen_sources: dict[str, int] = {}
    docs: list[Document] = []
    for pt in hits:
        pl = pt.payload or {}
        src = pl.get("source_file", "Unknown")
        if seen_sources.get(src, 0) >= 1:
            continue
        seen_sources[src] = 1
        docs.append(Document(
            page_content=pl.get("text", ""),
            metadata={
                "source_file": src,
                "context_header": pl.get("context_header", ""),
                "page_number": pl.get("page_number"),
                "element_category": pl.get("element_category", ""),
                "doc_type": pl.get("doc_type"),
                "score": float(pt.score),
            },
        ))
        if len(docs) >= top_n:
            break
    return docs


# ---------------------------------------------------------------------------
# Query understanding: conversational rewrite + multi-query expansion.
# ---------------------------------------------------------------------------


def _looks_self_contained(query: str) -> bool:
    """Cheap heuristic: is the query understandable without earlier turns?

    A query is considered self-contained if:
      - It contains no coreferential pronouns/anaphoric tokens (see
        _COREF_TOKENS), OR
      - It is long enough (>= 8 tokens) that any pronouns are almost
        certainly disambiguated by the query text itself
        (e.g. "What is the difference between construct validity and
        content validity?" contains 'the' but is self-contained).
    """
    tokens = re.findall(r"\w+", query)
    if len(tokens) >= 8:
        return True
    return _COREF_TOKENS.search(query) is None


def rewrite_query(query: str, history: list[dict[str, str]]) -> str:
    """Rewrite a follow-up question into a self-contained search query.

    We only make the extra LLM call when both conditions hold:
      1. There *is* prior conversation to draw from.
      2. The current query looks like it references earlier context
         (see `_looks_self_contained`).

    Falls back to the original query on any LLM error — a failed rewrite
    is much better than blocking retrieval on a network hiccup.

    Uses `temperature=0` because rewriting is a deterministic
    transformation: given the same history and follow-up, we want the same
    canonical form every time.
    """
    if not history:
        return query
    if _looks_self_contained(query):
        return query

    # Include only the last 3 turns of the history for cost / latency.
    tail = history[-3:]
    convo = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in tail
    )

    system = (
        "You are a query rewriter. Given conversation history and a "
        "follow-up question, rewrite the follow-up as a fully self-"
        "contained search query. Return ONLY the rewritten query, no "
        "explanation."
    )
    user_msg = f"Conversation:\n{convo}\n\nFollow-up: {query}\n\nRewritten query:"

    try:
        rewriter = get_llm().bind(temperature=0)
        resp = rewriter.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ])
        rewritten = (getattr(resp, "content", "") or "").strip()
        # Strip surrounding quotes the model sometimes adds.
        rewritten = rewritten.strip('"').strip("'").strip()
        # Sanity: if the model returned nothing, keep the original.
        if not rewritten:
            return query
        return rewritten
    except Exception:
        return query


def generate_query_variants(query: str) -> list[str]:
    """Return `[original] + up to MULTI_QUERY_VARIANTS alt phrasings`.

    Multi-query RRF is a well-known recall win: paraphrases catch chunks
    whose lexical form differs from the original query but whose meaning
    matches (e.g. "sample size for surveys" vs. "how many respondents do
    I need for a survey"). We fuse per-variant Qdrant results via RRF.

    Gracefully returns `[query]` on any LLM failure so retrieval never
    breaks on a network error.
    """
    system = (
        "Generate 3 alternative phrasings of this search query for a "
        "university marketing research course. Return ONLY the 3 queries, "
        "one per line, no numbering."
    )
    try:
        rewriter = get_llm().bind(temperature=0.3)
        resp = rewriter.invoke([
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ])
        raw = (getattr(resp, "content", "") or "").strip()
        variants: list[str] = []
        for line in raw.splitlines():
            cleaned = re.sub(r"^\s*[-*\d.)\s]+", "", line).strip().strip('"').strip("'")
            if cleaned and cleaned.lower() != query.lower():
                variants.append(cleaned)
        variants = variants[:MULTI_QUERY_VARIANTS]
        return [query] + variants
    except Exception:
        return [query]


def _reciprocal_rank_fusion(
    ranked_lists: list[list[Any]],
    k: int = 60,
) -> list[tuple[Any, float]]:
    """Standard RRF: score(doc) = sum(1 / (k + rank_in_list_i))
    across all lists in which the doc appears.

    We identify documents by (source_file, chunk_index) — matches the
    same key used elsewhere for dedup so a doc that shows up under
    multiple variants is fused correctly.

    Returns a list of (representative_point, score) sorted by score desc.
    """
    from collections import defaultdict

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


def _detect_doc_type_filter(query: str) -> str | None:
    """Return "syllabus_outline" for admin queries, else None.

    Admin queries (midterm date, office hours, deadlines) should be answered
    from the course outline / syllabus, not from a textbook chapter that
    happens to mention the word "midterm" in an example.

    Returning None means "no filter" — search all doc_types.
    """
    if ADMIN_QUERY_RE.search(query):
        return "syllabus_outline"
    return None


def search_docs(
    query: str,
    doc_type_filter: str | None = None,
) -> list[Document]:
    """Hybrid dense + sparse retrieval, multi-query expansion, RRF fused.

    Pipeline:
      1. Generate query variants (original + N paraphrases) via the LLM.
      2. Batch-embed all variants in one OpenAI call.
      3. For each variant, Prefetch both dense and sparse branches with
         RETRIEVER_FETCH_K limit. Optionally filter by doc_type.
      4. Fuse *all* branch result lists with RRF (client-side, so we can
         span variants).
      5. Merge session-pinned chunks up front (unconditional).
      6. Dedup on (source_file, chunk_index) + normalised-text prefix
         (belt-and-braces against session-05.qmd/.html twin duplicates).
      7. Cap per-source-file at MAX_HITS_PER_SOURCE so one textbook
         chapter can't monopolise the context window.
      8. Return the top RETRIEVER_K survivors.

    Note: the reranker (cross-encoder) runs AFTER this function returns —
    it's called from the chat loop, not here, so search_docs stays a
    pure retrieval primitive that eval scripts can call independently.

    Uses raw qdrant_client (not LangChain's QdrantVectorStore) because
    the ingest script writes payload at the top level, not nested under
    'metadata', and because LangChain's Qdrant wrapper does not expose
    the hybrid Query API.
    """
    # Build the doc_type filter once; every Prefetch below reuses it.
    qdrant_filter: Filter | None = None
    if doc_type_filter:
        qdrant_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_type",
                    match=MatchValue(value=doc_type_filter),
                )
            ]
        )

    # ---- Query expansion + batched embedding ---------------------------
    variants = generate_query_variants(query)
    # Embed all variants in one API call. embed_documents preserves order.
    variant_vectors = get_embeddings().embed_documents(variants)
    # The 'primary' query vector — used for session-pin ranking below — is
    # the first embedding, i.e. the original (post-rewrite) query.
    query_vector = variant_vectors[0]
    variant_sparse = [encode_sparse_query(v) for v in variants]

    # Session-aware augmentation. If the student asks about a specific
    # session ("what are the topics in session 1?", "week 4 readings?"),
    # pin the top-N chunks from that session's slide decks at the front
    # of the result list. Without this, the actual session-N slides can
    # lose the cosine race to README.md's session table or session-M's
    # opener chunk if it uses the word "topics" more literally — see
    # scripts/debug_session1.py for the smoking-gun diagnostic.
    session_n = _detect_session_number(query)
    pinned_docs: list[Document] = []
    if session_n is not None:
        pinned_docs = _fetch_session_chunks(
            session_n, query_vector, top_n=SESSION_PIN_TOP_N
        )

    # Jargon term pinning. When the query explicitly names a concept that
    # loses the dense cosine race to a competing term (e.g. "cluster
    # analysis" vs "factor analysis" in a contrast query), run a dedicated
    # probe search and prepend those chunks alongside any session-pinned ones.
    for term_pattern, probe, pin_n in TERM_PINNING:
        if term_pattern.search(query):
            pinned_docs.extend(_fetch_term_pinned_chunks(probe, pin_n))

    # ---- Per-variant Prefetch and manual RRF ---------------------------
    # We run one dense + one sparse Prefetch per variant. Each individual
    # Prefetch is scored server-side, then we pull *unfused* per-branch
    # results (via a light-weight query per branch) and RRF them
    # client-side across ALL branches. This gives us multi-query recall
    # gains that Qdrant's single-request Fusion.RRF can't reach on its
    # own (it fuses only within one query_points call).
    client = get_qdrant_client()
    branch_results: list[list[Any]] = []

    for variant_vec, (sp_indices, sp_values) in zip(
        variant_vectors, variant_sparse, strict=True
    ):
        # Dense branch: cosine against text-embedding-3-small vectors.
        try:
            dense_resp = client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=variant_vec,
                using=DENSE_VECTOR_NAME,
                limit=RETRIEVER_FETCH_K,
                with_payload=True,
                query_filter=qdrant_filter,
            )
            branch_results.append(list(dense_resp.points))
        except Exception:
            # A single-branch failure shouldn't take down retrieval.
            pass

        # Sparse branch: skip when the tokenizer emitted nothing (query
        # was all stopwords or purely non-alphanumeric). Sending an empty
        # SparseVector is a Qdrant validation error.
        if sp_indices:
            try:
                sparse_resp = client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=SparseVector(indices=sp_indices, values=sp_values),
                    using=SPARSE_VECTOR_NAME,
                    limit=RETRIEVER_FETCH_K,
                    with_payload=True,
                    query_filter=qdrant_filter,
                )
                branch_results.append(list(sparse_resp.points))
            except Exception:
                pass

    # Fuse across every branch (dense-for-variant-1, sparse-for-variant-1,
    # dense-for-variant-2, ...). RRF is order-agnostic.
    fused = _reciprocal_rank_fusion(branch_results)
    results = [p for p, _score in fused]

    seen_chunks: set[tuple[str, Any]] = set()
    seen_text_prefix: set[str] = set()
    per_source_seen: dict[str, int] = {}
    kept: list[Document] = []

    # Merge pinned session chunks FIRST so they always land in the LLM's
    # context. They still go through the same dedup checks as fused hits
    # (so a pinned session-01.qmd chunk doesn't get re-added when the
    # hybrid branch also surfaces it) and count against per_source_seen
    # so the diversity cap continues to work — but crucially, they are
    # NOT subject to the cap themselves. When a student explicitly asks
    # about a session, up to SESSION_PIN_TOP_N chunks from that session
    # get through unconditionally.
    for pdoc in pinned_docs:
        src = pdoc.metadata.get("source_file", "Unknown")
        # Pinned docs don't carry a chunk_index in metadata; use a
        # separate sentinel so they never collide with real fused hits.
        chunk_key = (src, f"__pinned__{pdoc.page_content[:64]}")
        seen_chunks.add(chunk_key)
        text_key = _normalise_for_dedup(pdoc.page_content)[:200]
        if text_key:
            seen_text_prefix.add(text_key)
        per_source_seen[src] = per_source_seen.get(src, 0) + 1
        kept.append(pdoc)

    for point in results:
        payload = point.payload or {}
        source = payload.get("source_file", "Unknown")
        chunk_idx = payload.get("chunk_index")
        text = payload.get("text", "") or ""

        # Duplicate-chunk suppression. (source, chunk_index) is a stable
        # per-file identifier; if we see two hits with the same key, the
        # second is a re-ingest artifact and adds nothing.
        chunk_key = (source, chunk_idx)
        if chunk_key in seen_chunks:
            continue
        seen_chunks.add(chunk_key)

        # Cross-file text-identity check: catches session-NN.qmd vs
        # session-NN.html twins (same content indexed from Quarto source
        # AND its rendered HTML) and any other case where two payloads
        # happen to carry identical text under different filenames.
        # Normalisation removes: whitespace variance, curly-vs-straight
        # quote differences (HTML render smart-quotes them, source .qmd
        # keeps ASCII), and case. First 200 normalised chars is unique
        # enough for this corpus.
        text_key = _normalise_for_dedup(text)[:200]
        if text_key and text_key in seen_text_prefix:
            continue
        seen_text_prefix.add(text_key)

        # Diversity cap. Prevents "all 6 hits are from Chapter 11" pattern
        # where one well-covered topic monopolises the entire context window.
        if per_source_seen.get(source, 0) >= MAX_HITS_PER_SOURCE:
            continue
        per_source_seen[source] = per_source_seen.get(source, 0) + 1

        kept.append(Document(
            page_content=payload.get("text", ""),
            metadata={
                "source_file": source,
                "context_header": payload.get("context_header", ""),
                "page_number": payload.get("page_number"),
                "element_category": payload.get("element_category", ""),
                "doc_type": payload.get("doc_type"),
                "video_name": payload.get("video_name"),
                "start_seconds": payload.get("start_seconds"),
                "end_seconds": payload.get("end_seconds"),
                "start_time": payload.get("start_time"),
                "end_time": payload.get("end_time"),
                # point.score for fused results is a raw RRF-branch score,
                # which is not comparable across queries; kept for debugging.
                "score": getattr(point, "score", None),
            },
        ))
        if len(kept) >= RETRIEVER_K:
            break

    return kept


@st.cache_resource
def get_llm() -> ChatOpenAI:
    """Chat model = DeepSeek-V3 via OpenRouter (OpenAI-API-compatible).

    Cached so the underlying HTTP client is reused instead of rebuilt
    each rerun.
    """
    return ChatOpenAI(
        model=OPENROUTER_CHAT_MODEL_ID,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.2,
    )


@st.cache_resource
def get_reranker():  # type: ignore[no-untyped-def]
    """Load the cross-encoder reranker on first use, cache thereafter.

    `cross-encoder/ms-marco-MiniLM-L-6-v2` is ~90 MB, runs comfortably on
    CPU, and scores (query, passage) pairs on their true topical
    relevance — a step up from bi-encoder cosine similarity because it
    reads both texts jointly instead of encoding them independently.

    Import is lazy so the app boots even if sentence-transformers is not
    installed (retrieval still works, we just skip reranking).
    """
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANKER_MODEL_ID)


def rerank_docs(query: str, docs: list[Document]) -> list[Document]:
    """Rerank retrieved docs by cross-encoder relevance to `query`.

    We over-fetch a big candidate pool from Qdrant (RETRIEVER_FETCH_K),
    let the cross-encoder score each (query, passage) pair, and return
    the top RETRIEVER_K by score. If sentence-transformers isn't
    installed or scoring fails, we degrade to the original order so
    retrieval never breaks on a reranker outage.
    """
    if not docs:
        return docs
    try:
        reranker = get_reranker()
        # Prepend context_header so the cross-encoder sees the LLM-generated
        # topic summary alongside the raw chunk text. Without this, chunks
        # whose text is mostly R output (numbers/code) score poorly even when
        # the context_header correctly names the relevant concept.
        pairs = [
            (
                query,
                (
                    (d.metadata.get("context_header") or "") + " " + d.page_content
                ).strip(),
            )
            for d in docs
        ]
        scores = reranker.predict(pairs)
        scored = sorted(zip(scores, docs), key=lambda t: float(t[0]), reverse=True)
        top = [d for _s, d in scored[:RETRIEVER_K]]
        # Stash the rerank score in metadata for debugging.
        for (s, _), d in zip(scored[:RETRIEVER_K], top):
            d.metadata["rerank_score"] = float(s)
        return top
    except Exception:
        return docs[:RETRIEVER_K]


# --- Citation post-processor (P3a) -----------------------------------------
# Matches inline [N] citations (single or multiple like [1][3]) so we can
# style them consistently in the Streamlit render.
_CITATION_RE = re.compile(r"\[(\d+)\]")


def _render_citations(text: str, citation_map: dict[int, str]) -> str:
    """Emphasise inline [N] citations for the Streamlit renderer.

    `citation_map` is `{ passage_number: human_readable_source }`. We use
    it to wrap each `[N]` in **bold** — Streamlit renders that as visible
    tags without producing a wall of colour, and the mapping is a
    stepping-stone for future features (tooltip hover, click-to-scroll).
    """

    def _sub(m: re.Match[str]) -> str:
        n = int(m.group(1))
        if n in citation_map:
            return f"**[{n}]**"
        return m.group(0)

    return _CITATION_RE.sub(_sub, text)


llm = get_llm()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def _format_timestamp(seconds: int | float | None) -> str:
    """Format a second offset as MM:SS or H:MM:SS. Used in citations
    for video-transcript chunks so students can jump to the exact
    moment in the lecture recording. Returns "?" for missing/negative
    inputs so the citation still renders instead of crashing."""
    if seconds is None:
        return "?"
    try:
        s = int(float(seconds))
    except (TypeError, ValueError):
        return "?"
    if s < 0:
        return "?"
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def _format_citation(doc: Document) -> str:
    """Human-readable source pointer.

    Three shapes, picked in order:
      1. Video transcript: `source_file (MM:SS-MM:SS)` (P3b) — the
         transcript filename disambiguates recordings with duplicate
         labels, and the timestamp span lets students jump to the exact
         spot in the video. Written to match the payload written by
         scripts/ingest_video.py (start_time / end_time strings).
      2. Paged source (PDFs): `filename (p. 42)`.
      3. Everything else (slide decks parsed from .qmd/.html): just the
         filename.
    """
    if doc.metadata.get("doc_type") == "video_transcript":
        source_file = doc.metadata.get("source_file") or doc.metadata.get("video_name") or "video"
        # Prefer the pre-formatted start_time / end_time strings written
        # by ingest_video.py; fall back to computing from *_seconds so
        # older/legacy payloads still render.
        start = doc.metadata.get("start_time") or _format_timestamp(doc.metadata.get("start_seconds"))
        end = doc.metadata.get("end_time") or _format_timestamp(doc.metadata.get("end_seconds"))
        # Use an en-dash between times so the range visually distinguishes
        # from the "-" in filenames like "session-01.qmd".
        return f"{source_file} ({start}–{end})"

    src = doc.metadata.get("source_file", "Unknown")
    page = doc.metadata.get("page_number")
    if page is not None:
        return f"{src} (p. {page})"
    return src


def _build_context_block(docs: list[Document]) -> str:
    """Format retrieved chunks as a numbered context block. The [n] tags
    make it easy for the LLM to cite specific passages inline; the
    parenthetical source header gives it the filename/page it should
    quote when a student asks 'where did that come from?'.

    The LLM-generated context_header (one-sentence topic summary) is
    prepended to the raw chunk text so the model can orient itself even
    when the chunk text is code or tabular output that wouldn't be
    self-explanatory on its own."""
    lines: list[str] = []
    for i, d in enumerate(docs, start=1):
        header = _format_citation(d)
        ctx_hdr = (d.metadata.get("context_header") or "").strip()
        body = d.page_content.strip()
        text = f"{ctx_hdr}\n{body}" if ctx_hdr else body
        lines.append(f"[{i}] Source: {header}\n{text}")
    return "\n\n".join(lines)


def _history_messages(turns: int) -> list[dict[str, str]]:
    """Return the last `turns` user+assistant pairs from session state,
    trimmed to fit and formatted for the OpenAI chat schema.

    Why include history: without it, a follow-up like 'can you give an
    example?' has no referent because each request is answered in
    isolation from the vector store. Passing the last few turns lets
    the model treat the current question as a continuation."""
    if turns <= 0:
        return []
    hist = st.session_state.get("messages", [])
    # Take the last (turns * 2) messages; slicing tolerates odd counts.
    tail = hist[-(turns * 2):]
    return [{"role": m["role"], "content": m["content"]} for m in tail]


if prompt := st.chat_input("Ask a question about the course material..."):
    # Persist the ORIGINAL prompt in history/display — students see and can
    # refer back to what they actually typed, not the rewritten form.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Step 1: conversational rewrite. If the query is a follow-up
            # ("give me an example of it"), rewrite it to a stand-alone
            # search query using the last few turns of conversation.
            # History passed here EXCLUDES the current prompt because the
            # rewriter needs *prior* turns, not the message it's rewriting.
            prior_hist = st.session_state.messages[:-1]
            rewritten = rewrite_query(prompt, prior_hist)

            # Step 2: doc-type routing. Admin queries (midterm date, office
            # hours) get filtered to the syllabus/outline; everything else
            # searches all doc_types.
            doc_type = _detect_doc_type_filter(rewritten)

            # Step 3: hybrid + multi-query retrieval.
            docs = search_docs(rewritten, doc_type_filter=doc_type)

            # Step 4: cross-encoder rerank (P2.3). We over-fetched to give
            # the reranker a wider pool; it picks the best RETRIEVER_K.
            docs = rerank_docs(rewritten, docs)

            if not docs:
                # No relevant retrieval — tell the student rather than let
                # the model hallucinate from an empty context.
                msg = (
                    "I could not find anything relevant to that question "
                    "in the course materials I have indexed. Try rephrasing, "
                    "or ask about a topic covered in the lecture slides or "
                    "textbook readings."
                )
                st.markdown(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            else:
                context = _build_context_block(docs)

                # Passage number -> human-readable source. Used both for
                # the Referenced Material line and to validate that model
                # [N] citations correspond to real passages.
                citation_map: dict[int, str] = {}
                for i, d in enumerate(docs, start=1):
                    citation_map[i] = _format_citation(d)

                # De-dup citations while preserving retrieval order so the
                # "Referenced Material" line reads top-hit-first.
                seen: set[str] = set()
                citations: list[str] = []
                for d in docs:
                    c = _format_citation(d)
                    if c not in seen:
                        seen.add(c)
                        citations.append(c)

                # System prompt (P1.3 + P3a): grounding + refusal +
                # sub-sentence citation placement.
                system_prompt = (
                    f"You are a helpful teaching assistant for {COURSE_NAME}. "
                    "Answer the student's question using ONLY the numbered "
                    "context passages below.\n\n"
                    "Grounding rules:\n"
                    "1. Ground every factual claim by citing the supporting "
                    "passage inline with [N] where N is the passage number. "
                    "Place the [N] IMMEDIATELY after the specific claim it "
                    "supports, not just at the end of the sentence. Example: "
                    "\"The typical response rate for mail surveys is 10-15% [1], "
                    "compared to 30-40% for telephone interviews [2].\"\n"
                    "2. If multiple passages support the same claim, cite "
                    "them together: [1][3].\n"
                    "3. If the question cannot be answered from the "
                    "retrieved passages, you MUST respond with EXACTLY this "
                    "form: \"Based on the provided course materials, I "
                    "cannot find information about this. [General knowledge, "
                    "not from course materials:] ...\" — then give your best "
                    "general-knowledge answer. Never fabricate a citation.\n"
                    "4. Prefer the specific concept, framework, or method "
                    "the course material references (e.g. \"Cronbach's alpha\", "
                    "\"stratified sampling\", \"conjoint analysis\") over "
                    "generic phrasing.\n\n"
                    "The conversation history above is provided so that "
                    "follow-up questions ('give me an example', 'why?') can "
                    "be interpreted correctly.\n\n"
                    f"Context:\n{context}"
                )

                # LLM message order: system prompt (with retrieval context)
                # -> prior conversation turns -> the new user question.
                messages: list[dict[str, str]] = [
                    {"role": "system", "content": system_prompt},
                ]
                # The just-appended user turn is already in session_state,
                # so history slice includes it. Drop the last message from
                # history to avoid sending the current prompt twice.
                hist = _history_messages(CHAT_HISTORY_TURNS)
                if hist and hist[-1]["role"] == "user" and hist[-1]["content"] == prompt:
                    hist = hist[:-1]
                messages.extend(hist)
                messages.append({"role": "user", "content": prompt})

                # Stream the reply, collect the full text so we can post-
                # process citations before persisting to history.
                raw_reply = st.write_stream(
                    chunk.content for chunk in llm.stream(messages)
                )
                reply_content = _render_citations(raw_reply, citation_map)

                if citations:
                    attribution = f"\n\n*Referenced Material: {', '.join(citations)}*"
                    st.markdown(attribution)
                    reply_content += attribution

                st.session_state.messages.append({"role": "assistant", "content": reply_content})

        except Exception as e:
            # Surface the real error rather than swallowing it — makes it
            # obvious whether the failure was Qdrant, OpenAI, or OpenRouter.
            st.error(f"Could not generate a response: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"[Error: {e}]"})
