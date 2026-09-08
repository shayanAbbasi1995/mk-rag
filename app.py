"""Streamlit chat UI for the Marketing Research course RAG assistant.

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
RETRIEVER_FETCH_K: int = 12
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


st.set_page_config(page_title="M731 - Marketing Research: Ask a question", layout="centered")
st.title("M731 - Marketing Research: Ask a question")


_SECRETS_PATHS = [
    pathlib.Path.home() / ".streamlit" / "secrets.toml",
    pathlib.Path(__file__).parent / ".streamlit" / "secrets.toml",
]
_HAS_SECRETS_FILE = any(p.exists() for p in _SECRETS_PATHS)


def _get_secret(key: str) -> str | None:
    """Read a secret from Streamlit Secrets first, then the environment.

    Only touches st.secrets if a secrets.toml actually exists — otherwise
    Streamlit prints a noisy warning on every rerun even though we fall back
    to env vars successfully.
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
            f"Missing required secret `{name}`. Set it in the repo-root `.env` "
            "file, or in `.streamlit/secrets.toml` when deploying."
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
                "page_number": payload.get("page_number"),
                "element_category": payload.get("element_category", ""),
                "score": float(score),
            },
        ))
    return docs


def search_docs(query: str) -> list[Document]:
    """Hybrid dense + sparse retrieval fused with Reciprocal Rank Fusion.

    Pipeline:
      1. Embed the query (OpenAI text-embedding-3-small).
      2. Sparse-encode the query (in-process BM25-style hasher).
      3. Ask Qdrant for `RETRIEVER_FETCH_K` candidates from EACH branch:
         - dense: cosine similarity against the `dense` named vector.
         - sparse: dot-product against the `sparse` named vector, whose
           inverted index gives us exact-token overlap (great for jargon
           like "TURF", "Cronbach alpha", "conjoint").
      4. Fuse the two ranked lists with RRF (server-side, `Fusion.RRF`).
         The output is a single ranked list biased towards items that
         appear high in *both* lists — the standard hybrid-retrieval win.
      5. Collapse duplicate (source_file, chunk_index) hits — belt-and-
         braces against the historical duplication incident (session-05.qmd
         vs session-05.html covered the same content under different
         extensions).
      6. Cross-file text-identity check to catch any residual twins.
      7. Cap per-source-file at MAX_HITS_PER_SOURCE so a single textbook
         chapter doesn't monopolize the context window.
      8. Return the top RETRIEVER_K survivors in fused score order.

    Uses raw qdrant_client (not LangChain's QdrantVectorStore) because the
    ingest script writes metadata at the payload top level, not nested
    under a 'metadata' key, and because LangChain's Qdrant wrapper does
    not expose the hybrid Query API.
    """
    query_vector = get_embeddings().embed_query(query)
    sp_indices, sp_values = encode_sparse_query(query)

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

    # If the query has no encodable tokens (all stopwords, empty), skip
    # the sparse branch — sending an empty SparseVector would surface a
    # Qdrant validation error. Fall back to dense-only.
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

    response = get_qdrant_client().query_points(
        collection_name=QDRANT_COLLECTION,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=RETRIEVER_FETCH_K,
        with_payload=True,
    )
    results = response.points

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
                "page_number": payload.get("page_number"),
                "element_category": payload.get("element_category", ""),
                "score": point.score,
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


llm = get_llm()


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

def _format_citation(doc: Document) -> str:
    """Human-readable source pointer: 'filename (p. 42)' when a page
    number exists in the payload, otherwise just the filename. Slide
    decks parsed from .qmd/.html have no page concept, so we fall back
    gracefully."""
    src = doc.metadata.get("source_file", "Unknown")
    page = doc.metadata.get("page_number")
    if page is not None:
        return f"{src} (p. {page})"
    return src


def _build_context_block(docs: list[Document]) -> str:
    """Format retrieved chunks as a numbered context block. The [n] tags
    make it easy for the LLM to cite specific passages inline; the
    parenthetical source header gives it the filename/page it should
    quote when a student asks 'where did that come from?'."""
    lines: list[str] = []
    for i, d in enumerate(docs, start=1):
        header = _format_citation(d)
        lines.append(f"[{i}] Source: {header}\n{d.page_content.strip()}")
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            docs = search_docs(prompt)

            if not docs:
                # Everything was filtered by SCORE_FLOOR. Rather than let
                # the model hallucinate from an empty context, tell the
                # student explicitly.
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
                # De-dup citations while preserving retrieval order so the
                # "Referenced Material" line reads top-hit-first.
                seen: set[str] = set()
                citations: list[str] = []
                for d in docs:
                    c = _format_citation(d)
                    if c not in seen:
                        seen.add(c)
                        citations.append(c)

                system_prompt = (
                    "You are a helpful teaching assistant for a graduate-level "
                    "Marketing Research course. Answer the student's question "
                    "strictly using the numbered context passages below. If "
                    "the answer cannot be found in the context, say you do not "
                    "know based on the provided material. Do not fabricate "
                    "answers. When possible, ground your answer in the specific "
                    "research method, framework, or concept the course material "
                    "references, and reference passage numbers like [1], [2] "
                    "when appropriate.\n\n"
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

                reply_content = st.write_stream(
                    chunk.content for chunk in llm.stream(messages)
                )

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
