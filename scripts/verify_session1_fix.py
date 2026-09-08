"""Verify the session-aware retrieval fix.

Imports the real `search_docs` from ``app.py`` (bypassing the Streamlit
runtime by monkey-patching ``st.cache_resource`` and ``st.set_page_config``
before import) and runs it against three queries:

1. "What are the topics taught in session 1?" — the original bug report.
2. "What are the topics taught in session 4?" — different session.
3. "Explain Cronbach's alpha" — no session number; must not regress.

For each, prints the final kept documents' source_file + score + text
preview and asserts that session-N chunks are present when a session
number was in the query.
"""

from __future__ import annotations

import io
import os
import pathlib
import sys
from typing import Any

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure env is loaded before app.py imports anything that needs it.
from dotenv import load_dotenv  # noqa: E402

load_dotenv(override=False)

# app.py calls st.set_page_config at import time, which needs a Streamlit
# script-run context. Monkey-patch it (and st.cache_resource, which app.py
# uses to memoise the qdrant client and embeddings) so we can import
# search_docs from a plain Python script.
import streamlit as st  # noqa: E402


def _noop(*args: Any, **kwargs: Any) -> None:  # noqa: D401
    return None


st.set_page_config = _noop  # type: ignore[assignment]
st.title = _noop  # type: ignore[assignment]


def _cache_resource_passthrough(func):  # type: ignore[no-untyped-def]
    """Replacement for st.cache_resource that just returns the function.

    We don't need caching for a one-shot verification script, and the real
    cache decorator requires a live Streamlit runtime.
    """
    return func


st.cache_resource = _cache_resource_passthrough  # type: ignore[assignment]

# Now safe to import.
import app  # noqa: E402


def _preview(text: str, n: int = 160) -> str:
    if not text:
        return ""
    return " ".join(text.split())[:n]


def run(query: str, expect_session: int | None) -> None:
    print("=" * 78)
    print(f"QUERY: {query!r}    (expect session-{expect_session:02d} present)"
          if expect_session is not None
          else f"QUERY: {query!r}    (no session pin expected)")
    print("=" * 78)
    docs = app.search_docs(query)
    print(f"Returned {len(docs)} docs:")
    session_matches = 0
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source_file", "?")
        score = d.metadata.get("score")
        print(f"  #{i}  score={score:.4f}  {src}")
        print(f"       {_preview(d.page_content)}")
        if expect_session is not None and f"session-{expect_session:02d}" in src:
            session_matches += 1
    if expect_session is not None:
        status = "PASS" if session_matches >= 1 else "FAIL"
        print(f"\n  [{status}] session-{expect_session:02d} chunks present: {session_matches}")
    print()


if __name__ == "__main__":
    run("What are the topics taught in session 1?", expect_session=1)
    run("What are the topics taught in session 4?", expect_session=4)
    run("Explain Cronbach's alpha", expect_session=None)
