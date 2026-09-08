"""End-to-end smoke test: search_docs -> DeepSeek via OpenRouter.

Bypasses Streamlit but runs every other production path so we know the
LLM is grounded in real retrieved context and the credentials chain
still works. Prints the first ~400 chars of the reply plus the source
citations that would render under 'Referenced Material' in the UI.
"""

from __future__ import annotations

import sys
import types

# Stub Streamlit before importing app (see scripts.eval_after_improvements
# for the same shim).
fake_st = types.ModuleType("streamlit")
fake_st.set_page_config = lambda **kw: None  # type: ignore[attr-defined]
fake_st.title = lambda *a, **k: None  # type: ignore[attr-defined]
fake_st.error = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(a[0] if a else "st.error"))  # type: ignore[attr-defined]
fake_st.stop = lambda: (_ for _ in ()).throw(SystemExit(1))  # type: ignore[attr-defined]
fake_st.chat_input = lambda *a, **k: None  # type: ignore[attr-defined]
fake_st.chat_message = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no chat here"))  # type: ignore[attr-defined]
fake_st.markdown = lambda *a, **k: None  # type: ignore[attr-defined]
fake_st.write_stream = lambda *a, **k: ""  # type: ignore[attr-defined]

class _SessionState(dict):
    def __getattr__(self, k): return self[k]
    def __setattr__(self, k, v): self[k] = v
fake_st.session_state = _SessionState(messages=[])  # type: ignore[attr-defined]

def _cache_resource(f): return f
fake_st.cache_resource = _cache_resource  # type: ignore[attr-defined]

class _Secrets:
    def get(self, k, default=None): return None
fake_st.secrets = _Secrets()  # type: ignore[attr-defined]

sys.modules["streamlit"] = fake_st

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

from app import _build_context_block, _format_citation, llm, search_docs  # noqa: E402


def main() -> int:
    q = "What is conjoint analysis? Answer in 3 sentences."
    docs = search_docs(q)
    print(f"retrieved {len(docs)} chunks")
    ctx = _build_context_block(docs)
    system_prompt = (
        "You are a helpful teaching assistant for a Marketing Research course. "
        "Answer strictly using the numbered context. If not present, say so.\n\n"
        f"Context:\n{ctx}"
    )
    reply = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": q},
    ])
    print("--- reply ---")
    print(reply.content[:600])
    print("--- citations ---")
    for d in docs:
        print(" -", _format_citation(d))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
