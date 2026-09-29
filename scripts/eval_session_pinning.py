"""Retrieval eval for session-specific questions (session pinning).

Runs each question through the app's real retrieval path
(search_docs -> rerank_docs, i.e. what the student sees as sources) and
checks whether the expected deck appears in the final top-K. Covers the
per-session companion decks (R Lab, solutions) plus main-deck questions as
regression guards.

Query variants are generated at temperature 0.3, so results are slightly
stochastic — use --repeats to average over runs.

Usage:
    .venv\\Scripts\\python.exe -m scripts.eval_session_pinning [--repeats N]
"""

from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

# Stub Streamlit before importing app (same shim as scripts.smoke_e2e).
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
fake_st.cache_resource = lambda f: f  # type: ignore[attr-defined]


class _Secrets:
    def get(self, k, default=None): return None


fake_st.secrets = _Secrets()  # type: ignore[attr-defined]
sys.modules["streamlit"] = fake_st

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

from app import rerank_docs, search_docs  # noqa: E402

# (question, expected deck stem). A hit is any final source whose filename
# stem equals the expected stem (so .qmd and .html both count, but
# session-01 does NOT match session-01-RLab).
CASES: list[tuple[str, str]] = [
    # Companion decks — the failure this eval was written for.
    ("What do we do in the session 1 R lab?", "session-01-RLab"),
    ("What R code do we run in the week 2 lab?", "session-02-RLab"),
    ("session 3 R lab exercises", "session-03-RLab"),
    ("Walk me through the session 3 lab on power calculations in R", "session-03-RLab"),
    ("Where are the solutions for session 5?", "solutions-05"),
    ("What is the answer to the session 8 exercise?", "solutions-08"),
    ("Show me the solutions for week 10", "solutions-10"),
    # Regression guards — main deck must still surface.
    ("What are the topics in session 1?", "session-01"),
    ("What did we cover in lecture 4?", "session-04"),
    ("Summarise session 7", "session-07"),
    ("What are the learning objectives for week 12?", "session-12"),
    ("What does session 2 say about research design?", "session-02"),
]


def run_case(question: str, expected: str) -> tuple[int | None, list[str]]:
    """Return (1-based rank of first expected-deck hit or None, final stems)."""
    docs = rerank_docs(question, search_docs(question))
    stems = [Path(d.metadata.get("source_file", "")).stem for d in docs]
    rank = next((i + 1 for i, s in enumerate(stems) if s == expected), None)
    return rank, stems


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()

    hits = 0
    total = 0
    for question, expected in CASES:
        ranks = []
        for _ in range(args.repeats):
            rank, stems = run_case(question, expected)
            ranks.append(rank)
            hits += rank is not None
            total += 1
        shown = ",".join("-" if r is None else str(r) for r in ranks)
        print(f"[{shown:>7}]  {expected:<16}  {question}")
        print(f"            last top-K: {stems}")
    print(f"\nhit@K: {hits}/{total} = {hits / total:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
