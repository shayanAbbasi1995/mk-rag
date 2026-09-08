"""Run the same eval queries through the *new* app.py search_docs()
so we see the practical effect of over-fetch + dedupe + threshold +
per-source cap. Reports the same summary stats as scripts.eval_retrieval
but on the post-filter output rather than the raw Qdrant hits.
"""

from __future__ import annotations

import os
import sys
from statistics import mean, pstdev

from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
except Exception:  # noqa: BLE001
    pass

load_dotenv()

# Import the retrieval helper from app.py. app.py touches st.set_page_config
# at module import time; stub streamlit so it doesn't blow up when we run
# outside Streamlit.
import types

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
    def __getattr__(self, k):
        return self[k]
    def __setattr__(self, k, v):
        self[k] = v
fake_st.session_state = _SessionState(messages=[])  # type: ignore[attr-defined]

def _cache_resource(f):  # streamlit.cache_resource shim
    return f
fake_st.cache_resource = _cache_resource  # type: ignore[attr-defined]

class _Secrets:
    def get(self, k, default=None):
        return None
fake_st.secrets = _Secrets()  # type: ignore[attr-defined]

sys.modules["streamlit"] = fake_st

from app import search_docs  # noqa: E402

QUERIES = [
    "What is conjoint analysis?",
    "How do you design a survey questionnaire?",
    "What is the difference between qualitative and quantitative research?",
    "Explain focus groups in marketing research",
    "What are the steps in the marketing research process?",
    "What is a Likert scale?",
    "Explain NPS (Net Promoter Score)",
    "How does ANOVA work in marketing research?",
    "How do I show where brands sit relative to each other on a chart?",
    "What topics are covered in the course?",
    # Deliberately off-topic — should get filtered by SCORE_FLOOR.
    "What is the airspeed velocity of an unladen swallow?",
]


def main() -> int:
    for q in QUERIES:
        docs = search_docs(q)
        scores = [d.metadata.get("score") for d in docs if d.metadata.get("score") is not None]
        sources = [d.metadata.get("source_file") for d in docs]
        print("=" * 100)
        print(f"Q: {q}")
        print(f"   returned {len(docs)} chunk(s); unique sources: {len(set(sources))}")
        if scores:
            print(
                f"   score max={max(scores):.3f}  min={min(scores):.3f}  "
                f"mean={mean(scores):.3f}  stdev={pstdev(scores):.3f}"
            )
        for i, d in enumerate(docs, start=1):
            snip = (d.page_content or "").replace("\n", " ")[:140]
            print(
                f"  {i}. [{d.metadata.get('score', 0):.3f}] "
                f"{d.metadata.get('source_file')} "
                f"p={d.metadata.get('page_number')}"
            )
            print(f"     {snip}...")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
