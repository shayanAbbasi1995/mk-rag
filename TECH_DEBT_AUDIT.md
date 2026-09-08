# Tech Debt Audit — mk-rag
Generated: 2026-09-07

---

## Executive Summary

- 0 Critical, 2 High, 3 Medium, 4 Low findings
- This is a 2-file, ~200 LOC project in its first day of commits — the debt is light but the unpinned dependencies are the single biggest risk
- Largest debt concentration: `requirements.txt` (dependency hygiene) and `app.py` (error handling)
- mypy passes cleanly; both files compile without errors
- ruff not installed in the env — not run

---

## Architectural Mental Model

A two-component RAG system for a university marketing research course. `ingest.py` is a one-shot CLI that reads PDFs/Markdown from `docs/`, splits them into chunks, and upserts them into a Qdrant cloud vector store tracked by MD5 hash in `index_tracker.json`. `app.py` is a Streamlit chat UI that retrieves from that same collection and sends retrieved context to DeepSeek-V3 via OpenRouter. No shared modules, no orchestration layer, no tests. Exactly the right complexity level for a solo academic project.

The README says nothing because there is no README. That's the only thing that contradicts expectations for the architecture.

---

## Findings

| ID | Category | File:Line | Severity | Effort | Description | Recommendation |
|----|----------|-----------|----------|--------|-------------|----------------|
| F001 | Dependency debt | `requirements.txt:1-9` | High | S | All 9 packages are completely unpinned. LangChain breaks its own API between minor versions; one `pip install` on a new machine can produce a broken environment. | Run `pip freeze > requirements.txt` after confirming the current env works, then pin to exact versions. |
| F002 | Error handling | `app.py:52-71` | High | S | No try/except anywhere in the chat path. If Qdrant is unreachable or the LLM call times out, students see a raw Python traceback instead of a friendly error message. | Wrap the retriever and LLM calls in a try/except and surface failures with `st.error()`. |
| F003 | Config debt | `requirements.txt:2` | Medium | S | `langchain` meta-package is listed but may not be needed — the code imports only from `langchain-openai`, `langchain-qdrant`, and `langchain-community`. The meta-package pulls in a large dependency graph. | Test removing it; if nothing breaks on import, drop it. |
| F004 | Config debt | `.` (root) | Medium | S | `index_tracker.json` is tracked by git and changes on every `ingest.py` run. Every ingestion will dirty the working tree, creating noise in `git status` and polluting commit history. | Add `index_tracker.json` to `.gitignore`; let each developer's local env maintain their own tracker state. |
| F005 | Documentation | `.` (root) | Medium | S | No README. There are no instructions for setting up the env, running ingestion, or launching the app. The `.streamlit/secrets.toml.example` helps but is insufficient on its own. | Add a minimal README with: env setup, `pip install -r requirements.txt`, ingest step, `streamlit run app.py`. |
| F006 | Dependency debt | `ingest.py:5-6` | Low | S | `from langchain_community.document_loaders import PyPDFLoader, TextLoader` — `langchain_community` is the deprecated catch-all; these loaders are scheduled to migrate to `langchain-community` sub-packages. Not breaking today. | Track this; update when `langchain-community` emits deprecation warnings on these specific imports. |
| F007 | UX / performance | `app.py:63` | Low | M | `llm.invoke()` returns the full response at once. Streamlit supports streaming via `llm.stream()` + `st.write_stream()`, which gives a much better student-facing experience. | Replace `response = llm.invoke(...)` with `st.write_stream(llm.stream(...))` and remove the separate `st.markdown(reply_content)` call. Note: source attribution will need to be appended after the stream. |
| F008 | Type debt | `ingest.py:42,47,52,57,67` | Low | S | None of the five functions in `ingest.py` have type annotations. mypy passes only because of `--ignore-missing-imports`. `load_document()` in particular has an implicit `list[Any]` return type. | Add return type annotations to the public functions: `-> dict`, `-> None`, `-> list[Document]`. |

---

## Top 5 — Fix These First

### 1. F001 — Pin all dependencies

Run once in the working env:
```bash
pip freeze | grep -E "streamlit|langchain|qdrant|pypdf|dotenv|openai" > requirements.txt
```
Then commit. Without pins you cannot reproduce the environment.

### 2. F002 — Add error handling to the chat path (`app.py:50-71`)

```python
with st.chat_message("assistant"):
    try:
        docs = retriever.invoke(prompt)
        ...
        response = llm.invoke([...])
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
        st.stop()
```

### 3. F004 — Gitignore `index_tracker.json`

Add to `.gitignore`:
```
index_tracker.json
```
Delete it from tracking:
```bash
git rm --cached index_tracker.json
```
Commit a blank `index_tracker.json` template if you want a documented starting state, or let `ingest.py` create it on first run (it already handles the missing-file case at `ingest.py:47`).

### 4. F005 — Add a README

Bare minimum for a collaborator (or future-you) to get running:
- Python version, virtual env setup
- `pip install -r requirements.txt`
- Copy `.streamlit/secrets.toml.example` → `.streamlit/secrets.toml` and fill in keys
- Place PDFs in `docs/`, run `python ingest.py`
- `streamlit run app.py`

### 5. F003 — Test removing the `langchain` meta-package

The meta-package is probably pulling in transitive deps you don't need, and it's a common source of version conflicts. One command: `pip uninstall langchain && streamlit run app.py` — if nothing errors, drop it from `requirements.txt`.

---

## Quick Wins

- [ ] **F004**: `echo "index_tracker.json" >> .gitignore && git rm --cached index_tracker.json` — 2 commands, no code change
- [ ] **F001**: `pip freeze > requirements.txt` — 1 command
- [ ] **F008**: Add 5 type annotations to `ingest.py` functions — 5 lines changed
- [ ] **F005**: Write a 10-line README

---

## Things That Look Bad But Are Actually Fine

- **`retriever = get_retriever()` at module level** (`app.py:29`) — looks like it runs on every Streamlit rerun and hammers Qdrant. It doesn't. The `@st.cache_resource` decorator on `get_retriever()` makes this a singleton for the server process lifetime. This is the Streamlit-idiomatic pattern.

- **Dual `st.secrets.get() / os.getenv()` pattern** (`app.py:12-15`) — looks redundant. It's not; it correctly handles Streamlit Cloud (where secrets come from the dashboard, not a file) and local dev (where they come from `.env`). The pattern is correct.

- **`langchain-community` in requirements after removing the ChatOpenAI import** — at first glance looks like a stale dep. `ingest.py` still imports `PyPDFLoader` and `TextLoader` from it, so the dependency is still load-bearing. Do not remove.

- **No abstraction between retrieval and generation** — could look like a missing service layer. For 78 lines and one developer, inline is correct. Don't extract.

---

## Open Questions for the Maintainer

1. Is `index_tracker.json` intentionally committed so a collaborator can share ingestion state, or is it incidental? If intentional, ignore F004. If not, add it to `.gitignore`.

2. The `docs/` directory has only a `.gitkeep`. Are course PDFs being stored elsewhere (e.g., OneDrive) and manually copied in before ingestion? If so, the README should say so.

3. `ingest.py` has no way to remove documents from Qdrant when a file is deleted from `docs/`. If lecture notes get reorganized, stale chunks will persist in the vector store indefinitely. Is this a concern for the course timeline?
