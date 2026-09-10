# Course RAG Assistant

A low-cost, open-source template for building a **Retrieval-Augmented Generation (RAG) chatbot for any university course**. Students ask questions in plain English; the app retrieves the most relevant passages from your course materials and answers via DeepSeek-V3 through OpenRouter.

**Estimated cost: under $3 for a full semester with 50 students.**  
Drop in your own slides, textbooks, and readings — no ML infrastructure required.

---

## How it works

```
Student question
  ↓
Hybrid search (dense semantic + sparse BM25) over your course docs in Qdrant
  ↓
Top chunks → DeepSeek-V3 via OpenRouter
  ↓
Grounded answer with numbered source citations
```

Key features:
- **Hybrid retrieval** — combines OpenAI dense embeddings with BM25 sparse vectors via Reciprocal Rank Fusion (RRF), so both semantic meaning and exact terminology are matched
- **Contextual chunk headers** — each chunk is prefixed with its section breadcrumb (e.g. `[Chapter 4 > Survey Design]`) so it's self-contained when retrieved
- **Session/lecture pinning** — queries mentioning "session N / week N / lecture N" always surface content from that specific session
- **Conversation history** — last 10 turns included so follow-up questions ("give me an example") resolve correctly
- **Source citations** — every answer links to the source file and page number

---

## Stack & cost

| Component | Tool | Cost |
|-----------|------|------|
| LLM | DeepSeek-V3 via OpenRouter (`deepseek/deepseek-chat`) | ~$0.26/1M input tokens |
| Embeddings | OpenAI `text-embedding-3-small` | $0.02/1M tokens (~$0.04 per full corpus ingest) |
| Vector store | Qdrant Cloud free tier | $0 (1 GB perpetual free, ~500k vectors) |
| UI | Streamlit | $0 |

---

## Setup

### 1. Accounts (one-time)

| Service | Sign up | What you need |
|---------|---------|---------------|
| [OpenRouter](https://openrouter.ai) | Free | API key + $5 deposit |
| [OpenAI](https://platform.openai.com) | Free | API key + $5 credit (embeddings only) |
| [Qdrant Cloud](https://cloud.qdrant.io) | Free | Cluster URL + API key |

### 2. Clone and install

```powershell
git clone https://github.com/shayanAbbasi1995/mk-rag.git my-course-rag
cd my-course-rag
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Runtime deps only (what the deployed app needs):
pip install -r requirements.txt

# Add ingestion deps if you plan to run scripts/ingest.py locally
# (partitioners for PDF, PPTX, DOCX, HTML, Markdown, Quarto):
pip install -r requirements-ingest.txt
```

### 3. Configure your course

Create a `.env` file in the repo root (already gitignored):

```env
# Course branding — shown as the app title
COURSE_NAME=My Course Name: Ask a question

# OpenRouter (LLM generation)
OPENROUTER_API_KEY=sk-or-...

# OpenAI (embeddings only)
OPEN_AI_EMBEDDINGS_API_KEY=sk-...

# Qdrant Cloud
QDRANT_URL=https://<cluster-id>.cloud.qdrant.io:6333
QDRANT_API_KEY=...
QDRANT_CLUSTER_NAME=...
```

That's the only configuration needed. The app title, system prompt, and source citations all adapt to `COURSE_NAME` automatically.

### 4. Add your course materials

Copy your slides, textbooks, readings, and course outline into the `docs/` folder. Supported formats: **PDF, PPTX, DOCX, HTML, Markdown, Quarto (.qmd), plain text**.

```
docs/
├── slides/        # Lecture slides
├── textbooks/     # Textbook PDFs
├── outline/       # Syllabus / course outline
└── readings/      # Supplementary readings
```

**Nothing in `docs/` is tracked by git** — your materials stay private regardless of whether the repo is public or private.

### 5. Ingest your materials

```powershell
.venv\Scripts\python.exe -m scripts.ingest
```

This partitions, chunks, embeds, and uploads everything to your Qdrant collection. Only new or changed files are re-processed on subsequent runs. To force a full re-ingest:

```powershell
.venv\Scripts\python.exe -m scripts.ingest --reset
```

### 6. Launch the app

```powershell
.venv\Scripts\streamlit.exe run app.py
```

Opens at `http://localhost:8501`.

---

## Deploying to Streamlit Cloud

The deployed app connects to your Qdrant Cloud cluster over HTTPS, so it never needs your local `docs/` files or the offline ingestion toolchain. Follow these steps once you have already ingested your course materials into Qdrant locally.

### 1. Push your repo to GitHub

Course materials are gitignored (`docs/` and `data/`) so nothing sensitive gets exposed even if you push to a public repo. Verify with:

```powershell
git status --ignored
```

### 2. Create the Streamlit app

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with the same GitHub account.
2. Click **"New app"** and pick your repo + branch (usually `main`).
3. Set the entry point to `app.py`.
4. Under **Advanced settings → Python version**, select **3.11** (the repo's `runtime.txt` also pins this, so either is fine).

### 3. Add secrets

Under **App settings → Secrets**, paste the following TOML (fill in your real values — the same ones from your local `.env`):

```toml
COURSE_NAME = "My Course Name: Ask a question"

OPENROUTER_API_KEY = "sk-or-v1-..."
OPEN_AI_EMBEDDINGS_API_KEY = "sk-..."

QDRANT_URL = "https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY = "..."
```

`.streamlit/secrets.toml.example` in the repo mirrors this schema.

### 4. Deploy and share

Streamlit Cloud installs `requirements.txt` (runtime only — the heavier ingestion deps in `requirements-ingest.txt` are NOT installed on the server, keeping cold start under 2 minutes) and starts `app.py`. Once the build finishes, share the `https://<your-app>.streamlit.app` URL with students.

### Re-ingesting new course materials

Ingestion still runs locally on your machine, never on Streamlit Cloud:

```powershell
pip install -r requirements.txt -r requirements-ingest.txt   # once
.venv\Scripts\python.exe -m scripts.ingest
```

Because the deployed app queries Qdrant directly, new chunks show up in the app as soon as your local ingest completes — no redeploy required.

---

## Project structure

```
course-rag/
├── app.py                        # Streamlit chat UI (deployed target)
├── requirements.txt              # Runtime deps — installed by Streamlit Cloud
├── requirements-ingest.txt       # Extra deps for local ingestion only
├── runtime.txt                   # Pins Python version for Streamlit Cloud
├── .env                          # Your secrets — gitignored, never commit
├── .streamlit/
│   ├── config.toml               # Theme + telemetry-off (committed)
│   └── secrets.toml.example      # Key template for Streamlit Cloud
├── docs/                         # Your course materials — gitignored
│   └── README.md                 # Instructions for what to put here
├── scripts/
│   ├── ingest.py                 # Ingestion CLI: partition → chunk → embed → upsert
│   ├── partitioner.py            # Multi-format file partitioner
│   ├── chunker.py                # Semantic chunking + boilerplate cleaning
│   ├── sparse_encoder.py         # BM25-style sparse encoder (no extra deps)
│   └── inspect_chunks.py         # Diagnostic: chunk size distribution
└── data/
    └── elements_cache/           # Parsed element cache — gitignored
```

---

## Tuning for your course

All tuning parameters are constants at the top of `app.py`:

| Constant | Default | What it does |
|----------|---------|--------------|
| `RETRIEVER_K` | 6 | Chunks passed to the LLM per query |
| `RETRIEVER_FETCH_K` | 12 | Raw hits fetched before de-dup and diversity cap |
| `MAX_HITS_PER_SOURCE` | 2 | Max chunks from any single source file |
| `CHAT_HISTORY_TURNS` | 10 | Prior turns included for follow-up resolution |
| `SESSION_PIN_TOP_N` | 3 | Session-specific chunks pinned when query names a session |
| `QDRANT_COLLECTION` | `"course-docs"` | Qdrant collection name |
| `OPENROUTER_CHAT_MODEL_ID` | `"deepseek/deepseek-chat"` | Swap to any OpenRouter model |

The session-pinning regex (`SESSION_QUERY_RE`) defaults to sessions 1–13. Update the range in `_detect_session_number()` in `app.py` if your course has a different number of sessions/weeks.

