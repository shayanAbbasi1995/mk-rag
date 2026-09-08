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
pip install -r requirements.txt
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

1. Push your fork to GitHub (docs are gitignored — no course materials are exposed)
2. Connect the repo at [share.streamlit.io](https://share.streamlit.io)
3. Add your secrets under **App settings → Secrets** (same key names as `.env`)

The deployed app reads from Qdrant at runtime — `docs/` is never needed on the server.

---

## Project structure

```
course-rag/
├── app.py                        # Streamlit chat UI
├── requirements.txt              # Pinned dependencies
├── .env                          # Your secrets — gitignored, never commit
├── .streamlit/
│   ├── config.toml               # Disables Streamlit usage-stats prompt
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

