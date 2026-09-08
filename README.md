# M731 Marketing Research — Course RAG

A low-cost RAG assistant for the M731 Marketing Research course. Students ask questions; the app retrieves relevant content from 92 course documents (lecture slides, textbooks, course outline) and answers via DeepSeek-V3 through OpenRouter.

**Estimated cost: under $3 for a full semester with 50 students.**

## Stack

| Component | Tool | Cost |
|-----------|------|------|
| LLM | DeepSeek-V3 via OpenRouter (`deepseek/deepseek-chat`) | ~$0.26/1M input tokens |
| Embeddings | `text-embedding-3-small` (OpenAI) | $0.02/1M tokens (~$0.04 to embed entire corpus) |
| Vector store | Qdrant Cloud free tier | $0 (1 GB perpetual free) |
| UI | Streamlit | $0 |

## Retrieval pipeline

```
Student query
  ↓
[Session-aware pinning]   ← if query mentions "session N / week N / lecture N",
  ↓                          top-3 chunks from that session are pinned first
[Hybrid search — RRF]
  ├─ Dense:  OpenAI text-embedding-3-small  →  Qdrant ANN (top 12)
  └─ Sparse: BM25 (FNV-1a hashed TF)       →  Qdrant sparse index (top 12)
  ↓                          RRF fusion
[De-dup + diversity cap]  ← collapse (source, chunk_index) twins; max 2 chunks per source file
  ↓
[Top 6 chunks → LLM context]
  ↓
DeepSeek-V3 via OpenRouter  (last 10 turns of conversation history included)
  ↓
Answer + numbered source citations
```

Each chunk is stored with a **contextual header** breadcrumb (e.g. `[Part 2 — Reliability > Cronbach's alpha]`) prepended before embedding, so chunks are self-contained even when retrieved out of context.

## Setup

**1. Accounts required (one-time)**

| Service | Free tier | What you need |
|---------|-----------|---------------|
| [OpenRouter](https://openrouter.ai) | Pay-as-you-go | API key + $5 deposit |
| [OpenAI](https://platform.openai.com) | Pay-as-you-go | API key + $5 credit (embeddings only) |
| [Qdrant Cloud](https://cloud.qdrant.io) | 1 GB free forever | Cluster URL + API key |

**2. Clone and create a virtual environment**

```powershell
git clone <repo-url>
cd mk-rag
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

**3. Configure secrets**

Create a `.env` file in the repo root (gitignored):

```env
OPENROUTER_API_KEY=sk-or-...
OPEN_AI_EMBEDDINGS_API_KEY=sk-...
QDRANT_URL=https://<cluster-id>.cloud.qdrant.io:6333
QDRANT_API_KEY=...
QDRANT_CLUSTER_NAME=...
```

For Streamlit Cloud deployment, copy the same keys into `.streamlit/secrets.toml` (see `.streamlit/secrets.toml.example`).

**4. Add course materials**

Place PDFs, `.qmd`, `.md`, `.html`, `.docx`, or `.txt` files anywhere under `docs/`. Course PDFs are gitignored — store them in OneDrive and copy locally before ingesting.

**5. Ingest documents**

```powershell
.venv\Scripts\python.exe -m scripts.ingest
```

Only new or changed files are re-processed (MD5-tracked via `index_tracker.json`). To force a full re-ingest:

```powershell
.venv\Scripts\python.exe -m scripts.ingest --reset
```

**6. Launch the app**

```powershell
.venv\Scripts\streamlit.exe run app.py
```

Opens at `http://localhost:8501`.

## Project structure

```
mk-rag/
├── app.py                        # Streamlit chat UI (hybrid retrieval + DeepSeek-V3)
├── requirements.txt              # Pinned dependencies
├── .env                          # Secrets — gitignored, never commit
├── .streamlit/
│   ├── config.toml               # Disables Streamlit usage stats prompt
│   └── secrets.toml.example      # Key template for Streamlit Cloud deploy
├── docs/
│   ├── Outline/                  # Course outline (.docx)
│   ├── slides/                   # Quarto lecture slides (Sessions 1–13, .qmd + .html)
│   │   └── Session N/
│   │       ├── session-NN.qmd
│   │       ├── session-NN-extended.qmd
│   │       ├── solutions-NN.qmd
│   │       └── data/             # In-class datasets (.csv)
│   ├── TextBooks/                # Textbook PDFs — gitignored
│   └── overview.qmd              # Course overview
├── scripts/
│   ├── ingest.py                 # Ingestion CLI: partition → chunk → embed → upsert
│   ├── partitioner.py            # Multi-format partitioner (PDF, PPTX, QMD, HTML, DOCX, MD)
│   ├── chunker.py                # chunk_by_title + boilerplate cleaning
│   ├── sparse_encoder.py         # BM25-style sparse encoder (FNV-1a hashing, no extra deps)
│   ├── inspect_chunks.py         # Diagnostic: chunk size distribution
│   ├── eval_retrieval.py         # Retrieval quality probe (pre-fix baseline)
│   ├── eval_after_improvements.py # Retrieval quality probe (post-fix)
│   ├── audit_duplicates.py       # Qdrant duplicate detector
│   ├── dedupe_qdrant.py          # One-time dedup of the live collection
│   ├── debug_session1.py         # Session-retrieval diagnostic
│   ├── verify_session1_fix.py    # Session-pinning verification
│   └── smoke_e2e.py              # End-to-end smoke test
└── data/
    └── elements_cache/           # JSON element cache per source file — gitignored
```

## Re-ingesting after updates

`index_tracker.json` (gitignored) tracks processed file hashes. Adding new files to `docs/` and re-running `scripts/ingest.py` only processes what changed. The ingester is safe to re-run — it purges stale Qdrant points for any file it re-processes before upserting fresh ones.

## Qdrant collection schema

- **Collection:** `course-docs`
- **Dense vector:** `dense` — 1536-dim, cosine distance (`text-embedding-3-small`)
- **Sparse vector:** `sparse` — BM25-style TF, Qdrant sparse index
- **Payload fields:** `text`, `context_header`, `source_file`, `source_path`, `page_number`, `element_category`, `chunk_index`, `element_id`, `parent_id`, `file_hash`
- **Payload indexes:** `source_file`, `source_path`, `element_category`, `file_hash` (keyword); `page_number` (integer)
