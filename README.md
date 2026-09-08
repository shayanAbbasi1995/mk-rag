# M731 Marketing Research — Course RAG

A low-cost RAG assistant for course materials. Students ask questions; the app retrieves relevant lecture content from Qdrant and answers via DeepSeek-V3 through OpenRouter.

**Estimated cost: under $3 for a full semester with 50 students.**

## Stack

| Component | Tool | Cost |
|-----------|------|------|
| LLM | DeepSeek-V3 via OpenRouter | ~$0.26/1M input tokens |
| Embeddings | `text-embedding-3-small` (OpenAI) | $0.02/1M tokens |
| Vector store | Qdrant Cloud free tier | $0 (1 GB perpetual free) |
| UI | Streamlit | $0 |

## Setup

**1. Create and activate a virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure secrets**

Copy the example secrets file and fill in your keys:

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Edit `.streamlit/secrets.toml`:

```toml
OPENROUTER_API_KEY = "sk-or-..."      # from openrouter.ai
OPEN_AI_EMBEDDINGS_API_KEY = "sk-..."  # from platform.openai.com
QDRANT_URL = "https://..."             # from cloud.qdrant.io
QDRANT_API_KEY = "..."                 # from cloud.qdrant.io
```

Alternatively, set the same keys as environment variables in a `.env` file (same key names).

**4. Add course materials**

Place PDFs, Markdown (`.md`), Quarto (`.qmd`), or plain text (`.txt`) files in the `docs/` folder. Course PDFs are not committed to git — store them in OneDrive/shared drive and copy locally before ingesting.

**5. Ingest documents into Qdrant**

```bash
python ingest.py
```

Only new or changed files are re-processed (tracked by MD5 hash). Re-run whenever you add new materials.

**6. Launch the app**

```bash
streamlit run app.py
```

## Project structure

```
mk-rag/
├── app.py               # Streamlit chat interface
├── ingest.py            # Document ingestion CLI
├── requirements.txt     # Pinned dependencies
├── docs/                # Course materials (not committed)
├── .streamlit/
│   └── secrets.toml.example   # Key template (copy → secrets.toml)
└── TECH_DEBT_AUDIT.md   # Codebase health audit
```

## Re-ingesting after updates

`ingest.py` tracks file hashes in `index_tracker.json` (local only, not committed). To force a full re-ingest, delete `index_tracker.json` and re-run.
