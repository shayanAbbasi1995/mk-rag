# docs/

Place your course materials here before running the ingestion script.

Supported formats: **PDF, PPTX, DOCX, HTML, Markdown (`.md`), Quarto (`.qmd`), plain text (`.txt`)**

Organise files however you like — the ingester walks the entire `docs/` tree recursively.

## Example layout

```
docs/
├── slides/          # Lecture slides (PDF or Quarto .qmd)
├── textbooks/       # Textbook PDFs
├── outline/         # Course outline / syllabus
└── readings/        # Supplementary readings
```

## What gets committed to git

**Nothing in this folder is tracked by git** — your course materials stay local and private. Only this `README.md` is committed as a placeholder.

After adding your files, run:

```powershell
.venv\Scripts\python.exe -m scripts.ingest
```

This embeds everything and uploads vectors to your Qdrant collection. The Streamlit app reads from Qdrant at runtime, so the docs folder is never needed on the server.
