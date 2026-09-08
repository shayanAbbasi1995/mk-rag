"""
Ingest documents from docs/ into the Qdrant vector store.

Run: python ingest.py
Only files that are new or have changed (by MD5 hash) are re-processed.
"""

import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()

DOCS_DIR = Path("docs")
TRACKER_FILE = Path("index_tracker.json")
COLLECTION_NAME = "course-docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("OPEN_AI_EMBEDDINGS_API_KEY"),
)

client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
)


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_tracker() -> dict:
    if TRACKER_FILE.exists():
        return json.loads(TRACKER_FILE.read_text())
    return {"last_updated": None, "files": {}}


def save_tracker(tracker: dict) -> None:
    from datetime import datetime, timezone
    tracker["last_updated"] = datetime.now(timezone.utc).isoformat()
    TRACKER_FILE.write_text(json.dumps(tracker, indent=2))


def ensure_collection() -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print(f"Created collection '{COLLECTION_NAME}'")


def load_document(path: Path) -> list:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".qmd", ".md", ".txt"}:
        return TextLoader(str(path), encoding="utf-8").load()
    return []


def ingest() -> None:
    ensure_collection()
    tracker = load_tracker()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )

    changed: list[Path] = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.name.startswith(".") or path.suffix.lower() not in {
            ".pdf", ".qmd", ".md", ".txt"
        }:
            continue
        file_hash = md5(path)
        if tracker["files"].get(str(path)) != file_hash:
            changed.append(path)
            tracker["files"][str(path)] = file_hash

    if not changed:
        print("Nothing new to ingest.")
        return

    print(f"Ingesting {len(changed)} file(s): {[p.name for p in changed]}")

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )

    for path in changed:
        docs = load_document(path)
        if not docs:
            print(f"  Skipped (unsupported or empty): {path.name}")
            continue
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["source_file"] = path.name
        store.add_documents(chunks)
        print(f"  Indexed {len(chunks)} chunks from {path.name}")

    save_tracker(tracker)
    print("Done.")


if __name__ == "__main__":
    ingest()
