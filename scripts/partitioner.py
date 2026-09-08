"""
Stage 1 of the RAG ingestion pipeline: Partitioning & Model Inference.

Turns format-specific source files (PDF, HTML, Markdown/QMD, DOCX) into a
uniform list of `unstructured` Element objects (Title, NarrativeText, Table,
ListItem, ...) with metadata (page number, category, parent_id) attached.

Serialised element JSON is cached under data/elements_cache/<file_hash>.json.
Per the reference PDF ("Decouples Pipeline Stages: Document extraction / OCR
is computationally expensive; downstream chunking is cheap. Caching JSON
allows endless chunking experiments without re-parsing files.") this keeps
iteration on chunking / retrieval fast even after the corpus grows.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable

from unstructured.documents.elements import (
    Element,
    NarrativeText,
    Text,
    Title,
)
from unstructured.staging.base import elements_from_json, elements_to_json

logger = logging.getLogger(__name__)

# Extensions we know how to partition. `.qmd` is Quarto markdown — same
# structure as plain markdown for partitioning purposes, so we treat it
# as `.md`. Anything else is logged and skipped rather than silently dropped.
SUPPORTED_EXTENSIONS: set[str] = {
    ".pdf",
    ".html",
    ".htm",
    ".md",
    ".qmd",
    ".docx",
    ".pptx",
    ".txt",
}


def file_md5(path: Path) -> str:
    """MD5 hash of a file's bytes. Used both for cache keys and change
    detection in the tracker, so both call sites stay in sync."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _strip_quarto_scaffolding(text: str) -> str:
    """Remove YAML front matter and Quarto div fences from a .qmd source.

    Removed:
      * Leading YAML block delimited by `---` on its own line.
      * Quarto fence lines: `::: {.notes}`, `::::`, `:::` (closing).
      * `{background-color=...}` and similar slide-attribute suffixes on
        `##`/`#` headings — these clutter Title elements.

    Kept:
      * All prose, code chunks, bullet lists, and speaker-notes bodies.
        Speaker notes carry high-signal instructor commentary and are
        exactly what students want the assistant to be able to explain.
    """
    import re as _re

    # 1) Strip YAML front matter.
    yaml_re = _re.compile(r"^---\r?\n.*?\r?\n---\r?\n", flags=_re.DOTALL)
    text = yaml_re.sub("", text, count=1)

    # 2) Drop Quarto div fences (::: on its own line, with optional attrs).
    text = _re.sub(r"^:::+.*$", "", text, flags=_re.MULTILINE)

    # 3) Strip {attribute=...} tails from ATX headings.
    text = _re.sub(r"^(#+\s.+?)\s*\{[^}]*\}\s*$", r"\1", text, flags=_re.MULTILINE)

    return text


def _partition_pdf_fast(path: Path) -> list[Element]:
    """Rule-based PDF -> Element extraction using pypdf.

    Preserves the element model (Title vs. NarrativeText) that
    chunk_by_title needs to work, and stamps page numbers into
    element.metadata.page_number so retrieval can cite them.

    Heuristic: a line is treated as a Title if it is short (<=80 chars),
    doesn't end in sentence punctuation, and is either fully uppercase
    or starts with a chapter/section marker. Otherwise consecutive
    non-blank lines form a NarrativeText paragraph. This is intentionally
    conservative — it errs on the side of "narrative" so we never lose
    body text — but gives chunk_by_title enough Title anchors to keep
    lecture-slide sections coherent.
    """
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    elements: list[Element] = []

    title_prefixes = (
        "chapter ", "section ", "part ", "appendix ",
        "learning objectives", "key terms", "summary", "questions",
        "references", "bibliography",
    )

    for page_index, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception as e:  # noqa: BLE001
            logger.warning("pypdf extract_text failed on p%d of %s: %s",
                           page_index, path.name, e)
            continue

        # Split into logical paragraphs on blank lines. Within a paragraph
        # we join runs of short lines that pypdf gave us from a wrapped
        # column layout.
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        for para in paragraphs:
            lines = [ln.strip() for ln in para.splitlines() if ln.strip()]
            if not lines:
                continue

            # First line becomes a Title candidate on its own only when
            # it looks like a heading, so we don't strip meaningful body
            # text out of the paragraph.
            first = lines[0]
            first_lower = first.lower()
            is_title = (
                len(first) <= 80
                and not first.endswith((".", "?", "!", ":", ";", ","))
                and (
                    first.isupper()
                    or any(first_lower.startswith(p) for p in title_prefixes)
                )
            )

            if is_title and len(lines) == 1:
                el: Element = Title(text=first)
                el.metadata.page_number = page_index
                elements.append(el)
                continue

            if is_title:
                t = Title(text=first)
                t.metadata.page_number = page_index
                elements.append(t)
                body_lines = lines[1:]
            else:
                body_lines = lines

            body = " ".join(body_lines)
            if body:
                # Very short residual fragments (<20 chars) are typically
                # header/footer artifacts — emit as generic Text so the
                # downstream cleaner can drop them without also killing
                # real narrative.
                if len(body) < 20:
                    n: Element = Text(text=body)
                else:
                    n = NarrativeText(text=body)
                n.metadata.page_number = page_index
                elements.append(n)

    return elements


def _partition_file(path: Path) -> list[Element]:
    """Dispatch to the format-specific `partition_*()` for a given file.

    We deliberately avoid `unstructured.partition.auto.partition` because
    it defers to file-type detection that occasionally mis-classifies our
    .qmd files (they lack a standard extension). Explicit dispatch also
    lets us pick per-format strategy flags (e.g. PDF strategy="fast").
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # We deliberately do NOT use unstructured.partition.pdf here.
        # partition_pdf imports unstructured_inference at module-load
        # time (even with strategy="fast"), which pulls in PyTorch,
        # transformers, opencv, and ~1 GB of vision-model weights we
        # never invoke. Our corpus is 100% digital PDFs, so we go
        # directly to pypdf for a fast, rule-based extraction and
        # emit unstructured Element objects ourselves. This is the
        # "Rule-Based (Local / Open Source)" processing path in the
        # reference PDF: fast, lightweight, no OCR.
        return _partition_pdf_fast(path)

    if suffix in {".html", ".htm"}:
        from unstructured.partition.html import partition_html
        return partition_html(filename=str(path))

    if suffix in {".md", ".qmd"}:
        # partition_md wants a real markdown extension. Read the text and
        # pass it via `text=` so Quarto's .qmd files are handled uniformly.
        # For .qmd we first strip YAML front matter and Quarto div fences
        # (::: {.notes}, ::: columns, etc.); those markup tokens don't
        # carry retrievable meaning and would otherwise dominate short
        # slide-title chunks.
        from unstructured.partition.md import partition_md
        text = path.read_text(encoding="utf-8", errors="replace")
        if suffix == ".qmd":
            text = _strip_quarto_scaffolding(text)
        return partition_md(text=text)

    if suffix == ".docx":
        from unstructured.partition.docx import partition_docx
        return partition_docx(filename=str(path))

    if suffix == ".pptx":
        from unstructured.partition.pptx import partition_pptx
        return partition_pptx(filename=str(path))

    if suffix == ".txt":
        from unstructured.partition.text import partition_text
        return partition_text(filename=str(path))

    raise ValueError(f"Unsupported extension: {suffix}")


def partition_with_cache(
    path: Path,
    cache_dir: Path,
    file_hash: str | None = None,
) -> list[Element]:
    """Return elements for `path`, using an on-disk JSON cache keyed by
    the file's MD5. If the cache hit is fresh, we skip re-parsing entirely.

    Passing `file_hash` in avoids a second read of the file when the caller
    already computed the hash (typical, since the ingest loop uses the hash
    for its change-detection tracker).
    """
    if file_hash is None:
        file_hash = file_md5(path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{file_hash}.json"

    if cache_path.exists():
        try:
            elements = elements_from_json(filename=str(cache_path))
            logger.debug("Cache hit: %s (%d elements)", path.name, len(elements))
            return elements
        except Exception as e:  # noqa: BLE001 — cache corruption isn't fatal
            logger.warning(
                "Cache read failed for %s (%s); re-partitioning.",
                cache_path.name,
                e,
            )

    logger.info("Partitioning %s ...", path.name)
    elements = _partition_file(path)

    # Stamp each element's metadata with the true source path. `unstructured`
    # sometimes stores a temp filename here, which would break downstream
    # source-attribution ("cite the file this claim came from").
    for el in elements:
        el.metadata.filename = path.name
        el.metadata.file_directory = str(path.parent)

    elements_to_json(elements, filename=str(cache_path))
    logger.debug("Cached elements to %s", cache_path)
    return elements


def iter_source_files(root: Path) -> Iterable[Path]:
    """Yield ingestable files under `root`, deterministically ordered.

    Skips dotfiles, empty files, and any files whose extension isn't in
    SUPPORTED_EXTENSIONS. We also exclude the R history file and any
    duplicate "-extended" slide-deck variants — the extended decks are
    superset content that would inflate embedding cost without adding
    much retrieval value (they cover the same topics with more speaker
    notes).
    """
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.stat().st_size == 0:
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # De-dup: keep session-NN.qmd/.html, drop the -extended twin.
        # Both cover the same session topically; ingesting both would
        # double-count the same material at retrieval time.
        stem = path.stem.lower()
        if stem.endswith("-extended"):
            continue
        yield path
