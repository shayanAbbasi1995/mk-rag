"""
Stage 2 (element cleaning) + Stage 3 (element-based semantic chunking).

Follows the reference PDF's recipe:
  * Drop noise categories (Header, Footer, PageNumber) that interrupt
    narrative continuity and pollute similarity search.
  * Drop the References/Bibliography section and all elements that hang
    off it via parent_id, so citation dumps don't crowd out real content.
  * Feed the cleaned element stream into chunk_by_title(), which packs
    adjacent atomic elements upward under their governing Title until a
    new Title (or the max size) forces a break. This preserves topical
    boundaries far better than blind character-count splitting.
"""

from __future__ import annotations

import logging
import re

from unstructured.chunking.title import chunk_by_title
from unstructured.documents.elements import Element

logger = logging.getLogger(__name__)

# Categories that carry no retrievable meaning and interrupt paragraph flow.
# UncategorizedText is included because unstructured occasionally emits it
# for tiny fragments (single characters, punctuation) that pdfminer picks
# out of headers/footers.
NOISE_CATEGORIES: set[str] = {"Header", "Footer", "PageNumber"}

# Section titles whose entire subtree we want to drop. Kept as a regex so we
# tolerate common variants (e.g. "REFERENCES", "References", "R e f e r e n c e s").
REFERENCE_TITLE_RE = re.compile(
    r"^\s*(references|bibliography|works\s+cited|literature\s+cited)\s*[:.]?\s*$",
    re.IGNORECASE,
)

# Pattern for classic page-footer noise pdfminer/pypdf leak: "12 of 47",
# "Page 3", bare integer runs, and publisher/book-title running footers
# like "Essentials of Marketing Research: Putting Research into Practice".
#
# Applied twice:
#   (a) at element level for short elements (<200 chars)  -> catches raw
#       footer paragraphs before they reach chunk_by_title;
#   (b) at chunk level after chunking, to catch chunks whose entire text
#       is one of these residues (chunk_by_title can leave a tail chunk
#       consisting solely of running footer text).
PAGE_FOOTER_RE = re.compile(
    r"^\s*("
    r"page\s+\d+"                                       # "Page 12"
    r"|\d+\s+of\s+\d+.*"                                # "19 of 35 …"
    r"|\d{1,4}"                                         # bare page number
    r"|(sage|wiley|routledge|springer|elsevier)\s*publications?"
    r"|((the\s+)?(absolute\s+)?essentials\s+of\s+marketing\s+research"
    r"[:,\s]*(putting\s+research\s+into\s+practice)?)"  # book running footer
    r"|research\s*:\s*putting\s+research\s+into\s+practice"
    r"|putting\s+research\s+into\s+practice"
    r")\s*$",
    re.IGNORECASE,
)


def _element_category(el: Element) -> str:
    """`Element.category` is the class name in most unstructured versions,
    but some element types expose it via .to_dict()['type']. Normalise
    here so the filter is robust across releases."""
    return getattr(el, "category", None) or el.to_dict().get("type", "")


def clean_elements(elements: list[Element]) -> list[Element]:
    """Remove headers/footers/page numbers and any element belonging to a
    References subtree.

    Reference-removal is a two-pass process:
      pass 1: find every Title element whose text matches REFERENCE_TITLE_RE
              and collect their element_ids into `banned_parent_ids`.
      pass 2: keep an element only if it is neither the banned title itself
              nor a child of one via parent_id chaining.

    Chaining matters: a "References" title may have a "Chapter 3 Refs"
    sub-heading, whose own children would inherit that sub-heading's
    parent_id, not the top-level References id. To be safe we iterate to
    fixed point and add any element whose parent is already banned.
    """
    # Pass 1: seed banned ids with reference-section titles.
    banned_ids: set[str] = set()
    for el in elements:
        category = _element_category(el)
        text = (el.text or "").strip()
        if category == "Title" and REFERENCE_TITLE_RE.match(text):
            el_id = getattr(el, "id", None) or el.to_dict().get("element_id")
            if el_id:
                banned_ids.add(el_id)

    # Pass 2: expand banned set to descendants (fixed-point).
    if banned_ids:
        changed = True
        while changed:
            changed = False
            for el in elements:
                el_id = getattr(el, "id", None) or el.to_dict().get("element_id")
                parent_id = getattr(el.metadata, "parent_id", None)
                if el_id and parent_id in banned_ids and el_id not in banned_ids:
                    banned_ids.add(el_id)
                    changed = True

    kept: list[Element] = []
    dropped_noise = 0
    dropped_refs = 0
    dropped_footer = 0
    for el in elements:
        category = _element_category(el)
        if category in NOISE_CATEGORIES:
            dropped_noise += 1
            continue
        el_id = getattr(el, "id", None) or el.to_dict().get("element_id")
        if el_id in banned_ids:
            dropped_refs += 1
            continue
        text = (el.text or "").strip()
        # Empty/whitespace-only elements add nothing but tokens.
        if not text:
            continue
        # Page-footer noise: short strings matching known footer patterns.
        # We only apply this to short elements so a paragraph that happens
        # to start with "Page 12 was ..." isn't discarded.
        if len(text) < 200 and PAGE_FOOTER_RE.match(text):
            dropped_footer += 1
            continue
        kept.append(el)

    logger.debug(
        "clean_elements: kept %d, dropped %d noise, %d reference-tree, %d footer",
        len(kept),
        dropped_noise,
        dropped_refs,
        dropped_footer,
    )
    return kept


def _is_pure_footer_chunk(text: str) -> bool:
    """After chunking, some chunks are just concatenated footer residues
    (e.g., 'Essentials of Marketing Research | Putting Research into
    Practice'). Detect these by stripping known footer substrings and
    checking whether anything of substance remains."""
    stripped = re.sub(PAGE_FOOTER_RE, "", text.strip())
    for line in text.splitlines():
        line = line.strip()
        if line and PAGE_FOOTER_RE.match(line):
            stripped = stripped.replace(line, "")
    return len(re.sub(r"[\s|,.:;-]+", "", stripped)) < 15


def chunk_elements(
    elements: list[Element],
    max_characters: int = 1200,
    combine_text_under_n_chars: int = 200,
    new_after_n_chars: int = 1000,
) -> list[Element]:
    """Wrap unstructured.chunking.title.chunk_by_title with defaults tuned
    for lecture slides + textbook chapters.

    Rationale for the numbers:
      * max_characters=1200 -> ~300 tokens per chunk on average, well
        below the OpenAI embedding input limit (8191 tokens) but big
        enough to preserve a single conceptual unit (one slide + notes,
        or one textbook subsection).
      * combine_text_under_n_chars=200 -> lecture decks are dense with
        one-line bullets; without combining, we'd get many tiny chunks
        whose embeddings carry no context.
      * new_after_n_chars=1000 -> soft break; the chunker will start a
        new chunk once it hits this size *unless* it's mid-section, in
        which case it can grow up to max_characters.
    """
    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        combine_text_under_n_chars=combine_text_under_n_chars,
        new_after_n_chars=new_after_n_chars,
    )
    # Post-chunk footer suppression: drop chunks that are (a) entirely
    # running-footer content, or (b) too short (<30 chars) to carry
    # useful context on their own. chunk_by_title's combine_text_under_n_chars
    # already merges most small fragments upward; anything that survives
    # into a standalone chunk under 30 chars is almost always a footer
    # residue or a stray heading that lost its body content.
    filtered: list[Element] = []
    dropped = 0
    for c in chunks:
        text = (c.text or "").strip()
        if len(text) < 30 or _is_pure_footer_chunk(text):
            dropped += 1
            continue
        filtered.append(c)
    if dropped:
        logger.debug("chunk_elements: dropped %d footer/tiny chunks", dropped)
    return filtered
