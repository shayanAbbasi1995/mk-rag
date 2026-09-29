"""Tests for scripts/ingest.py::find_removed_files — no network, no Qdrant."""
from pathlib import Path

from scripts.ingest import find_removed_files

DOCS = Path("/repo/docs")


def _key(rel: str) -> str:
    return str(DOCS / rel)


def test_nothing_removed_when_all_tracked_files_discovered():
    tracked = {_key("slides/Session 1/session-01.qmd"): "h1"}
    discovered = [DOCS / "slides/Session 1/session-01.qmd"]
    removed, unresolvable = find_removed_files(tracked, discovered, DOCS)
    assert removed == []
    assert unresolvable == []


def test_removed_file_maps_to_posix_source_path():
    tracked = {
        _key("slides/Session 1/session-01.qmd"): "h1",
        _key("slides/Session 1/solutions-01.qmd"): "h2",
    }
    discovered = [DOCS / "slides/Session 1/session-01.qmd"]
    removed, _ = find_removed_files(tracked, discovered, DOCS)
    # source_path must match what ingest_file stores on points:
    # docs/-relative, forward slashes, spaces preserved.
    assert removed == [
        (_key("slides/Session 1/solutions-01.qmd"), "slides/Session 1/solutions-01.qmd")
    ]


def test_new_files_are_not_reported_as_removed():
    tracked: dict[str, str] = {}
    discovered = [DOCS / "slides/Session 1/session-01-RLab.qmd"]
    removed, unresolvable = find_removed_files(tracked, discovered, DOCS)
    assert removed == []
    assert unresolvable == []


def test_tracker_entry_outside_docs_is_never_pruned():
    other_checkout = str(Path("/elsewhere/docs/slides/session-01.qmd"))
    removed, unresolvable = find_removed_files({other_checkout: "h"}, [], DOCS)
    assert removed == []
    assert unresolvable == [other_checkout]
