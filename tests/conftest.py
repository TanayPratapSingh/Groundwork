"""Shared fixtures.

Tests that need the real corpus skip cleanly when it has not been fetched, so
`pytest` works on a fresh clone. Tests that exercise logic build their own
tiny fixtures and never touch the network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from groundwork.config import RAW_DIR  # noqa: E402
from groundwork.rag.corpus import Document  # noqa: E402
from groundwork.rag.index import HybridIndex  # noqa: E402

requires_corpus = pytest.mark.skipif(
    not (RAW_DIR / "squad_v2_dev.json").exists(),
    reason="corpus not fetched; run scripts/fetch_data.py",
)

requires_attacks = pytest.mark.skipif(
    not (RAW_DIR / "prompt_injections.jsonl").exists(),
    reason="attack corpora not fetched; run scripts/fetch_data.py",
)


@pytest.fixture
def tiny_documents() -> list[Document]:
    return [
        Document("doc-1", "Photosynthesis",
                 "Photosynthesis is the process by which green plants convert light energy "
                 "into chemical energy. Chlorophyll absorbs light in the blue and red parts "
                 "of the spectrum."),
        Document("doc-2", "Mitochondria",
                 "Mitochondria generate most of the chemical energy needed to power a cell's "
                 "biochemical reactions. They are often called the powerhouse of the cell."),
        Document("doc-3", "Rhine",
                 "Duisburg is the home of Europe's largest inland port. The Rhine flows "
                 "through Switzerland, Germany and the Netherlands before reaching the sea."),
    ]


@pytest.fixture
def tiny_index(tiny_documents) -> HybridIndex:
    return HybridIndex().build(tiny_documents)
