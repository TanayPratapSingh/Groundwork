#!/usr/bin/env python3
"""Build the hybrid retrieval index over the real corpus.

    python3 scripts/build_index.py
    python3 scripts/build_index.py --stats-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from groundwork.config import INDEX_META_PATH, INDEX_PATH  # noqa: E402
from groundwork.rag.corpus import corpus_stats, load_corpus, load_questions  # noqa: E402
from groundwork.rag.index import HybridIndex  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats-only", action="store_true",
                        help="describe the corpus without building")
    args = parser.parse_args()

    documents = load_corpus()
    questions = load_questions()
    stats = corpus_stats(documents, questions)

    print("Corpus (SQuAD v2.0 dev, real Wikipedia paragraphs)")
    for key, value in stats.items():
        print(f"  {key:<16} {value}")

    if args.stats_only:
        return 0

    print("\nBuilding hybrid index (BM25 + hashed dense vectors)")
    started = time.perf_counter()
    index = HybridIndex().build(documents)
    index.save()
    elapsed = time.perf_counter() - started

    print(f"  built in {elapsed:.2f}s")
    for key, value in index.stats().items():
        print(f"  {key:<16} {value}")
    print(f"\n  vectors -> {INDEX_PATH}")
    print(f"  metadata -> {INDEX_META_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
