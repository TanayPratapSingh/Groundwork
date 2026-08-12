"""Hybrid index: BM25 sparse plus dense hashing vectors.

Two retrievers with uncorrelated failure modes. BM25 nails rare exact terms
(proper nouns, identifiers, years) and is blind to paraphrase. The dense side
tolerates surface variation and drowns rare terms in collisions. Fusing them
beats either alone, which the eval harness demonstrates rather than asserts:
`scripts/run_eval.py --ablate` reports recall@5 for sparse-only, dense-only,
and fused over the same real questions.

BM25 is implemented here rather than pulled in as a dependency because the
whole point of the file is to be readable, and because the scoring detail that
matters most (how document length normalisation is applied) is exactly the
thing a black-box import hides.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import INDEX_META_PATH, INDEX_PATH, SETTINGS
from .corpus import Document
from .embed import HashingEmbedder, tokenize


@dataclass
class Posting:
    """Inverted-index entry for one term."""

    doc_freq: int
    postings: dict[int, int]  # doc index -> term frequency


class BM25:
    """Okapi BM25 over a fixed document set.

    k1 controls term-frequency saturation, b controls length normalisation.
    The defaults (1.5, 0.75) are the standard ones and are not worth tuning
    before the retrieval metrics say they are the bottleneck.
    """

    def __init__(self, k1: float | None = None, b: float | None = None) -> None:
        cfg = SETTINGS.retrieval
        self.k1 = k1 if k1 is not None else cfg.bm25_k1
        self.b = b if b is not None else cfg.bm25_b
        self.index: dict[str, Posting] = {}
        self.doc_lengths: np.ndarray = np.zeros(0, dtype=np.float32)
        self.avg_doc_length: float = 0.0
        self.n_docs: int = 0

    def fit(self, tokenized_docs: list[list[str]]) -> "BM25":
        self.n_docs = len(tokenized_docs)
        self.doc_lengths = np.array([len(d) for d in tokenized_docs], dtype=np.float32)
        self.avg_doc_length = float(self.doc_lengths.mean()) if self.n_docs else 0.0

        index: dict[str, Posting] = {}
        for doc_idx, tokens in enumerate(tokenized_docs):
            for term, freq in Counter(tokens).items():
                entry = index.get(term)
                if entry is None:
                    index[term] = Posting(doc_freq=1, postings={doc_idx: freq})
                else:
                    entry.doc_freq += 1
                    entry.postings[doc_idx] = freq
        self.index = index
        return self

    def _idf(self, doc_freq: int) -> float:
        # Robertson-Sparck Jones IDF with the +0.5 smoothing. The max() floor
        # keeps terms that appear in more than half the corpus from
        # contributing a negative score, which would let a common word push a
        # document *down* the ranking.
        return max(1e-6, math.log(1.0 + (self.n_docs - doc_freq + 0.5) / (doc_freq + 0.5)))

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float32)
        if self.n_docs == 0:
            return scores

        for term in tokenize(query):
            entry = self.index.get(term)
            if entry is None:
                continue
            idf = self._idf(entry.doc_freq)
            for doc_idx, tf in entry.postings.items():
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * self.doc_lengths[doc_idx] / (self.avg_doc_length or 1.0)
                )
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0)) / denom
        return scores

    def to_dict(self) -> dict:
        return {
            "k1": self.k1,
            "b": self.b,
            "n_docs": self.n_docs,
            "avg_doc_length": self.avg_doc_length,
            "doc_lengths": self.doc_lengths.tolist(),
            "index": {
                term: {"df": p.doc_freq, "p": {str(k): v for k, v in p.postings.items()}}
                for term, p in self.index.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "BM25":
        bm = cls(k1=payload["k1"], b=payload["b"])
        bm.n_docs = payload["n_docs"]
        bm.avg_doc_length = payload["avg_doc_length"]
        bm.doc_lengths = np.array(payload["doc_lengths"], dtype=np.float32)
        bm.index = {
            term: Posting(doc_freq=e["df"], postings={int(k): v for k, v in e["p"].items()})
            for term, e in payload["index"].items()
        }
        return bm


class HybridIndex:
    """Everything needed to retrieve: documents, BM25, and dense vectors."""

    def __init__(self, embedder: HashingEmbedder | None = None) -> None:
        self.embedder = embedder or HashingEmbedder()
        self.documents: list[Document] = []
        self.vectors: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self.bm25 = BM25()
        self._id_to_idx: dict[str, int] = {}

    # -- build ------------------------------------------------------------

    def build(self, documents: list[Document]) -> "HybridIndex":
        self.documents = list(documents)
        self._id_to_idx = {d.doc_id: i for i, d in enumerate(self.documents)}

        # Title is prepended to the indexed text so that questions naming the
        # article subject ("When did the Normans arrive?") get a lexical hit
        # even when the paragraph body never repeats the title.
        indexed = [f"{d.title}. {d.text}" for d in self.documents]
        self.vectors = self.embedder.embed_many(indexed)
        self.bm25.fit([tokenize(t) for t in indexed])
        return self

    # -- query ------------------------------------------------------------

    def dense_scores(self, query: str) -> np.ndarray:
        if not self.documents:
            return np.zeros(0, dtype=np.float32)
        return self.vectors @ self.embedder.embed_one(query)

    def sparse_scores(self, query: str) -> np.ndarray:
        return self.bm25.score(query)

    # -- lookup -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.documents)

    def get(self, doc_id: str) -> Document | None:
        idx = self._id_to_idx.get(doc_id)
        return self.documents[idx] if idx is not None else None

    def position_of(self, doc_id: str) -> int | None:
        return self._id_to_idx.get(doc_id)

    # -- persistence ------------------------------------------------------
    # Vectors go to .npz (compact, binary); everything else to JSON (readable,
    # diffable, and inspectable without loading numpy).

    def save(self, vectors_path: Path | None = None, meta_path: Path | None = None) -> None:
        vectors_path = vectors_path or INDEX_PATH
        meta_path = meta_path or INDEX_META_PATH
        vectors_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(vectors_path, vectors=self.vectors)
        meta_path.write_text(json.dumps({
            "embed_dim": self.embedder.dim,
            "documents": [
                {"doc_id": d.doc_id, "title": d.title, "text": d.text} for d in self.documents
            ],
            "bm25": self.bm25.to_dict(),
        }))

    @classmethod
    def load(cls, vectors_path: Path | None = None, meta_path: Path | None = None) -> "HybridIndex":
        vectors_path = vectors_path or INDEX_PATH
        meta_path = meta_path or INDEX_META_PATH
        if not vectors_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"No index at {vectors_path}. Build it first:\n\n"
                f"    python3 scripts/build_index.py\n"
            )

        meta = json.loads(meta_path.read_text())
        index = cls(embedder=HashingEmbedder(dim=meta["embed_dim"]))
        index.documents = [Document(**d) for d in meta["documents"]]
        index._id_to_idx = {d.doc_id: i for i, d in enumerate(index.documents)}
        index.vectors = np.load(vectors_path)["vectors"]
        index.bm25 = BM25.from_dict(meta["bm25"])
        return index

    def stats(self) -> dict:
        return {
            "documents": len(self.documents),
            "vocabulary": len(self.bm25.index),
            "embed_dim": self.embedder.dim,
            "mean_doc_tokens": round(float(self.bm25.doc_lengths.mean()), 1) if len(self.bm25.doc_lengths) else 0.0,
        }
