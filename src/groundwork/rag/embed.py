"""A dependency-free hashing embedder.

Honest description: this is a signed feature hash of unigrams and bigrams with
sublinear term frequency and L2 normalisation. It has no semantic
generalisation. "car" and "automobile" land in unrelated dimensions.

It is here for two reasons. It makes the entire stack runnable offline and
deterministically, so the eval numbers in docs/06-results.md are reproducible
by anyone with a clone and no API key. And fused with BM25 it is a genuinely
reasonable lexical retriever, which sets an honest floor: any dense model you
swap in has to beat this on recall@5 to justify its latency and cost.

Swapping is a two-method contract. `retriever`, `cache`, and the whole eval
harness are unaffected, and `scripts/run_eval.py` is exactly how you would
argue the swap paid for itself.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from ..config import SETTINGS

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Deliberately short. Aggressive stoplists quietly destroy recall on the short
# queries that dominate real traffic ("who wrote it", "when did it end").
_STOPWORDS = frozenset(
    """a an and are as at be by for from has have in into is it its of on or
    that the to was were with""".split()
)


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if drop_stopwords:
        return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
    return [t for t in tokens if len(t) > 1]


class HashingEmbedder:
    """Text to a fixed-dimension unit vector via signed feature hashing."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or SETTINGS.retrieval.embed_dim

    def _slot(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "big")
        # The sign bit makes collisions cancel in expectation instead of
        # accumulating, which matters a great deal at these dimensions.
        return raw % self.dim, 1.0 if (raw >> 63) & 1 else -1.0

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        if not tokens:
            return vec

        # Bigrams let multi-word terms ("world war", "expected calibration
        # error") contribute a feature of their own rather than dissolving.
        features = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

        counts: dict[str, int] = {}
        for feature in features:
            counts[feature] = counts.get(feature, 0) + 1

        for feature, count in counts.items():
            slot, sign = self._slot(feature)
            # Sublinear tf: a term repeated ten times is not ten times as
            # informative as one occurrence of it.
            vec[slot] += sign * (1.0 + float(np.log(count)))

        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 0.0 else vec

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i] = self.embed_one(text)
        return out


def cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row against `query`.

    Both sides are already L2-normalised by `embed_*`, so this is a plain
    matrix-vector product. Kept as a named function so the assumption is
    documented at the one place it could silently break.
    """
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    return matrix @ query
