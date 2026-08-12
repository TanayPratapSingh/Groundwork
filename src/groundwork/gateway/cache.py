"""Two-layer response cache: exact, then semantic.

Exact hits are free and always correct. Semantic hits are cheap and *usually*
correct, which is why they sit behind a high similarity threshold and are
counted separately in telemetry. A semantic cache that quietly serves a
near-miss is a correctness bug wearing a performance costume, and the only way
to notice is to be able to see its hit rate independently.

Entries are scoped per tenant. That is a security boundary, not a tuning
choice: a cache keyed on prompt text alone leaks one tenant's retrieved
context to another the moment their questions rhyme.

In-process here; the interface is deliberately the subset Redis gives you, so
the production swap is mechanical.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..config import SETTINGS
from ..rag.embed import HashingEmbedder


@dataclass
class CacheEntry:
    key: str
    value: object
    vector: np.ndarray | None
    created_at: float
    tenant: str
    hits: int = 0


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def lookups(self) -> int:
        return self.exact_hits + self.semantic_hits + self.misses

    @property
    def hit_rate(self) -> float:
        return 0.0 if not self.lookups else (self.exact_hits + self.semantic_hits) / self.lookups

    def as_dict(self) -> dict:
        return {
            "lookups": self.lookups,
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": round(self.hit_rate, 4),
        }


class ResponseCache:
    def __init__(
        self,
        ttl_seconds: int | None = None,
        threshold: float | None = None,
        max_entries: int | None = None,
        embedder: HashingEmbedder | None = None,
    ) -> None:
        cfg = SETTINGS.gateway
        self.ttl = ttl_seconds if ttl_seconds is not None else cfg.cache_ttl_seconds
        self.threshold = threshold if threshold is not None else cfg.semantic_cache_threshold
        self.max_entries = max_entries if max_entries is not None else cfg.cache_max_entries
        self._entries: dict[str, CacheEntry] = {}
        self._embedder = embedder or HashingEmbedder()
        self.stats = CacheStats()

    # -- public -----------------------------------------------------------

    def get(self, key: str, semantic_text: str | None = None,
            tenant: str = "default") -> tuple[object | None, str]:
        """Returns (value, kind) where kind is exact | semantic | miss."""
        self._expire()

        entry = self._entries.get(key)
        if entry is not None and entry.tenant == tenant:
            entry.hits += 1
            self.stats.exact_hits += 1
            return entry.value, "exact"

        if semantic_text:
            hit = self._semantic_lookup(semantic_text, tenant)
            if hit is not None:
                hit.hits += 1
                self.stats.semantic_hits += 1
                return hit.value, "semantic"

        self.stats.misses += 1
        return None, "miss"

    def put(self, key: str, value: object, semantic_text: str | None = None,
            tenant: str = "default") -> None:
        self._entries[key] = CacheEntry(
            key=key,
            value=value,
            vector=self._embedder.embed_one(semantic_text) if semantic_text else None,
            created_at=time.time(),
            tenant=tenant,
        )
        self._evict()

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    # -- internals --------------------------------------------------------

    def _expire(self) -> None:
        cutoff = time.time() - self.ttl
        for key in [k for k, e in self._entries.items() if e.created_at < cutoff]:
            del self._entries[key]
            self.stats.evictions += 1

    def _evict(self) -> None:
        # Oldest-first rather than least-recently-used. LRU needs an access
        # order structure; at this size the difference is not measurable, and
        # Redis with a TTL behaves like this anyway.
        while len(self._entries) > self.max_entries:
            oldest = min(self._entries.values(), key=lambda e: e.created_at)
            del self._entries[oldest.key]
            self.stats.evictions += 1

    def _semantic_lookup(self, text: str, tenant: str) -> CacheEntry | None:
        candidates = [e for e in self._entries.values()
                      if e.vector is not None and e.tenant == tenant]
        if not candidates:
            return None
        query = self._embedder.embed_one(text)
        matrix = np.vstack([e.vector for e in candidates])
        sims = matrix @ query  # both sides L2-normalised, so this is cosine
        best = int(np.argmax(sims))
        return candidates[best] if float(sims[best]) >= self.threshold else None
