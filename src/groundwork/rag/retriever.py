"""Fuse the two rankings and hand back a small, ordered, cited context.

Reciprocal rank fusion rather than a weighted score blend, for one practical
reason: BM25 scores and cosine similarities live on incomparable scales, and
any weighting you pick is really a hidden normalisation choice that silently
rots when the corpus changes. RRF only reads *rank*, so it needs no tuning and
no rescaling.

    score(d) = sum over retrievers of 1 / (k + rank_r(d))

The retriever also has to be able to say "I found nothing". That is what makes
abstention possible downstream, and SQuAD v2's 5,945 unanswerable questions
are how it gets measured.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SETTINGS
from .index import HybridIndex


@dataclass
class Retrieved:
    """One passage plus the scores that put it here.

    Keeping the component scores, not just the fused one, is what makes a bad
    retrieval debuggable: it tells you immediately whether the lexical or the
    dense side was responsible.
    """

    doc_id: str
    title: str
    text: str
    rank: int
    fused_score: float
    dense_score: float
    sparse_score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def citation_tag(self) -> str:
        """The label the model is instructed to cite, e.g. [S1]."""
        return f"S{self.rank}"

    def as_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "tag": self.citation_tag,
            "fused_score": round(self.fused_score, 6),
            "dense_score": round(float(self.dense_score), 6),
            "sparse_score": round(float(self.sparse_score), 6),
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "chars": len(self.text),
        }


@dataclass
class RetrievalResult:
    query: str
    passages: list[Retrieved]
    # True when fusion produced nothing above `min_fused_score`. The pipeline
    # abstains rather than calling a model over noise.
    empty: bool = False

    @property
    def top_score(self) -> float:
        return self.passages[0].fused_score if self.passages else 0.0

    @property
    def score_margin(self) -> float:
        """Gap between the best and second-best passage.

        A strong top hit with nothing behind it usually means a precise match.
        A flat distribution across five passages usually means the retriever is
        guessing. This is one of the features the confidence model reads.
        """
        if len(self.passages) < 2:
            return self.top_score
        return self.passages[0].fused_score - self.passages[1].fused_score

    def context_block(self, max_chars: int | None = None) -> str:
        """Render passages as a tagged block for the prompt.

        Tags are what make citations checkable: the output rail can verify that
        every [Sn] in an answer refers to a passage that was actually supplied,
        and the groundedness check can trace claims back to specific text.
        """
        budget = max_chars or SETTINGS.retrieval.max_context_chars
        parts, used = [], 0
        for p in self.passages:
            block = f"[{p.citation_tag}] ({p.title}) {p.text}"
            if used + len(block) > budget and parts:
                break
            parts.append(block)
            used += len(block)
        return "\n\n".join(parts)


class Retriever:
    def __init__(self, index: HybridIndex, config=None) -> None:
        self.index = index
        self.cfg = config or SETTINGS.retrieval

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        mode: str = "hybrid",
    ) -> RetrievalResult:
        """`mode` is one of hybrid | dense | sparse.

        The single-retriever modes exist so the ablation in the eval harness is
        a real code path rather than a claim in a README.
        """
        top_k = top_k or self.cfg.top_k_final
        if len(self.index) == 0:
            return RetrievalResult(query=query, passages=[], empty=True)

        dense = self.index.dense_scores(query)
        sparse = self.index.sparse_scores(query)

        dense_rank = self._rank_map(dense, self.cfg.top_k_dense) if mode in ("hybrid", "dense") else {}
        sparse_rank = self._rank_map(sparse, self.cfg.top_k_sparse) if mode in ("hybrid", "sparse") else {}

        fused: dict[int, float] = {}
        for ranks, weight in (
            (dense_rank, self.cfg.dense_weight),
            (sparse_rank, self.cfg.sparse_weight),
        ):
            for doc_idx, rank in ranks.items():
                fused[doc_idx] = fused.get(doc_idx, 0.0) + weight / (self.cfg.rrf_k + rank)

        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
        ordered = [(i, s) for i, s in ordered if s >= self.cfg.min_fused_score]

        passages = [
            Retrieved(
                doc_id=self.index.documents[doc_idx].doc_id,
                title=self.index.documents[doc_idx].title,
                text=self.index.documents[doc_idx].text,
                rank=position + 1,
                fused_score=score,
                dense_score=float(dense[doc_idx]),
                sparse_score=float(sparse[doc_idx]),
                dense_rank=dense_rank.get(doc_idx),
                sparse_rank=sparse_rank.get(doc_idx),
            )
            for position, (doc_idx, score) in enumerate(ordered)
        ]
        return RetrievalResult(query=query, passages=passages, empty=not passages)

    def rank_all(self, query: str, mode: str = "hybrid", depth: int = 20) -> list[str]:
        """Ranked doc IDs to `depth`, for recall@k and nDCG@k."""
        result = self.retrieve(query, top_k=depth, mode=mode)
        return [p.doc_id for p in result.passages]

    @staticmethod
    def _rank_map(scores: np.ndarray, k: int) -> dict[int, int]:
        """Top-k document indices mapped to their 1-based rank.

        argpartition rather than a full argsort: at 1,204 documents the
        difference is noise, but this is the line that stops being free at a
        million, and getting it right here means the shape of the code does not
        have to change later.
        """
        if scores.size == 0:
            return {}
        k = min(k, scores.size)
        candidates = np.argpartition(-scores, k - 1)[:k]
        ordered = candidates[np.argsort(-scores[candidates], kind="stable")]
        return {int(idx): rank for rank, idx in enumerate(ordered, start=1) if scores[idx] > 0}
