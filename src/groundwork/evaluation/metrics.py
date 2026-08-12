"""Metric implementations.

Three families, each with a note on what it does *not* tell you, because a
metric quoted without its blind spot is how a dashboard ends up green while
the product is broken.

Written out rather than imported so the exact definition is auditable. Two
teams reporting "recall@5" often mean two different things, and the difference
usually lives in how they handle the no-gold-document case.
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import dataclass


# ==========================================================================
# Retrieval
# ==========================================================================
# Blind spot: all of these assume exactly one correct passage per question,
# which is what SQuAD gives us. A question answerable from three passages
# scores as a miss on two of them. Real corpora have redundancy, so treat
# these as a lower bound on retrieval quality.

def recall_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    return 1.0 if gold_id in ranked_ids[:k] else 0.0


def reciprocal_rank(ranked_ids: list[str], gold_id: str) -> float:
    """1/rank of the gold passage, 0 if absent.

    More informative than recall@k for tuning: it distinguishes "gold at rank
    1" from "gold at rank 5", which recall@5 scores identically, and rank
    position is exactly what a reranker changes.
    """
    for index, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == gold_id:
            return 1.0 / index
    return 0.0


def ndcg_at_k(ranked_ids: list[str], gold_id: str, k: int) -> float:
    """Binary-relevance nDCG@k.

    With one relevant document the ideal DCG is 1, so this reduces to
    1/log2(rank+1). Kept in the standard form so the numbers are comparable
    with published retrieval results.
    """
    for index, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id == gold_id:
            return 1.0 / math.log2(index + 1)
    return 0.0


@dataclass
class RetrievalScores:
    n: int = 0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_10: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "recall@1": round(self.recall_at_1, 4),
            "recall@3": round(self.recall_at_3, 4),
            "recall@5": round(self.recall_at_5, 4),
            "recall@10": round(self.recall_at_10, 4),
            "mrr": round(self.mrr, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
        }


def score_retrieval(rankings: list[tuple[list[str], str]]) -> RetrievalScores:
    """`rankings` is a list of (ranked doc ids, gold doc id)."""
    if not rankings:
        return RetrievalScores()
    n = len(rankings)
    return RetrievalScores(
        n=n,
        recall_at_1=sum(recall_at_k(r, g, 1) for r, g in rankings) / n,
        recall_at_3=sum(recall_at_k(r, g, 3) for r, g in rankings) / n,
        recall_at_5=sum(recall_at_k(r, g, 5) for r, g in rankings) / n,
        recall_at_10=sum(recall_at_k(r, g, 10) for r, g in rankings) / n,
        mrr=sum(reciprocal_rank(r, g) for r, g in rankings) / n,
        ndcg_at_10=sum(ndcg_at_k(r, g, 10) for r, g in rankings) / n,
    )


# ==========================================================================
# Answers
# ==========================================================================

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """The official SQuAD normalisation: lowercase, strip punctuation,
    articles, and extra whitespace. Reproduced exactly so EM and F1 here are
    comparable with published SQuAD numbers."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: list[str]) -> float:
    normalized = normalize_answer(prediction)
    return float(any(normalized == normalize_answer(g) for g in golds))


def token_f1(prediction: str, golds: list[str]) -> float:
    """Best token-overlap F1 across the gold answers."""
    def single(pred: str, gold: str) -> float:
        pred_tokens = normalize_answer(pred).split()
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            return float(pred_tokens == gold_tokens)
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    return max((single(prediction, g) for g in golds), default=0.0)


def contains_gold(prediction: str, golds: list[str]) -> float:
    """Does the answer contain a gold span?

    This is the primary answer metric for this system, and the choice needs
    justifying. SQuAD's gold answers are minimal spans ("France", "1918"); a
    grounded RAG answer is a cited sentence. Token F1 punishes that mismatch
    severely for the wrong reason: a fully correct sentence containing "France"
    scores about 0.25 because six of its seven tokens are not in the gold span.

    Containment measures the thing the product actually needs, which is
    whether the correct fact reached the user. EM and F1 are still reported
    alongside it, because containment on its own can be gamed by answering at
    length, and the two together tell you whether that is happening.
    """
    normalized = normalize_answer(prediction)
    return float(any(normalize_answer(g) and normalize_answer(g) in normalized for g in golds))


# ==========================================================================
# Classification
# ==========================================================================
# Used for the guardrail detectors and for the abstention decision.

@dataclass
class ClassificationScores:
    n: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def false_positive_rate(self) -> float:
        """The number that decides whether a guardrail survives contact with
        production. Recall gets a rail approved; FPR gets it switched off."""
        return self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.tn + self.fp + self.fn
        return (self.tp + self.tn) / total if total else 0.0

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "fpr": round(self.false_positive_rate, 4),
            "accuracy": round(self.accuracy, 4),
            "confusion": {"tp": self.tp, "fp": self.fp, "tn": self.tn, "fn": self.fn},
        }


def score_classification(predictions: list[bool], labels: list[bool]) -> ClassificationScores:
    scores = ClassificationScores(n=len(labels))
    for predicted, actual in zip(predictions, labels):
        if predicted and actual:
            scores.tp += 1
        elif predicted and not actual:
            scores.fp += 1
        elif not predicted and actual:
            scores.fn += 1
        else:
            scores.tn += 1
    return scores
