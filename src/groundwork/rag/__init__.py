"""Retrieval primitives: corpus, embeddings, index, retriever, prompts.

`pipeline` is deliberately not re-exported here. It is the composition root
that wires retrieval to the gateway and the guardrails, and importing it from
this package's `__init__` creates a cycle (guardrails needs prompt constants,
pipeline needs guardrails). Import it explicitly:

    from groundwork.rag.pipeline import RagPipeline

The cycle was worth listening to rather than working around: it was telling us
that a low-level retrieval package should not depend on the high-level object
that consumes it.
"""

from .confidence import ConfidenceModel, extract_features
from .corpus import Document, Question, load_corpus, load_questions
from .embed import HashingEmbedder, tokenize
from .index import BM25, HybridIndex
from .retriever import RetrievalResult, Retrieved, Retriever

__all__ = [
    "BM25",
    "ConfidenceModel",
    "Document",
    "HashingEmbedder",
    "HybridIndex",
    "Question",
    "RetrievalResult",
    "Retrieved",
    "Retriever",
    "extract_features",
    "load_corpus",
    "load_questions",
    "tokenize",
]
