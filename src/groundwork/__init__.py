"""Groundwork: the plumbing that turns a RAG demo into a service you can ship.

Four concerns, one codebase, measured on real public datasets:

    rag/          retrieve, rerank, generate with citations, abstain when thin
    guardrails/   input and output rails, each scored on a labelled attack set
    gateway/      auth, routing, caching, budgets, audit for every model call
    evaluation/   retrieval metrics, answer metrics, calibration, release gate

Nothing here generates its own evaluation data. The corpus is Wikipedia
paragraphs from SQuAD v2.0, the questions and gold answers are the SQuAD
crowdworkers', and every guardrail is scored against an independently
published, labelled attack corpus. See data/manifest.json for provenance.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
