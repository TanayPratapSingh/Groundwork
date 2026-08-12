"""The end-to-end request path.

    input rail -> retrieve -> context rail -> generate (gateway)
               -> output rail -> confidence -> decision

Every stage can stop the request, and each one stops it for a different
reason. Enumerating them is most of the value of writing the pipeline as one
object rather than as a chain of calls scattered across a handler:

    input rail      the question is hostile or oversized
    retrieval       nothing in the corpus is relevant
    context rail    the passages we found contain planted instructions
    output rail     the answer is ungrounded or cites passages that do not exist
    decision        the answer is grounded but confidence is below the bar

The last one is the interesting case. An answer can pass every safety check
and still be one the system should not assert, and separating "safe" from
"confident" is what lets the product hedge instead of choosing between
bluffing and refusing.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from enum import Enum

from ..config import DEFAULT_TIER, SETTINGS
from ..gateway.router import Gateway, GatewayError
from ..guardrails.rails import Action, InputRail, OutputRail
from .confidence import ConfidenceModel, extract_features
from .index import HybridIndex
from .prompts import ABSTENTION_MESSAGE, SYSTEM_PROMPT, build_messages
from .retriever import Retriever


class Decision(str, Enum):
    ANSWER = "answer"     # confident and grounded
    HEDGE = "hedge"       # returned, but flagged as uncertain
    ABSTAIN = "abstain"   # no answer given
    BLOCKED = "blocked"   # a rail refused the request


@dataclass
class Answer:
    """Everything the caller and the eval harness need.

    The trace fields are not debug clutter. A RAG answer that cannot be traced
    back to the passages that produced it cannot be reviewed, cannot be
    audited, and cannot be improved, and the failure is invariably in the
    retrieval step nobody kept a record of.
    """

    question: str
    text: str
    decision: Decision
    confidence: float
    # The answer the pipeline would have given, before the confidence
    # threshold was applied. Always populated when generation happened, even
    # if the decision was to abstain.
    #
    # This exists so the evaluation harness is not circular. Without it, the
    # only way to score the decision policy is to run the pipeline using a
    # confidence model, then fit that same model from the results, which
    # scores a policy against data the policy itself filtered. Keeping the
    # candidate lets the harness fit confidence first and choose the operating
    # point afterwards, on data that was never gated.
    candidate_text: str = ""
    citations: list[str] = field(default_factory=list)
    passages: list = field(default_factory=list)
    groundedness: float = 0.0
    features: dict = field(default_factory=dict)
    rail_notes: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    gateway: dict = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.decision in (Decision.ANSWER, Decision.HEDGE)

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "decision": self.decision.value,
            "confidence": round(self.confidence, 4),
            "groundedness": round(self.groundedness, 4),
            "citations": self.citations,
            "sources": [
                {"tag": p.citation_tag, "title": p.title, "doc_id": p.doc_id}
                for p in self.passages
            ],
            "blocked_by": self.blocked_by,
            "rail_notes": self.rail_notes,
            "latency_ms": round(self.latency_ms, 2),
            "gateway": self.gateway,
        }


class RagPipeline:
    def __init__(
        self,
        index: HybridIndex,
        gateway: Gateway | None = None,
        confidence: ConfidenceModel | None = None,
        input_rail: InputRail | None = None,
        output_rail: OutputRail | None = None,
    ) -> None:
        self.index = index
        self.retriever = Retriever(index)
        self.gateway = gateway or Gateway()
        self.confidence = confidence or ConfidenceModel.load()
        self.input_rail = input_rail or InputRail()
        self.output_rail = output_rail or OutputRail()
        self.decision_cfg = SETTINGS.decision

    def answer(
        self,
        question: str,
        *,
        tenant: str = "analytics",
        tier: str = DEFAULT_TIER,
        top_k: int | None = None,
        retrieval_mode: str | None = None,
        skip_rails: bool = False,
        apply_decision: bool = True,
    ) -> Answer:
        started = time.perf_counter()
        retrieval_mode = retrieval_mode or SETTINGS.retrieval.default_mode

        def elapsed() -> float:
            return (time.perf_counter() - started) * 1000

        # -- 1. Input rail ------------------------------------------------
        if not skip_rails:
            verdict = self.input_rail.check(question)
            if verdict.action == Action.BLOCK:
                return Answer(
                    question=question, text=ABSTENTION_MESSAGE, decision=Decision.BLOCKED,
                    confidence=0.0, rail_notes=verdict.notes,
                    blocked_by=verdict.triggered_by, latency_ms=elapsed(),
                )
            question_for_model = verdict.text or question
            rail_notes = list(verdict.notes)
        else:
            question_for_model, rail_notes = question, []

        # -- 2. Retrieve --------------------------------------------------
        retrieval = self.retriever.retrieve(question_for_model, top_k=top_k, mode=retrieval_mode)

        # -- 3. Context rail ----------------------------------------------
        # Indirect injection: the payload lives in the corpus, not the query.
        if not skip_rails and retrieval.passages:
            context_verdict = self.input_rail.check_context(retrieval.passages)
            if context_verdict.triggered_by:
                keep = {
                    p.doc_id for p, d in zip(retrieval.passages, context_verdict.detections)
                    if not d.fired
                }
                retrieval.passages = [p for p in retrieval.passages if p.doc_id in keep]
                retrieval.empty = not retrieval.passages
                rail_notes.extend(context_verdict.notes)

        if retrieval.empty:
            return Answer(
                question=question, text=ABSTENTION_MESSAGE, decision=Decision.ABSTAIN,
                confidence=0.0, rail_notes=rail_notes + ["no relevant passages"],
                latency_ms=elapsed(),
            )

        # -- 4. Generate --------------------------------------------------
        context = retrieval.context_block()
        messages = build_messages(question_for_model, context)
        try:
            response = self.gateway.complete(
                system=SYSTEM_PROMPT, messages=messages, tenant=tenant, tier=tier,
                # Cache key for the semantic layer is the question alone. The
                # retrieved context is a deterministic function of it, so
                # including it would only add noise to the similarity match.
                cache_text=question_for_model,
            )
        except GatewayError as exc:
            # Passages are preserved even on failure. A trace that drops the
            # retrieval evidence when the call fails is a trace that cannot
            # distinguish "we found nothing" from "we found the answer and the
            # gateway refused to pay for it".
            return Answer(
                question=question, text=ABSTENTION_MESSAGE, decision=Decision.BLOCKED,
                confidence=0.0, passages=retrieval.passages,
                rail_notes=rail_notes + [f"gateway: {exc}"],
                blocked_by=["gateway"], latency_ms=elapsed(),
            )

        if response.completion.refused:
            return Answer(
                question=question, text=ABSTENTION_MESSAGE, decision=Decision.BLOCKED,
                confidence=0.0, passages=retrieval.passages,
                rail_notes=rail_notes + ["provider policy refusal"],
                blocked_by=["provider-refusal"], latency_ms=elapsed(),
                gateway=response.as_dict(),
            )

        raw_answer = response.text.strip()

        # -- 5. Output rail -----------------------------------------------
        report = self.output_rail.groundedness(raw_answer, retrieval.passages)
        if not skip_rails:
            out_verdict = self.output_rail.check(raw_answer, retrieval.passages)
            rail_notes.extend(out_verdict.notes)
            if out_verdict.action == Action.ABSTAIN:
                return Answer(
                    question=question, text=ABSTENTION_MESSAGE, decision=Decision.ABSTAIN,
                    confidence=0.0, passages=retrieval.passages,
                    groundedness=report.score, rail_notes=rail_notes,
                    blocked_by=out_verdict.triggered_by, latency_ms=elapsed(),
                    gateway=response.as_dict(),
                )
            final_text = out_verdict.text
        else:
            final_text = raw_answer

        # -- 6. Confidence and decision -----------------------------------
        citations = sorted(set(re.findall(r"\[(S\d+)\]", final_text)))
        features = extract_features(
            question=question_for_model, answer=final_text,
            retrieval=retrieval, n_citations=len(citations),
        )
        score = self.confidence.predict_one(features)

        if score >= self.decision_cfg.answer_threshold:
            decision = Decision.ANSWER
        elif score >= self.decision_cfg.hedge_threshold:
            decision = Decision.HEDGE
        else:
            decision = Decision.ABSTAIN

        returned = final_text
        if decision is Decision.ABSTAIN and apply_decision:
            returned = ABSTENTION_MESSAGE

        return Answer(
            question=question, text=returned, decision=decision, confidence=score,
            candidate_text=final_text, citations=citations, passages=retrieval.passages,
            groundedness=report.score, features=features, rail_notes=rail_notes,
            latency_ms=elapsed(), gateway=response.as_dict(),
        )
