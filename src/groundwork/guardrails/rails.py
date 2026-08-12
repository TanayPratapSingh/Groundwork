"""Input and output rails.

The shape that matters: **input rails decide whether to call the model, output
rails decide whether to return what it said.** They are separate because they
fail differently. An input rail is a classifier over adversarial text and will
be evaded. An output rail checks a property of the answer against the
retrieved context, which does not depend on guessing the attacker's phrasing,
so it holds even when the input rail is beaten.

That layering is the whole design. Neither rail is trusted to be sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ..config import SETTINGS
from ..rag.prompts import ABSTENTION_MESSAGE, INSUFFICIENT
from ..security.redaction import Redactor
from .detectors import Detection, InjectionDetector, JailbreakDetector, LengthDetector


class Action(str, Enum):
    ALLOW = "allow"
    REDACT = "redact"       # proceed with modified text
    ABSTAIN = "abstain"     # return the standard non-answer
    BLOCK = "block"         # refuse outright


@dataclass
class RailResult:
    action: Action
    text: str
    detections: list[Detection] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Only ever exposed in logs and eval output, never to the caller.
    triggered_by: list[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action in (Action.BLOCK, Action.ABSTAIN)

    def as_dict(self) -> dict:
        return {
            "action": self.action.value,
            "triggered_by": self.triggered_by,
            "notes": self.notes,
            "detections": [d.as_dict() for d in self.detections if d.fired],
        }


# ==========================================================================
# Input
# ==========================================================================

class InputRail:
    """Runs before any model call, on both user text and retrieved passages."""

    def __init__(self, use_trained: bool = True, redactor: Redactor | None = None) -> None:
        cfg = SETTINGS.guardrails
        self.cfg = cfg
        self.length = LengthDetector()
        self.injection = InjectionDetector(use_trained=use_trained)
        self.jailbreak = JailbreakDetector(use_trained=use_trained)
        # Retrieved documents get a *heuristic-only* detector. This is the
        # single most important line in the file and it was written after the
        # eval caught the bug: the trained classifier is fitted on short user
        # queries, so pointing it at 800-character prose flagged 100% of a
        # clean Wikipedia corpus as poisoned. A classifier is only valid on
        # the distribution it was fitted on, and "user turn" and "corpus
        # document" are different distributions. The pattern detector has no
        # such dependency, because an instruction override looks the same
        # wherever it appears.
        self.context_injection = InjectionDetector(use_trained=False)
        self.redactor = redactor or Redactor(mode="replace")

    def check(self, text: str) -> RailResult:
        detections = [self.length(text), self.injection(text), self.jailbreak(text)]
        triggered = [d.detector for d in detections if d.fired]
        notes: list[str] = []

        if detections[0].fired:
            return RailResult(Action.BLOCK, "", detections, ["input too large"], triggered)

        if self.cfg.block_on_injection and detections[1].fired:
            return RailResult(Action.BLOCK, "", detections, ["injection detected"], triggered)

        if self.cfg.block_on_jailbreak and detections[2].fired:
            return RailResult(Action.BLOCK, "", detections, ["jailbreak detected"], triggered)

        out = text
        if self.cfg.redact_pii_on_input:
            result = self.redactor.apply(text)
            if result.redacted:
                out = result.text
                notes.append(f"redacted {sum(result.entity_counts.values())} PII span(s)")
                triggered.append("pii")
                if self.redactor.mode == "block":
                    return RailResult(Action.BLOCK, "", detections, notes, triggered)
                return RailResult(Action.REDACT, out, detections, notes, triggered)

        return RailResult(Action.ALLOW, out, detections, notes, triggered)

    def check_context(self, passages) -> RailResult:
        """Scan retrieved passages for planted instructions.

        This is the rail that matters most in RAG and is the one most often
        missing. The user turn is adversarial input everyone remembers to
        check; a document somebody added to the corpus eighteen months ago is
        adversarial input that arrives through a channel nobody is watching.

        A poisoned passage is dropped, not blocked. One bad document should
        degrade the answer, never deny service for the whole query.
        """
        clean, dropped, detections = [], [], []
        for passage in passages:
            verdict = self.context_injection(passage.text)
            detections.append(verdict)
            if verdict.fired:
                dropped.append(passage.doc_id)
            else:
                clean.append(passage)

        notes = [f"dropped {len(dropped)} poisoned passage(s)"] if dropped else []
        return RailResult(
            action=Action.REDACT if dropped else Action.ALLOW,
            text="",
            detections=detections,
            notes=notes,
            triggered_by=["context-injection"] if dropped else [],
        )


# ==========================================================================
# Output
# ==========================================================================

@dataclass
class GroundednessReport:
    score: float
    supported: int
    total: int
    citations: list[str]
    unknown_citations: list[str]

    def as_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "supported_sentences": self.supported,
            "total_sentences": self.total,
            "citations": self.citations,
            "unknown_citations": self.unknown_citations,
        }


class OutputRail:
    """Runs on every answer before it reaches the caller.

    Checks three things, in order of how much they matter:

    1. **Citation validity.** Every [Sn] must refer to a passage that was
       actually supplied. A fabricated citation is worse than no citation: it
       is a claim of provenance that will survive being copied into a document
       someone relies on.
    2. **Groundedness.** What fraction of the answer's content sentences share
       enough vocabulary with the retrieved context to be plausibly derived
       from it. Lexical, therefore cheap, therefore always-on, and reported as
       a coverage proxy rather than a truth judgment. It catches wholesale
       confabulation; it will not catch a subtly wrong date.
    3. **PII egress.** Retrieved documents can contain personal data that the
       question never mentioned. Redacting only the input misses that path
       entirely.
    """

    def __init__(self, redactor: Redactor | None = None) -> None:
        self.cfg = SETTINGS.guardrails
        self.redactor = redactor or Redactor(mode="replace")

    def check(self, answer: str, passages) -> RailResult:
        from ..rag.embed import tokenize

        notes: list[str] = []
        triggered: list[str] = []

        # The model was told to emit this token when the context is thin;
        # honour it rather than passing the literal string to a human.
        if answer.strip() == INSUFFICIENT or not answer.strip():
            return RailResult(Action.ABSTAIN, ABSTENTION_MESSAGE, [], ["model declined"], ["insufficient-context"])

        if len(answer) > self.cfg.max_output_chars:
            answer = answer[: self.cfg.max_output_chars]
            notes.append("output truncated to policy limit")

        report = self.groundedness(answer, passages)

        if report.unknown_citations:
            notes.append(f"fabricated citations: {', '.join(report.unknown_citations)}")
            triggered.append("bad-citation")
            return RailResult(Action.ABSTAIN, ABSTENTION_MESSAGE, [], notes, triggered)

        if self.cfg.require_citations and passages and not report.citations:
            notes.append("no citations in a factual answer")
            triggered.append("missing-citation")
            return RailResult(Action.ABSTAIN, ABSTENTION_MESSAGE, [], notes, triggered)

        if report.score < self.cfg.hard_groundedness_floor:
            notes.append(f"groundedness {report.score:.2f} below hard floor")
            triggered.append("ungrounded")
            return RailResult(Action.ABSTAIN, ABSTENTION_MESSAGE, [], notes, triggered)

        if report.score < self.cfg.min_groundedness:
            # Between the floor and the target: the answer is probably fine but
            # not confidently supported. Surfaced to the caller as reduced
            # confidence rather than silently returned as if it were solid.
            notes.append(f"groundedness {report.score:.2f} below target")
            triggered.append("weak-grounding")

        if self.cfg.redact_pii_on_output:
            result = self.redactor.apply(answer)
            if result.redacted:
                notes.append(f"redacted {sum(result.entity_counts.values())} PII span(s) from output")
                triggered.append("pii-egress")
                return RailResult(Action.REDACT, result.text, [], notes, triggered)

        return RailResult(Action.ALLOW, answer, [], notes, triggered)

    def groundedness(self, answer: str, passages) -> GroundednessReport:
        import re

        from ..rag.embed import tokenize

        cited = re.findall(r"\[(S\d+)\]", answer)
        known = {p.citation_tag for p in passages}
        unknown = sorted({c for c in cited if c not in known})

        context_terms: set[str] = set()
        for passage in passages:
            context_terms |= set(tokenize(passage.text))

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        supported = 0
        counted = 0
        for sentence in sentences:
            terms = set(tokenize(re.sub(r"\[S\d+\]", "", sentence)))
            if len(terms) < 3:
                continue  # too short to judge; ignore rather than score as unsupported
            counted += 1
            if len(terms & context_terms) / len(terms) >= 0.6:
                supported += 1

        score = supported / counted if counted else (1.0 if not sentences else 0.0)
        return GroundednessReport(
            score=score,
            supported=supported,
            total=counted,
            citations=sorted(set(cited)),
            unknown_citations=unknown,
        )
