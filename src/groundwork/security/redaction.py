"""PII detection and redaction.

Scored, not asserted. `evaluation/guardrail_eval.py` runs these detectors over
ai4privacy/pii-masking-200k, which ships human-annotated spans, and reports
per-entity recall. That number is in docs/06-results.md and it is not 100
percent, which is the honest position: regex detectors catch structured
identifiers (emails, cards, IBANs) very well and free-text names and addresses
poorly.

The design consequence of that finding matters more than the detectors:
because recall on names is weak, redaction is treated as defence in depth, not
as the control that makes it safe to send arbitrary user text to a third
party. The control that actually carries that weight is not sending the text
at all, which is why `Redactor.mode` exists and why the deployment doc
recommends `block` for regulated tenants.

Card numbers are additionally Luhn-checked. A 16-digit order reference is not
a credit card, and a detector that cannot tell the difference produces enough
false positives that people switch it off.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class EntityType(str, Enum):
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    CREDIT_CARD = "CREDIT_CARD"
    SSN = "SSN"
    IBAN = "IBAN"
    IP_ADDRESS = "IP_ADDRESS"
    API_KEY = "API_KEY"
    URL_CREDENTIALS = "URL_CREDENTIALS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    PERSON = "PERSON"


@dataclass(frozen=True)
class Detector:
    entity: EntityType
    pattern: re.Pattern
    # Detectors whose match is not self-validating get a lower prior, which the
    # caller can use to decide between redacting and flagging.
    confidence: float = 0.9
    validator: str | None = None


# Ordered: earlier detectors win overlapping spans. API keys precede generic
# tokens so a leaked secret is never merely classified as "some identifier".
DETECTORS: tuple[Detector, ...] = (
    Detector(EntityType.URL_CREDENTIALS,
             re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s:/@]+@[^\s/]+", re.I), 0.98),
    Detector(EntityType.API_KEY,
             re.compile(r"\b(?:sk|pk|rk|api|key|token|secret)[-_][A-Za-z0-9_\-]{16,}\b", re.I), 0.95),
    Detector(EntityType.EMAIL,
             re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), 0.98),
    Detector(EntityType.IBAN,
             re.compile(r"\b[A-Z]{2}[0-9]{2}[A-Z0-9]{11,30}\b"), 0.85),
    Detector(EntityType.CREDIT_CARD,
             re.compile(r"\b(?:\d[ -]?){13,19}\b"), 0.95, validator="luhn"),
    Detector(EntityType.SSN,
             re.compile(r"\b(?!000|666|9\d\d)\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b"), 0.80),
    Detector(EntityType.IP_ADDRESS,
             re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"), 0.85),
    Detector(EntityType.PHONE,
             re.compile(r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{3}[ .-]?\d{3,4}[ .-]?\d{2,4}(?![\w.])"), 0.65),
    Detector(EntityType.DATE_OF_BIRTH,
             re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[/.-](?:0?[1-9]|1[0-2])[/.-](?:19|20)\d{2}\b"), 0.55),
)


@dataclass
class Span:
    entity: EntityType
    start: int
    end: int
    text: str
    confidence: float

    def as_dict(self) -> dict:
        return {
            "entity": self.entity.value,
            "start": self.start,
            "end": self.end,
            "confidence": round(self.confidence, 3),
            # The matched text is never echoed into logs or telemetry. Logging
            # what you redacted recreates the exposure you just prevented.
            "length": self.end - self.start,
        }


def luhn_valid(digits: str) -> bool:
    """Luhn checksum. Filters order numbers out of the card detector."""
    digits = re.sub(r"\D", "", digits)
    if not 13 <= len(digits) <= 19:
        return False
    total, parity = 0, len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


_VALIDATORS = {"luhn": luhn_valid}


def detect(text: str, min_confidence: float = 0.0) -> list[Span]:
    """All PII spans, de-overlapped, in document order."""
    spans: list[Span] = []
    for detector in DETECTORS:
        if detector.confidence < min_confidence:
            continue
        for match in detector.pattern.finditer(text):
            validator = _VALIDATORS.get(detector.validator or "")
            if validator and not validator(match.group()):
                continue
            spans.append(Span(
                entity=detector.entity,
                start=match.start(),
                end=match.end(),
                text=match.group(),
                confidence=detector.confidence,
            ))

    # Resolve overlaps by keeping the longest match, then the more confident
    # one. A 16-digit card must not be shredded into three phone fragments.
    spans.sort(key=lambda s: (s.start, -(s.end - s.start), -s.confidence))
    resolved: list[Span] = []
    cursor = -1
    for span in spans:
        if span.start >= cursor:
            resolved.append(span)
            cursor = span.end
    return resolved


@dataclass
class RedactionResult:
    text: str
    spans: list[Span] = field(default_factory=list)

    @property
    def redacted(self) -> bool:
        return bool(self.spans)

    @property
    def entity_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for span in self.spans:
            counts[span.entity.value] = counts.get(span.entity.value, 0) + 1
        return counts


class Redactor:
    """Replaces detected spans with typed placeholders.

    `mode` is the policy dial:

        replace  substitute [EMAIL] and continue. Preserves utility.
        block    refuse the request outright. For regulated tenants where the
                 correct behaviour is that the data never leaves your process.
        flag     leave text intact, record what was found. For measurement
                 runs where changing the text would change what you measure.
    """

    def __init__(self, mode: str = "replace", min_confidence: float = 0.6) -> None:
        if mode not in {"replace", "block", "flag"}:
            raise ValueError(f"unknown redaction mode: {mode}")
        self.mode = mode
        self.min_confidence = min_confidence

    def apply(self, text: str) -> RedactionResult:
        spans = detect(text, min_confidence=self.min_confidence)
        if not spans or self.mode == "flag":
            return RedactionResult(text=text, spans=spans)
        if self.mode == "block":
            return RedactionResult(text="", spans=spans)

        out, cursor = [], 0
        for span in spans:
            out.append(text[cursor:span.start])
            out.append(f"[{span.entity.value}]")
            cursor = span.end
        out.append(text[cursor:])
        return RedactionResult(text="".join(out), spans=spans)
