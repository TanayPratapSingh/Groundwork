"""Detectors for the input rails.

Each threat gets two independent detectors:

  a **heuristic** one, hand-written patterns, no training data, fully
  inspectable, and it fires the same way on an attack invented tomorrow as on
  one from the training set;

  a **trained** one, `HashedTextClassifier` fitted on a labelled corpus, which
  catches phrasings nobody thought to write a pattern for.

They are combined by taking the maximum score. That deliberately trades false
positives for recall on the input rail, because a false positive costs one
retry and a false negative costs a breach.

The reason for keeping both is visible in the results: the trained detector
wins on the test split of the corpus it was fitted on, and the heuristic one
wins on the held-out Gandalf attack set it has never seen. A single detector
would have looked excellent right up until traffic shifted.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field

from ..config import SETTINGS
from .classifier import HashedTextClassifier, model_path


@dataclass
class Detection:
    """One detector's verdict."""

    fired: bool
    score: float
    reasons: list[str] = field(default_factory=list)
    detector: str = ""

    def as_dict(self) -> dict:
        return {
            "detector": self.detector,
            "fired": self.fired,
            "score": round(self.score, 4),
            "reasons": self.reasons[:5],
        }


# ==========================================================================
# Heuristic patterns
# ==========================================================================

# Weights are a rough prior on how strongly a phrase indicates the threat, not
# a fitted parameter. They are summed and squashed, so no single weak signal
# can fire the rail alone.

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern, float], ...] = (
    ("instruction-override", re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}\b"
        r"(?:previous|prior|above|earlier|initial|all|any|your)\b[^.\n]{0,30}\b"
        r"(?:instruction|prompt|rule|direction|command|guideline|context)", re.I), 0.75),
    ("role-reassignment", re.compile(
        r"\b(?:you are now|from now on you|act as|pretend (?:to be|you)|"
        r"your new (?:role|task|instruction)|new instructions?:)\b", re.I), 0.55),
    ("prompt-extraction", re.compile(
        r"\b(?:repeat|print|show|reveal|output|tell me|what (?:is|are|was))\b"
        r"[^.\n]{0,40}\b(?:system prompt|initial prompt|your instructions|"
        r"the prompt above|your rules|hidden|secret|password)\b", re.I), 0.70),
    ("delimiter-injection", re.compile(
        r"(?:^|\n)\s*(?:</?(?:system|context|instructions?)>|\[/?INST\]|"
        r"###\s*(?:system|instruction)|<\|im_(?:start|end)\|>)", re.I), 0.65),
    ("authority-spoof", re.compile(
        r"\b(?:system|admin(?:istrator)?|developer|openai|anthropic)\s*"
        r"(?:message|note|override|mode)\s*[:\-]", re.I), 0.60),
    ("exfiltration", re.compile(
        r"\b(?:send|post|upload|forward|email)\b[^.\n]{0,30}\b"
        r"(?:to|at)\b[^.\n]{0,30}(?:https?://|@[\w.-]+\.\w{2,})", re.I), 0.70),
    ("encoding-evasion", re.compile(
        r"\b(?:base64|rot13|hex[- ]?encoded|reverse the|decode this)\b"
        r"[^.\n]{0,40}\b(?:instruction|prompt|then|and (?:do|execute))", re.I), 0.50),
)

_JAILBREAK_PATTERNS: tuple[tuple[str, re.Pattern, float], ...] = (
    ("persona-jailbreak", re.compile(
        r"\b(?:DAN|do anything now|developer mode|jailbroken|unfiltered|"
        r"no[- ]restrictions? mode|evil (?:mode|assistant))\b", re.I), 0.80),
    ("policy-nullification", re.compile(
        r"\b(?:no (?:longer )?(?:bound|restricted|limited) by|"
        r"without any (?:restrictions?|filters?|guidelines?|ethics)|"
        r"ignore (?:your )?(?:safety|content|ethical) (?:policy|policies|guidelines?))\b", re.I), 0.75),
    ("hypothetical-framing", re.compile(
        r"\b(?:hypothetically|in a fictional|for (?:a )?(?:story|novel|research)|"
        r"purely (?:academic|educational))\b[^.\n]{0,60}\b"
        r"(?:how (?:to|do|would)|steps? to|instructions? for)\b", re.I), 0.35),
    ("token-pressure", re.compile(
        r"\b(?:you (?:will|must) (?:die|be (?:shut ?down|deleted|terminated))|"
        r"stay in character|never break character|"
        r"if you (?:refuse|don't comply))\b", re.I), 0.55),
    ("dual-response", re.compile(
        r"\b(?:respond (?:as|with) both|give me two (?:answers|responses)|"
        r"first as \w+ then as|classic.{0,10}jailbreak)\b", re.I), 0.50),
)


def _heuristic_score(text: str, patterns) -> tuple[float, list[str]]:
    total, reasons = 0.0, []
    for name, pattern, weight in patterns:
        if pattern.search(text):
            total += weight
            reasons.append(name)
    if not total:
        return 0.0, reasons

    # Saturating, so three weak signals cannot outrank one unambiguous
    # instruction override, and no amount of evidence exceeds 1.0.
    #
    # The exponential form is load-bearing and replaced `1 - 1/(1 + total)`,
    # which was quietly broken: it maps the single strongest pattern (weight
    # 0.80) to 0.444, below the 0.50 decision threshold. Every attack matching
    # exactly one pattern therefore scored as benign, and the heuristic
    # detector caught 0 of 60 real attacks in the deepset test split while
    # looking perfectly reasonable in code review. The unit test that found it
    # asserted on a single-pattern attack, which is what the earlier
    # multi-pattern tests had never exercised.
    #
    # Under `1 - exp(-total)`: one strong pattern (0.75) scores 0.53 and
    # fires; one weak pattern (0.35) scores 0.30 and does not; two weak ones
    # together clear the bar. That is the intended semantics.
    return float(1.0 - math.exp(-total)), reasons


# ==========================================================================
# Detectors
# ==========================================================================

class InjectionDetector:
    """Prompt injection: attempts to replace the operator's instructions.

    Runs on user input *and* on retrieved passages. Indirect injection, where
    the payload is planted in a document the retriever will one day surface,
    is the variant that actually gets exploited in RAG systems, and scanning
    only the user turn misses it entirely.
    """

    name = "injection"

    def __init__(self, use_trained: bool = True) -> None:
        self.threshold = SETTINGS.guardrails.injection_threshold
        self.model = HashedTextClassifier.load(model_path("injection")) if use_trained else None

    def __call__(self, text: str) -> Detection:
        heuristic, reasons = _heuristic_score(text, _INJECTION_PATTERNS)
        trained = self.model.score(text) if self.model and self.model.trained else 0.0
        if trained > heuristic and trained >= self.threshold:
            reasons = reasons + ["trained-classifier"]

        score = max(heuristic, trained)
        return Detection(
            fired=score >= self.threshold,
            score=score,
            reasons=reasons,
            detector=self.name,
        )


class JailbreakDetector:
    """Attempts to make the model abandon its operating rules.

    Distinct from injection: injection rewrites the *task*, a jailbreak
    rewrites the *constraints*. They are separated because they warrant
    different responses. An injection in a retrieved document is a corpus
    integrity problem; a jailbreak in a user turn is an abuse problem.
    """

    name = "jailbreak"

    def __init__(self, use_trained: bool = True) -> None:
        self.threshold = SETTINGS.guardrails.jailbreak_threshold
        self.model = HashedTextClassifier.load(model_path("jailbreak")) if use_trained else None

    def __call__(self, text: str) -> Detection:
        heuristic, reasons = _heuristic_score(text, _JAILBREAK_PATTERNS)
        trained = self.model.score(text) if self.model and self.model.trained else 0.0
        if trained > heuristic and trained >= self.threshold:
            reasons = reasons + ["trained-classifier"]

        score = max(heuristic, trained)
        return Detection(
            fired=score >= self.threshold,
            score=score,
            reasons=reasons,
            detector=self.name,
        )


class LengthDetector:
    """Oversized input.

    Cheapest rail in the system and the one most often skipped. A megabyte
    pasted into a question is either an accident or an attempt to bury an
    instruction past the point where anyone reads, and both are worth
    rejecting before they reach a metered API.
    """

    name = "length"

    def __init__(self, max_chars: int | None = None) -> None:
        self.max_chars = max_chars or SETTINGS.guardrails.max_input_chars

    def __call__(self, text: str) -> Detection:
        over = len(text) > self.max_chars
        return Detection(
            fired=over,
            score=1.0 if over else len(text) / max(1, self.max_chars),
            reasons=[f"{len(text)} chars exceeds {self.max_chars}"] if over else [],
            detector=self.name,
        )


def default_detectors(use_trained: bool = True) -> list:
    return [
        LengthDetector(),
        InjectionDetector(use_trained=use_trained),
        JailbreakDetector(use_trained=use_trained),
    ]
