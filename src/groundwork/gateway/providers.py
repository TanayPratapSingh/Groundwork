"""Provider adapters behind one interface.

    AnthropicProvider   real Claude calls, used when ANTHROPIC_API_KEY is set
    OfflineProvider     deterministic extractive answers, used otherwise

The offline provider is not a mock that returns a canned string. It performs
real extractive answering over the retrieved passages and emits citations in
the format the prompt specifies, so the guardrails, the confidence model, the
metrics, and the release gate all exercise genuine logic with no network and
no spend.

That is a deliberate design constraint, not a convenience. An eval suite you
can only run with a budget is an eval suite that gets run once before launch
and never again.
"""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from ..config import ModelSpec


class ProviderError(RuntimeError):
    """Upstream or transport failure. The router decides whether to retry."""

    def __init__(self, message: str, *, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


@dataclass
class Completion:
    """Normalised result of one model call."""

    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int = 0
    stop_reason: str = "end_turn"
    refusal_category: str | None = None
    latency_ms: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def truncated(self) -> bool:
        return self.stop_reason == "max_tokens"


class Provider(Protocol):
    name: str

    def complete(self, *, system: str, messages: list[dict], spec: ModelSpec,
                 max_tokens: int | None = None) -> Completion: ...

    def count_tokens(self, *, system: str, messages: list[dict], spec: ModelSpec) -> int: ...


# ==========================================================================
# Offline
# ==========================================================================

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_PASSAGE_RE = re.compile(r"\[(S\d+)\]\s*(?:\([^)]*\)\s*)?(.*?)(?=\n\n\[S\d+\]|\Z)", re.S)


def _approx_tokens(text: str) -> int:
    """Character-ratio token estimate, offline path only.

    Deliberately crude and never used for anything that must be right. Real
    budgets and context fitting go through the provider's own count_tokens,
    because ratio estimates are off by 15 to 20 percent on English prose and
    far worse on code or non-Latin scripts.
    """
    return max(1, len(text) // 4)


class OfflineProvider:
    """Deterministic extractive answerer.

    Scores each sentence of the retrieved context by term overlap with the
    question, returns the best one or two with their citation tags, and emits
    the abstention token when nothing clears the bar. Same input, same output,
    every time, which is what makes a regression gate meaningful.
    """

    name = "offline"

    # Below this overlap the best available sentence is not actually about the
    # question, and the honest response is the abstention token. Set from the
    # answerable/unanswerable split rather than by taste.
    MIN_OVERLAP = 0.34

    def complete(self, *, system: str, messages: list[dict], spec: ModelSpec,
                 max_tokens: int | None = None) -> Completion:
        started = time.perf_counter()
        user = messages[-1]["content"] if messages else ""
        question, context = self._split(user)
        answer = self._extract(question, context)

        return Completion(
            text=answer,
            model_id=f"offline:{spec.model_id}",
            input_tokens=self.count_tokens(system=system, messages=messages, spec=spec),
            output_tokens=_approx_tokens(answer),
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def count_tokens(self, *, system: str, messages: list[dict], spec: ModelSpec) -> int:
        return _approx_tokens(system) + sum(_approx_tokens(m["content"]) for m in messages)

    # -- internals --------------------------------------------------------

    @staticmethod
    def _split(user: str) -> tuple[str, str]:
        question = user
        if "<question>" in user:
            question = user.split("<question>", 1)[1].split("</question>", 1)[0]
        context = ""
        if "<context>" in user:
            context = user.split("<context>", 1)[1].split("</context>", 1)[0]
        return question.strip(), context

    def _extract(self, question: str, context: str) -> str:
        from ..rag.embed import tokenize
        from ..rag.prompts import INSUFFICIENT

        passages = _PASSAGE_RE.findall(context)
        if not passages:
            return INSUFFICIENT

        q_terms = set(tokenize(question))
        if not q_terms:
            return INSUFFICIENT

        scored: list[tuple[float, str, str]] = []
        for tag, body in passages:
            for sentence in _SENTENCE_SPLIT.split(body.strip()):
                sentence = sentence.strip()
                if len(sentence) < 20:
                    continue
                s_terms = set(tokenize(sentence))
                if not s_terms:
                    continue
                # Coverage of the *question* by the sentence, with a mild
                # penalty for very long sentences that match by sheer size.
                overlap = len(q_terms & s_terms) / len(q_terms)
                brevity = 1.0 / (1.0 + len(s_terms) / 60.0)
                scored.append((overlap * brevity, sentence, tag))

        if not scored:
            return INSUFFICIENT

        scored.sort(key=lambda row: (-row[0], row[1]))
        if scored[0][0] < self.MIN_OVERLAP * (1.0 / (1.0 + 25 / 60.0)):
            return INSUFFICIENT

        # Take the best sentence, and a second only if it is genuinely
        # comparable and from the same passage.
        #
        # Without this the reader appended the runner-up unconditionally, and
        # the demo made the failure obvious: a question about Europe's largest
        # inland port returned the correct Duisburg sentence followed by an
        # unrelated one about the population of Fresno. Both were properly
        # cited and perfectly grounded, which is precisely why it matters:
        # citation and groundedness checks pass happily on a true statement
        # that has nothing to do with the question. Relevance is a separate
        # property and has to be enforced separately.
        best_score, best_sentence, best_tag = scored[0]
        picked = [f"{best_sentence.rstrip('.')} [{best_tag}]."]

        for score, sentence, tag in scored[1:]:
            if sentence == best_sentence:
                continue
            if tag == best_tag and score >= best_score * 0.6:
                picked.append(f"{sentence.rstrip('.')} [{tag}].")
            break

        return " ".join(picked)


# ==========================================================================
# Anthropic
# ==========================================================================

class AnthropicProvider:
    """Live Claude calls.

    Three details here are load-bearing and easy to get wrong:

    1. The system prompt is sent as a cache-controlled block. It is large and
       byte-stable, so it caches; the volatile retrieved context and question
       sit in the user turn, after the breakpoint. Reversing that costs the
       cache entirely.
    2. `max_tokens` bounds thinking plus visible text together. Thinking is on
       by default on current models, so a budget sized around the answer alone
       truncates mid-sentence.
    3. `stop_reason` is checked before `content` is touched. A policy refusal
       arrives as HTTP 200 with an empty content array, so indexing content[0]
       unconditionally raises on exactly the requests you most want telemetry
       for.
    """

    name = "anthropic"

    def __init__(self) -> None:
        import anthropic  # imported lazily: the offline path needs no dependency

        self._sdk = anthropic
        self._client = anthropic.Anthropic()

    def complete(self, *, system: str, messages: list[dict], spec: ModelSpec,
                 max_tokens: int | None = None) -> Completion:
        started = time.perf_counter()
        try:
            resp = self._client.messages.create(
                model=spec.model_id,
                max_tokens=max_tokens or spec.max_output_tokens,
                system=[{"type": "text", "text": system,
                         "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": spec.effort},
                messages=messages,
            )
        except self._sdk.RateLimitError as exc:
            raise ProviderError(f"{spec.model_id}: rate limited", retryable=True) from exc
        except self._sdk.APIConnectionError as exc:
            raise ProviderError(f"{spec.model_id}: connection failure", retryable=True) from exc
        except self._sdk.APIStatusError as exc:
            # 4xx other than 429 will fail identically on retry; 5xx will not.
            raise ProviderError(
                f"{spec.model_id}: HTTP {exc.status_code}",
                retryable=exc.status_code >= 500,
            ) from exc

        latency = (time.perf_counter() - started) * 1000
        usage = resp.usage
        cached = getattr(usage, "cache_read_input_tokens", 0) or 0

        if resp.stop_reason == "refusal":
            details = getattr(resp, "stop_details", None)
            return Completion(
                text="", model_id=resp.model,
                input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                cached_input_tokens=cached, stop_reason="refusal",
                refusal_category=getattr(details, "category", None),
                latency_ms=latency,
            )

        text = "".join(block.text for block in resp.content if block.type == "text")
        return Completion(
            text=text, model_id=resp.model,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            cached_input_tokens=cached,
            stop_reason=resp.stop_reason or "end_turn",
            latency_ms=latency,
        )

    def count_tokens(self, *, system: str, messages: list[dict], spec: ModelSpec) -> int:
        resp = self._client.messages.count_tokens(
            model=spec.model_id, system=system, messages=messages,
        )
        return resp.input_tokens


# ==========================================================================
# Factory
# ==========================================================================

_PROVIDERS: dict[str, Provider] = {}


def get_provider(name: str | None = None) -> Provider:
    from ..config import SETTINGS

    name = name or SETTINGS.provider
    if name in _PROVIDERS:
        return _PROVIDERS[name]

    provider: Provider
    if name == "anthropic":
        try:
            provider = AnthropicProvider()
        except Exception:
            # Missing SDK or unresolvable credentials. Degrading to offline is
            # correct for a dev machine and wrong for production, so the
            # gateway logs the downgrade at WARNING rather than swallowing it.
            provider = OfflineProvider()
    else:
        provider = OfflineProvider()

    _PROVIDERS[name] = provider
    return provider


def reset_providers() -> None:
    """Drop cached provider instances. Used by tests."""
    _PROVIDERS.clear()


def fingerprint(system: str, messages: list[dict], spec: ModelSpec) -> str:
    """Stable cache key over everything that can change the answer."""
    h = hashlib.sha256()
    h.update(spec.model_id.encode())
    h.update(spec.effort.encode())
    h.update(system.encode())
    for message in messages:
        h.update(message["role"].encode())
        h.update(message["content"].encode())
    return h.hexdigest()[:32]
