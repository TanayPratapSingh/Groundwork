"""The gateway: one chokepoint every model call goes through.

Why a chokepoint at all. Without one, model calls happen from wherever
somebody needed one, and the answers to "what does this cost per tenant",
"which model served that response", "can we roll back the model we shipped
yesterday", and "who called this at 3am" all become archaeology. Routing every
call through a single object makes each of those a property of the system
rather than an investigation.

Responsibilities, in the order they execute:

    authorise -> estimate cost -> check budget and rate limit -> cache lookup
    -> select tier -> call with retries -> fall back on failure -> record
    spend -> cache store -> audit

The ordering is deliberate. Cost estimation precedes the cache lookup so a
budget-exhausted tenant is refused consistently rather than being kept alive
by cache hits. Auditing happens last so it records what actually occurred,
including the failures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import DEFAULT_TIER, MODELS, SETTINGS, ModelSpec
from ..security.audit import AuditLog
from ..security.authz import KeyStore, Principal
from .budget import BudgetExceeded, BudgetManager, RateLimited, TierNotAllowed
from .cache import ResponseCache
from .providers import Completion, Provider, ProviderError, fingerprint, get_provider


class GatewayError(RuntimeError):
    """Request could not be served. Carries an HTTP-ish status for the API."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class GatewayResponse:
    """A completion plus everything you would want on the invoice and in the trace."""

    completion: Completion
    tier: str
    tenant: str
    cache: str = "miss"          # exact | semantic | miss
    cost_usd: float = 0.0
    attempts: int = 1
    fell_back_from: str | None = None
    total_latency_ms: float = 0.0

    @property
    def text(self) -> str:
        return self.completion.text

    def as_dict(self) -> dict:
        return {
            "tenant": self.tenant,
            "tier": self.tier,
            "model": self.completion.model_id,
            "cache": self.cache,
            "cost_usd": round(self.cost_usd, 6),
            "attempts": self.attempts,
            "fell_back_from": self.fell_back_from,
            "input_tokens": self.completion.input_tokens,
            "output_tokens": self.completion.output_tokens,
            "cached_input_tokens": self.completion.cached_input_tokens,
            "stop_reason": self.completion.stop_reason,
            "latency_ms": round(self.total_latency_ms, 2),
        }


class Gateway:
    def __init__(
        self,
        provider: Provider | None = None,
        cache: ResponseCache | None = None,
        budgets: BudgetManager | None = None,
        keystore: KeyStore | None = None,
        audit: AuditLog | None = None,
        enable_cache: bool = True,
    ) -> None:
        self.cfg = SETTINGS.gateway
        self.provider = provider or get_provider()
        self.cache = cache or ResponseCache()
        self.budgets = budgets or BudgetManager()
        self.keystore = keystore if keystore is not None else KeyStore.from_env()
        self.audit = audit or AuditLog()
        self.enable_cache = enable_cache

    # -- auth -------------------------------------------------------------

    def authenticate(self, api_key: str | None) -> Principal:
        from ..security.authz import AuthError

        try:
            return self.keystore.authenticate(api_key)
        except AuthError as exc:
            # The audit entry records the attempt; the caller gets a generic
            # message. Telling an unauthenticated caller *why* they failed
            # turns the endpoint into a key-validity oracle.
            self.audit.append("auth", tenant="unknown", outcome="denied", reason=str(exc))
            raise GatewayError("unauthorized", status=401) from exc

    # -- main path --------------------------------------------------------

    def complete(
        self,
        *,
        system: str,
        messages: list[dict],
        tenant: str,
        tier: str = DEFAULT_TIER,
        cache_text: str | None = None,
        max_tokens: int | None = None,
        allow_fallback: bool = True,
    ) -> GatewayResponse:
        started = time.perf_counter()

        if tier not in MODELS:
            raise GatewayError(f"unknown tier '{tier}'", status=400)
        spec = MODELS[tier]

        # 1. Estimate before spending. Pessimistic on purpose: assume the full
        #    output budget is used, so the cap is never crossed by surprise.
        try:
            input_tokens = self.provider.count_tokens(system=system, messages=messages, spec=spec)
        except Exception:
            input_tokens = sum(len(m["content"]) for m in messages) // 4
        estimate = self.budgets.estimate(spec, input_tokens, max_tokens or spec.max_output_tokens)

        # 2. Entitlement, rate limit, and budget.
        try:
            self.budgets.check(tenant, tier, estimate)
        except TierNotAllowed as exc:
            self.audit.append("call", tenant, "denied", reason="tier", tier=tier)
            raise GatewayError(str(exc), status=403) from exc
        except RateLimited as exc:
            self.audit.append("call", tenant, "denied", reason="rate_limit", tier=tier)
            raise GatewayError(str(exc), status=429) from exc
        except BudgetExceeded as exc:
            downgraded = self._affordable_tier(tenant, spec, input_tokens, max_tokens) if allow_fallback else None
            if downgraded is None:
                self.audit.append("call", tenant, "denied", reason="budget", tier=tier)
                raise GatewayError(str(exc), status=402) from exc
            self.audit.append("call", tenant, "downgraded", reason="budget",
                              requested=tier, served=downgraded)
            response = self.complete(
                system=system, messages=messages, tenant=tenant, tier=downgraded,
                cache_text=cache_text, max_tokens=max_tokens, allow_fallback=False,
            )
            # Record what was asked for, not just what was served. Without
            # this the recursive call reports a clean `cheap` request and the
            # downgrade becomes invisible to everything downstream: the
            # response metadata, the trace, and anyone asking why quality
            # dropped for one tenant on one afternoon.
            response.fell_back_from = tier
            return response

        # 3. Cache. Keyed on the full prompt, scoped to the tenant.
        key = fingerprint(system, messages, spec)
        if self.enable_cache:
            cached, kind = self.cache.get(key, semantic_text=cache_text, tenant=tenant)
            if cached is not None:
                self.audit.append("call", tenant, "cache_hit", tier=tier, kind=kind)
                return GatewayResponse(
                    completion=cached, tier=tier, tenant=tenant, cache=kind, cost_usd=0.0,
                    total_latency_ms=(time.perf_counter() - started) * 1000,
                )

        # 4. Call, with retries and then tier fallback.
        chain = self._fallback_chain(tier) if allow_fallback else [tier]
        last_error: Exception | None = None

        for index, candidate in enumerate(chain):
            candidate_spec = MODELS[candidate]
            if index > 0 and not self.budgets.can_afford(
                tenant, candidate, self.budgets.estimate(candidate_spec, input_tokens, max_tokens)
            ):
                continue
            try:
                completion, attempts = self._call_with_retries(
                    system=system, messages=messages, spec=candidate_spec, max_tokens=max_tokens
                )
            except ProviderError as exc:
                last_error = exc
                self.audit.append("call", tenant, "error", tier=candidate, reason=str(exc))
                continue

            cost = self.budgets.record(
                tenant, candidate_spec, completion.input_tokens,
                completion.output_tokens, completion.cached_input_tokens,
            )
            response = GatewayResponse(
                completion=completion, tier=candidate, tenant=tenant,
                cache="miss", cost_usd=cost, attempts=attempts,
                fell_back_from=tier if candidate != tier else None,
                total_latency_ms=(time.perf_counter() - started) * 1000,
            )

            # Refusals and truncations are never cached: both are transient
            # states of the request, and serving one from cache turns a
            # one-off into a permanent wrong answer for that prompt.
            if self.enable_cache and not completion.refused and not completion.truncated:
                self.cache.put(key, completion, semantic_text=cache_text, tenant=tenant)

            self.audit.append(
                "call", tenant, "refused" if completion.refused else "ok",
                tier=candidate, model=completion.model_id, cost_usd=round(cost, 6),
                input_tokens=completion.input_tokens, output_tokens=completion.output_tokens,
                cached_input_tokens=completion.cached_input_tokens, attempts=attempts,
            )
            return response

        raise GatewayError(f"all tiers failed: {last_error}", status=503)

    # -- internals --------------------------------------------------------

    def _call_with_retries(self, *, system: str, messages: list[dict],
                           spec: ModelSpec, max_tokens: int | None) -> tuple[Completion, int]:
        last: ProviderError | None = None
        for attempt in range(1, self.cfg.max_retries + 2):
            try:
                return self.provider.complete(
                    system=system, messages=messages, spec=spec, max_tokens=max_tokens
                ), attempt
            except ProviderError as exc:
                last = exc
                if not exc.retryable:
                    # A 400 will fail identically on retry. Retrying it burns
                    # latency and the tenant's rate-limit budget for nothing.
                    raise
                if attempt <= self.cfg.max_retries:
                    time.sleep(self.cfg.retry_base_delay_s * (2 ** (attempt - 1)))
        raise last or ProviderError("exhausted retries")

    def _fallback_chain(self, tier: str) -> list[str]:
        """Requested tier first, then progressively cheaper ones."""
        chain = [tier]
        seen = {tier}
        started = tier in self.cfg.fallback_chain
        for candidate in self.cfg.fallback_chain:
            if not started:
                started = candidate == tier
                continue
            if candidate not in seen:
                chain.append(candidate)
                seen.add(candidate)
        return chain

    def _affordable_tier(self, tenant: str, spec: ModelSpec, input_tokens: int,
                         max_tokens: int | None) -> str | None:
        for candidate in self.cfg.fallback_chain:
            candidate_spec = MODELS[candidate]
            if candidate_spec.input_usd_per_mtok >= spec.input_usd_per_mtok:
                continue  # only ever downgrade
            estimate = self.budgets.estimate(candidate_spec, input_tokens, max_tokens)
            if self.budgets.can_afford(tenant, candidate, estimate):
                return candidate
        return None

    # -- introspection ----------------------------------------------------

    def stats(self) -> dict:
        return {
            "provider": self.provider.name,
            "cache": self.cache.stats.as_dict(),
            "tenants": self.budgets.report(),
        }
