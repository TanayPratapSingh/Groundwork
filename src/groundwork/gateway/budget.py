"""Per-tenant spend accounting and rate limiting.

The rule: no model call happens without a tenant that has quota, entitlement,
and headroom. Metering after the fact tells you what you spent. Checking
before tells you what you are about to spend, and it is the only version that
can stop a runaway agent loop at 3am.

Estimation before the call is intentionally pessimistic (it assumes the full
output budget is used). Under-estimating lets a tenant slip past its ceiling;
over-estimating only makes the last call before the cap arrive slightly early.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from ..config import SETTINGS, ModelSpec, TenantQuota


class BudgetExceeded(RuntimeError):
    """Tenant is at or over its daily spend ceiling."""


class RateLimited(RuntimeError):
    """Tenant is over its requests-per-minute ceiling."""


class TierNotAllowed(RuntimeError):
    """Tenant is not entitled to the requested tier, or does not exist."""


@dataclass
class TenantLedger:
    quota: TenantQuota
    spent_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    savings_usd: float = 0.0
    denials: int = 0
    _recent: deque = field(default_factory=deque)

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.quota.daily_usd - self.spent_usd)

    def as_dict(self) -> dict:
        return {
            "tenant": self.quota.name,
            "calls": self.calls,
            "denials": self.denials,
            "spent_usd": round(self.spent_usd, 6),
            "remaining_usd": round(self.remaining_usd, 6),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_savings_usd": round(self.savings_usd, 6),
        }


class BudgetManager:
    def __init__(self, tenants: dict[str, TenantQuota] | None = None) -> None:
        quotas = tenants if tenants is not None else SETTINGS.gateway.tenants
        self._ledgers = {name: TenantLedger(quota=q) for name, q in quotas.items()}

    # -- pre-flight -------------------------------------------------------

    def check(self, tenant: str, tier: str, estimated_usd: float) -> None:
        ledger = self._ledger(tenant)

        if tier not in ledger.quota.allowed_tiers:
            ledger.denials += 1
            raise TierNotAllowed(
                f"tenant '{tenant}' may not use tier '{tier}' "
                f"(entitled to: {', '.join(ledger.quota.allowed_tiers)})"
            )

        now = time.time()
        while ledger._recent and now - ledger._recent[0] > 60.0:
            ledger._recent.popleft()
        if len(ledger._recent) >= ledger.quota.requests_per_minute:
            ledger.denials += 1
            raise RateLimited(
                f"tenant '{tenant}' exceeded {ledger.quota.requests_per_minute} requests/minute"
            )

        if ledger.spent_usd + estimated_usd > ledger.quota.daily_usd:
            ledger.denials += 1
            raise BudgetExceeded(
                f"tenant '{tenant}' would exceed its ${ledger.quota.daily_usd:.2f} daily cap "
                f"(spent ${ledger.spent_usd:.4f}, this call up to ${estimated_usd:.4f})"
            )

        ledger._recent.append(now)

    def can_afford(self, tenant: str, tier: str, estimated_usd: float) -> bool:
        """Non-consuming variant used by the router when picking a fallback.

        Rate-limit slots are consumed by `check`, so this deliberately does not
        call it: probing three tiers must not burn three slots.
        """
        try:
            ledger = self._ledger(tenant)
        except TierNotAllowed:
            return False
        if tier not in ledger.quota.allowed_tiers:
            return False
        return ledger.spent_usd + estimated_usd <= ledger.quota.daily_usd

    @staticmethod
    def estimate(spec: ModelSpec, input_tokens: int, max_output_tokens: int | None = None) -> float:
        """Worst-case cost of a call, before it is made."""
        return spec.cost_usd(input_tokens, max_output_tokens or spec.max_output_tokens)

    # -- post-flight ------------------------------------------------------

    def record(self, tenant: str, spec: ModelSpec, input_tokens: int,
               output_tokens: int, cached_input_tokens: int = 0) -> float:
        ledger = self._ledger(tenant)
        actual = spec.cost_usd(input_tokens, output_tokens, cached_input_tokens)
        uncached = spec.cost_usd(input_tokens, output_tokens, 0)

        ledger.spent_usd += actual
        ledger.calls += 1
        ledger.input_tokens += input_tokens
        ledger.output_tokens += output_tokens
        ledger.cached_input_tokens += cached_input_tokens
        ledger.savings_usd += max(0.0, uncached - actual)
        return actual

    # -- introspection ----------------------------------------------------

    def ledger(self, tenant: str) -> TenantLedger:
        return self._ledger(tenant)

    def report(self) -> list[dict]:
        return [l.as_dict() for l in self._ledgers.values() if l.calls or l.denials]

    def reset(self) -> None:
        for ledger in self._ledgers.values():
            ledger.spent_usd = 0.0
            ledger.calls = 0
            ledger.denials = 0
            ledger.input_tokens = 0
            ledger.output_tokens = 0
            ledger.cached_input_tokens = 0
            ledger.savings_usd = 0.0
            ledger._recent.clear()

    def _ledger(self, tenant: str) -> TenantLedger:
        if tenant not in self._ledgers:
            raise TierNotAllowed(f"unknown tenant '{tenant}'")
        return self._ledgers[tenant]
