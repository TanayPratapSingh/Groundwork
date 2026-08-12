"""Gateway: budgets, rate limits, tier entitlement, caching, fallback."""

from __future__ import annotations

import pytest

from groundwork.config import MODELS, TenantQuota
from groundwork.gateway.budget import (
    BudgetExceeded,
    BudgetManager,
    RateLimited,
    TierNotAllowed,
)
from groundwork.gateway.cache import ResponseCache
from groundwork.gateway.providers import Completion, OfflineProvider, ProviderError, fingerprint
from groundwork.gateway.router import Gateway, GatewayError
from groundwork.security.audit import AuditLog
from groundwork.security.authz import KeyStore


@pytest.fixture
def quotas():
    return {
        "rich": TenantQuota("rich", daily_usd=100.0, requests_per_minute=1000),
        "poor": TenantQuota("poor", daily_usd=0.05, requests_per_minute=1000),
        "slow": TenantQuota("slow", daily_usd=100.0, requests_per_minute=2),
        "cheap-only": TenantQuota("cheap-only", daily_usd=100.0,
                                  requests_per_minute=1000, allowed_tiers=("cheap",)),
    }


@pytest.fixture
def gateway(quotas, tmp_path):
    keystore = KeyStore()
    keystore.add("test-key", "rich")
    return Gateway(
        provider=OfflineProvider(),
        budgets=BudgetManager(quotas),
        keystore=keystore,
        audit=AuditLog(tmp_path / "audit.jsonl"),
    )


def ask(gateway, tenant="rich", tier="balanced", question="what is energy"):
    return gateway.complete(
        system="You answer questions.",
        messages=[{"role": "user", "content":
                   f"<context>\n[S1] (T) Energy is a physical property.\n</context>\n\n"
                   f"<question>\n{question}\n</question>"}],
        tenant=tenant, tier=tier, cache_text=question,
    )


# -- cost model ------------------------------------------------------------

def test_cost_scales_with_tokens():
    spec = MODELS["balanced"]
    assert spec.cost_usd(2000, 1000) == pytest.approx(2000 * 3 / 1e6 + 1000 * 15 / 1e6)


def test_cached_input_is_billed_at_a_tenth():
    spec = MODELS["balanced"]
    assert spec.cost_usd(1000, 0, cached_input_tokens=1000) < spec.cost_usd(1000, 0) / 5


# -- budgets ---------------------------------------------------------------

def test_budget_blocks_when_over_cap(quotas):
    budgets = BudgetManager(quotas)
    with pytest.raises(BudgetExceeded):
        budgets.check("poor", "frontier", estimated_usd=1.0)


def test_rate_limit_trips_after_the_configured_count(quotas):
    budgets = BudgetManager(quotas)
    budgets.check("slow", "cheap", 0.0)
    budgets.check("slow", "cheap", 0.0)
    with pytest.raises(RateLimited):
        budgets.check("slow", "cheap", 0.0)


def test_tier_entitlement_is_enforced(quotas):
    with pytest.raises(TierNotAllowed):
        BudgetManager(quotas).check("cheap-only", "frontier", 0.0)


def test_unknown_tenant_is_rejected(quotas):
    with pytest.raises(TierNotAllowed):
        BudgetManager(quotas).check("ghost", "cheap", 0.0)


def test_can_afford_does_not_consume_a_rate_limit_slot(quotas):
    """The router probes tiers when falling back. Probing must be free, or a
    three-tier fallback burns three of the tenant's requests per minute."""
    budgets = BudgetManager(quotas)
    for _ in range(10):
        budgets.can_afford("slow", "cheap", 0.0)
    budgets.check("slow", "cheap", 0.0)  # still has its allowance


def test_spend_is_recorded(quotas):
    budgets = BudgetManager(quotas)
    cost = budgets.record("rich", MODELS["cheap"], 1000, 500)
    assert cost > 0
    assert budgets.ledger("rich").calls == 1
    assert budgets.ledger("rich").spent_usd == pytest.approx(cost)


# -- cache -----------------------------------------------------------------

def test_cache_returns_an_exact_hit():
    cache = ResponseCache()
    cache.put("k", "value", semantic_text="what is energy", tenant="a")
    assert cache.get("k", tenant="a") == ("value", "exact")


def test_cache_never_crosses_tenants():
    """A security property, not a tuning choice: a cache keyed on prompt text
    alone leaks one tenant's retrieved context to another."""
    cache = ResponseCache()
    cache.put("k", "secret", semantic_text="quarterly revenue", tenant="a")
    value, kind = cache.get("k", semantic_text="quarterly revenue", tenant="b")
    assert value is None and kind == "miss"


def test_semantic_hit_requires_high_similarity():
    cache = ResponseCache(threshold=0.99)
    cache.put("k1", "v", semantic_text="what is photosynthesis", tenant="a")
    value, kind = cache.get("k2", semantic_text="who was napoleon", tenant="a")
    assert value is None and kind == "miss"


def test_expired_entries_are_evicted():
    cache = ResponseCache(ttl_seconds=0)
    cache.put("k", "v", tenant="a")
    assert cache.get("k", tenant="a")[1] == "miss"


def test_fingerprint_changes_with_the_prompt():
    spec = MODELS["cheap"]
    a = fingerprint("sys", [{"role": "user", "content": "a"}], spec)
    b = fingerprint("sys", [{"role": "user", "content": "b"}], spec)
    assert a != b and len(a) == 32


# -- router ----------------------------------------------------------------

def test_a_call_succeeds_and_is_billed(gateway):
    response = ask(gateway)
    assert response.text
    assert response.cost_usd > 0
    assert response.cache == "miss"


def test_the_second_identical_call_hits_the_cache(gateway):
    ask(gateway)
    second = ask(gateway)
    assert second.cache in ("exact", "semantic")
    assert second.cost_usd == 0.0


def test_unauthorised_tier_returns_403(gateway):
    with pytest.raises(GatewayError) as exc:
        ask(gateway, tenant="cheap-only", tier="frontier")
    assert exc.value.status == 403


def test_over_budget_downgrades_rather_than_failing(gateway):
    """Degrade, do not deny. A tenant out of frontier budget is better served
    by a cheaper model than by an error page."""
    response = ask(gateway, tenant="poor", tier="frontier")
    assert response.tier != "frontier"
    assert response.fell_back_from == "frontier"


def test_authentication_rejects_a_bad_key(gateway):
    with pytest.raises(GatewayError) as exc:
        gateway.authenticate("wrong-key")
    assert exc.value.status == 401


def test_retries_stop_for_non_retryable_errors(gateway, monkeypatch):
    calls = {"n": 0}

    def always_400(**kwargs):
        calls["n"] += 1
        raise ProviderError("bad request", retryable=False)

    monkeypatch.setattr(gateway.provider, "complete", always_400)
    with pytest.raises(GatewayError):
        ask(gateway)
    # One attempt per tier in the fallback chain, never a retry within a tier.
    assert calls["n"] <= 3


def test_refusals_are_not_cached(gateway, monkeypatch):
    """A policy refusal is a state of one request. Caching it would make a
    transient refusal permanent for that prompt."""
    refusal = Completion(text="", model_id="m", input_tokens=10, output_tokens=0,
                         stop_reason="refusal")
    monkeypatch.setattr(gateway.provider, "complete", lambda **kw: refusal)
    ask(gateway)
    assert len(gateway.cache) == 0
