# The LLM gateway

One chokepoint that every model call goes through.

---

## 1. Why a chokepoint

Without one, model calls happen from wherever somebody needed one, and four
questions all become archaeology:

- What does this cost, per tenant, per feature?
- Which model produced that response last Tuesday?
- Can we roll back the model we shipped yesterday, today?
- Who called this at 3am, and were they allowed to?

Routing every call through a single object makes each of those a property of
the system rather than an investigation. It is the same argument as putting a
database behind a repository layer, and it is worth about as much.

The concrete payoff in this repo: model IDs appear in exactly one file
(`config.py`). Swapping a model is a one-line diff plus an eval run. Calling
code names a *tier*, never a model.

---

## 2. The request path

```
authorise
  ↓
estimate cost          pessimistically: assume the full output budget
  ↓
check entitlement      is this tenant allowed this tier?
check rate limit       requests per minute
check budget           would this call cross the daily cap?
  ↓
cache lookup           exact, then semantic, scoped to the tenant
  ↓
select tier            requested, or a cheaper one if over budget
  ↓
call with retries      exponential backoff, retryable errors only
  ↓
fall back              step down a tier on failure rather than erroring
  ↓
record spend           actual tokens, including cache reads
  ↓
cache store            unless refused or truncated
  ↓
audit                  what happened, who, what it cost
```

The ordering is deliberate. Cost estimation precedes the cache lookup so a
budget-exhausted tenant is refused consistently instead of being kept alive by
cache hits. Auditing is last so it records what actually happened, including
the failures.

---

## 3. Budgets: check before, not after

Metering after the fact tells you what you spent. Checking before tells you
what you are about to spend, and it is the only version that can stop a
runaway agent loop at 3am.

Pre-flight estimation is deliberately pessimistic: it assumes the full output
budget is consumed. Under-estimating lets a tenant slip past the cap;
over-estimating only makes the last call before the cap arrive slightly early.

```python
@dataclass(frozen=True)
class TenantQuota:
    name: str
    daily_usd: float
    requests_per_minute: int
    allowed_tiers: tuple[str, ...]
```

Three shipped tenants plus one that exists because the harness needed it:

| Tenant | Daily | RPM | Tiers |
|---|---|---|---|
| `analytics` | $25 | 60 | all |
| `support` | $10 | 120 | cheap, balanced |
| `public-demo` | $1 | 10 | cheap |
| `evaluation` | $500 | 100,000 | all |

**The `evaluation` tenant is a design decision, not a bypass.** The first full
eval run failed with 152 of 400 questions refused by the interactive rate
limit. That was the gateway working correctly: batch traffic and interactive
traffic have genuinely different shapes and should not share a bucket. Giving
the harness its own quota keeps every call metered and audited, where a
`skip_budget=True` flag would have created a code path that is never exercised
in production and is therefore never really tested.

One subtlety worth its own test: `can_afford` does not consume a rate-limit
slot. The router probes tiers when falling back, and probing three tiers must
not burn three of the tenant's requests per minute.

---

## 4. Degrade, do not deny

A tenant out of frontier budget is better served by a cheaper model than by an
error page. The router walks a fallback chain, skipping tiers the tenant
cannot afford, and records what was originally asked for:

```python
response.tier            # "cheap"      what served it
response.fell_back_from  # "frontier"   what was asked for
```

Recording the request, not just the result, is load-bearing. Without it the
recursive downgrade reports a clean `cheap` request and the degradation is
invisible to the response metadata, the trace, and anyone later asking why
quality dropped for one tenant on one afternoon. That was a real bug here,
caught by a test asserting on `fell_back_from`.

Statuses returned to the caller:

| Status | Meaning |
|---|---|
| 401 | Authentication failed. Deliberately no detail: telling an unauthenticated caller *why* turns the endpoint into a key-validity oracle |
| 403 | Authenticated, not entitled to that tier |
| 402 | Over budget, and no cheaper tier is affordable |
| 429 | Over rate limit |
| 503 | Every tier failed |

402 and 429 do carry detail, because unlike auth failures they concern the
caller's own quota: telling them is useful and leaks nothing.

---

## 5. Caching

Two layers.

**Exact**: SHA-256 over model ID, effort, system prompt, and every message.
Free and always correct.

**Semantic**: cosine similarity over the question, behind a 0.93 threshold.
Cheap and *usually* correct, which is why it is counted separately in
telemetry. A semantic cache that quietly serves a near-miss is a correctness
bug wearing a performance costume, and the only way to notice is to be able to
see its hit rate on its own.

**Entries never cross tenants.** That is a security boundary, not a tuning
choice: a cache keyed on prompt text alone leaks one tenant's retrieved
context to another the moment their questions rhyme. There is a test for it.

**Refusals and truncations are never cached.** Both are transient states of a
request. Caching one turns a one-off into a permanent wrong answer for that
prompt.

---

## 6. Prompt caching, and the thing that breaks it

Prompt caching is a *prefix match*. Any byte change anywhere in the prefix
invalidates everything after it. Render order is tools, then system, then
messages.

So the system prompt here is byte-stable across every request. Nothing dynamic
is interpolated into it: no timestamp, no tenant name, no retrieved text.
Everything volatile lives in the user turn, after the cache breakpoint.

The usual way teams lose their hit rate is a `datetime.now()` in a system
prompt header. It looks harmless and it invalidates the cache on every single
request. `usage.cache_read_input_tokens` sitting at zero across identical
prefixes is the symptom to watch.

Savings are reported per tenant rather than assumed, because "our costs went
down" is a claim someone will eventually ask you to substantiate:

```python
ledger.cache_savings_usd   # uncached cost minus actual, accumulated
```

---

## 7. Retries

Retryable and non-retryable errors are distinguished at the adapter, not
guessed at the router:

| Error | Retryable |
|---|---|
| Rate limit (429) | yes |
| Connection failure | yes |
| Server error (5xx) | yes |
| Bad request (4xx other than 429) | **no** |

A 400 fails identically on retry. Retrying it burns latency and the tenant's
rate-limit budget for nothing.

---

## 8. Provider adapters

Two implementations behind one interface.

`AnthropicProvider` makes real calls. Three details in it are load-bearing and
easy to get wrong:

1. **The system prompt is sent as a cache-controlled block**, and the volatile
   context sits after it. Reversing that costs the cache entirely.
2. **`max_tokens` bounds thinking plus visible text together.** Thinking is on
   by default on current models, so a budget sized around the answer alone
   truncates mid-sentence.
3. **`stop_reason` is checked before `content` is touched.** A policy refusal
   arrives as HTTP 200 with an empty content array, so indexing `content[0]`
   unconditionally raises on exactly the requests you most want telemetry for.

`OfflineProvider` performs real extractive answering over the retrieved
passages and emits citations in the format the prompt specifies. It is not a
mock returning a canned string: the guardrails, the confidence model, the
metrics, and the release gate all exercise genuine logic with no network and
no spend.

That is a deliberate constraint. **An eval suite you can only run with a
budget is an eval suite that gets run once before launch and never again.**

---

## 9. HTTP surface

FastAPI, optional dependency, `pip install 'groundwork[serve]'`.

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness. Deliberately free of version, tenant, and corpus detail: a health endpoint is the most-scanned URL you expose |
| `GET /ready` | none | Readiness: can this instance actually serve |
| `POST /v1/ask` | `X-API-Key` | The question path |
| `GET /v1/usage` | `X-API-Key` | A tenant's own ledger, and only its own |

The API key is read from a header, never a query parameter. Query strings land
in access logs, proxy logs, browser history, and `Referer` headers.

The index and pipeline are loaded once at startup. Rebuilding an index inside
a handler is the classic way a p99 becomes a p50.
