# Architecture

What the pieces are, why they are separate, and where the boundaries fall.

---

## 1. The request path

```
question
   │
   ├─> [ input rail ]        length, injection, jailbreak, PII
   │        │                                          └──> BLOCKED
   │        v
   ├─> [ retrieve ]          BM25 (+ optional dense), RRF fusion
   │        │                                          └──> ABSTAIN (nothing found)
   │        v
   ├─> [ context rail ]      scan retrieved passages for planted instructions
   │        │                                          └──> drop poisoned passages
   │        v
   ├─> [ gateway ]           authenticate, estimate, budget, cache, route, retry
   │        │                                          └──> 401 / 402 / 403 / 429 / 503
   │        v
   ├─> [ generate ]          system prompt (cached) + context + question
   │        │
   │        v
   ├─> [ output rail ]       citation validity, groundedness, PII egress
   │        │                                          └──> ABSTAIN
   │        v
   └─> [ decision ]          calibrated confidence vs operating threshold
            │                                          └──> ABSTAIN
            v
       answer + citations + confidence + provenance
```

Every stage can stop the request, and each stops it for a different reason.
Enumerating them is most of the value of writing the pipeline as one object
rather than a chain of calls scattered across a handler:

| Stage | Stops because |
|---|---|
| input rail | the question is hostile or oversized |
| retrieval | nothing in the corpus is relevant |
| context rail | the passages found contain planted instructions |
| gateway | unauthenticated, unentitled, rate-limited, or out of budget |
| output rail | the answer is ungrounded or cites passages that do not exist |
| decision | the answer is grounded but confidence is below the bar |

The last one is the interesting case. **An answer can pass every safety check
and still be one the system should not assert.** Separating "safe" from
"confident" is what lets the product hedge instead of choosing between
bluffing and refusing outright.

---

## 2. Module boundaries

```
src/groundwork/
  config.py         every threshold, quota, price, and gate floor
  rag/              corpus, embed, index, retriever, prompts, confidence, pipeline
  guardrails/       classifier, detectors, rails
  gateway/          providers, cache, budget, router, server
  evaluation/       datasets, metrics, calibration, guardrail_eval, runner, report
  security/         authz, audit, redaction
  telemetry/        logging, drift
```

Two rules keep this from tangling.

**`config.py` holds every number.** No magic constants in logic modules. A
reviewer should be able to read the whole policy surface in one file rather
than grepping for thresholds, and model IDs appear exactly once so a model
swap is a one-line diff plus an eval run.

**`rag/pipeline.py` is the composition root, and is not exported from
`rag/__init__.py`.** This was not a style preference; it was a circular import
that turned out to be telling the truth. `guardrails` needs prompt constants
from `rag`, and `pipeline` needs `guardrails`, so re-exporting `pipeline` from
the `rag` package created a cycle. The fix was to recognise that a low-level
retrieval package should not depend on the high-level object that consumes it.
Import it explicitly:

```python
from groundwork.rag.pipeline import RagPipeline
```

---

## 3. Dependency direction

```
   evaluation ───────────────┐
        │                    │
        v                    v
    pipeline ──> guardrails ──> security
        │             │
        v             v
    gateway  ───> providers
        │
        v
       rag  ──> config
```

Nothing in `rag/`, `security/`, or `telemetry/` imports from `evaluation/`.
The evaluation harness is a consumer of the system, never a participant in it,
which is what lets the same code path serve a request and be measured.

The one place this is nearly violated is worth noting: `rag/confidence.py`
contains a miniature ECE implementation rather than importing the full one
from `evaluation/calibration.py`. `confidence` ships in the request path, and
pulling the evaluation package into it would drag numpy-heavy report machinery
into every process that answers a question.

---

## 4. Where the extension points are

The parts most likely to be replaced, and what replacing them costs.

| Component | Now | Swap to | Blast radius |
|---|---|---|---|
| Embedder | hashing bag-of-words | any sentence embedder | two methods: `embed_one`, `embed_many` |
| Reader | offline extractive | Claude via `AnthropicProvider` | set `ANTHROPIC_API_KEY` |
| Index | in-process BM25 + numpy | a vector database | `HybridIndex` interface |
| Cache | in-process dict | Redis | `ResponseCache` interface, already the Redis subset |
| Budgets | in-process ledger | shared counter store | `BudgetManager` interface |
| Detectors | hashed n-grams | a fine-tuned classifier | `Detection` return type |

The evaluation harness is what makes any of these swaps arguable rather than
asserted. `scripts/run_eval.py` reports all three retrieval modes on every
run, so the day a real embedder makes hybrid fusion win is a number in the
report, not a hunch.

---

## 5. Statefulness

Everything is in-process, deliberately, and this is the honest limit.

The cache, the budget ledgers, and the rate-limit windows all live in memory.
That means one instance. Two replicas would each enforce half the rate limit
and each hold a cold cache.

The interfaces were chosen to be the ones Redis gives you, so the swap is
mechanical rather than a redesign: TTL-based expiry, oldest-first eviction,
and per-tenant key scoping are all Redis semantics already. The audit log is
already append-only on disk and would move to an append-only object store or a
log pipeline unchanged.

What does not need to change: the index is read-only after build, and the
confidence model and detector weights are read-only artefacts. Those replicate
by copying files.
