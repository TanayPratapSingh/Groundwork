# Groundwork

A retrieval-augmented question answering system with the parts that usually get
skipped: an evaluation harness that can fail a build, guardrails with published
false-positive rates, and a gateway that meters every model call.

**Every number in this repository is measured on real, public, human-labelled
data.** No corpus was written for the demo, no golden answers were authored to
be easy, and no attack was invented to be caught. `scripts/fetch_data.py`
downloads all five datasets and records their SHA-256 in `data/manifest.json`.

```bash
git clone https://github.com/TanayPratapSingh/groundwork.git
cd groundwork
pip install -r requirements.txt

python3 scripts/fetch_data.py     # ~7 MB of real datasets
python3 scripts/build_index.py    # 0.3s over 1,204 passages
python3 scripts/run_eval.py       # full evaluation + release gate
python3 scripts/demo.py           # walk the system end to end
```

No API key is required. With none set, the stack runs on a deterministic
offline reader, and every metric below is reproducible on a laptop for free.
Set `ANTHROPIC_API_KEY` and the same code path calls Claude instead.

---

## Why this exists

The hard part of shipping an LLM feature is not getting a good answer once. It
is knowing whether the answer is good, knowing when the system should decline,
proving the guardrails do not reject a tenth of your legitimate traffic, and
being able to say what a change cost you. This repo is those four things,
built around a small RAG system so they have something to be true about.

| Concern | What is here |
|---|---|
| **LLM/RAG evaluation** | Retrieval metrics, SQuAD-official answer metrics, abstention scored as a classifier, confidence calibration with reliability diagrams, an ablation across retrieval modes, and a release gate that exits non-zero |
| **Guardrails** | Input and output rails, each scored on a published labelled attack corpus and again on an attack corpus it has never seen |
| **Secure deployment** | Hashed API keys with constant-time comparison, a tamper-evident audit chain, PII redaction measured per entity type, a threat model, and a hardened container |
| **LLM gateway** | One chokepoint for every model call: auth, tier routing, two-layer cache, per-tenant budgets and rate limits, retries, graceful degradation, full cost accounting |

---

## The data

| Dataset | Role | Size | Licence |
|---|---|---|---|
| [SQuAD v2.0 dev](https://rajpurkar.github.io/SQuAD-explorer/) | Corpus (1,204 Wikipedia paragraphs), 11,873 questions, gold spans, and **5,945 genuinely unanswerable questions** | 4.2 MB | CC BY-SA 4.0 |
| [deepset/prompt-injections](https://huggingface.co/datasets/deepset/prompt-injections) | Injection rail: precision and recall | 662 | CC BY 4.0 |
| [jackhhao/jailbreak-classification](https://huggingface.co/datasets/jackhhao/jailbreak-classification) | Jailbreak rail | 1,306 | see card |
| [Lakera/gandalf_ignore_instructions](https://huggingface.co/datasets/Lakera/gandalf_ignore_instructions) | Held-out attacks, never trained on | 1,000 | MIT |
| [ai4privacy/pii-masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k) | PII redaction recall per entity | 1,000 | see card |

SQuAD v2 is the load-bearing choice. Half its questions were written by
crowdworkers to *look* answerable from a paragraph while having no answer in
it. That gives the abstention policy and the calibration study real ground
truth, which is the part of a RAG evaluation that is otherwise almost always
faked. The retrieval labels are free and exact: the gold passage is the
paragraph the question was written from.

---

## Headline results

Full report: [`docs/06-results.md`](docs/06-results.md). Reproduce with
`python3 scripts/run_eval.py`.

**Retrieval** (1,200 stratified questions, exact labels)

| Mode | recall@1 | recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| sparse (BM25) | 0.758 | **0.908** | **0.825** | 0.854 |
| hybrid (RRF) | 0.586 | 0.851 | 0.691 | 0.746 |
| dense (hashing) | 0.408 | 0.595 | 0.491 | 0.535 |

**Guardrails**

| Rail | Variant | Corpus | Recall | FPR | AUC |
|---|---|---|---|---|---|
| Injection | trained + heuristic | deepset test | 0.733 | 0.125 | 0.911 |
| Injection | trained + heuristic | **Gandalf, unseen** | 0.884 | n/a | |
| Injection | heuristic only | **Gandalf, unseen** | 0.553 | n/a | |
| Jailbreak | trained + heuristic | jailbreak test | 0.899 | 0.016 | 0.977 |

False-positive rate on 600 real benign user questions: **5.0%**.

**Calibration**: ECE 0.044, Brier 0.222, AUC 0.612 over 501 held-out questions.

**Answering**: at a 60% precision floor the system answers 15% of answerable
questions and correctly declines 87% of unanswerable ones.

---

## Three things the evaluation changed

The harness is not decoration. Each of these was a design decision reversed by
a measurement, and each is documented at the line of code it affected.

**1. Hybrid retrieval was removed.** Fusing dense and sparse retrieval is the
default recommendation, and here it is 5.7 points of recall@5 *worse* than
BM25 alone, because the hashing embedder has no semantic generalisation.
Down-weighting the dense side improves the fusion monotonically right up to the
point where it is switched off. The default is now `sparse`, all three modes
are still measured every run, and `config.py` records the numbers and the
condition under which the decision should flip.

**2. A guardrail was flagging 100% of a clean corpus.** The trained injection
classifier is fitted on short user queries. Pointed at 800-character
encyclopaedia prose it fired on every passage, so the pipeline retrieved
correctly and then discarded everything it found. Retrieved documents are now
scanned by the pattern detector only, which has no training distribution to
drift from. A classifier is valid on the distribution it was fitted on, and
"user turn" and "corpus document" are not the same distribution.

**3. A scoring function silently disabled a detector.** The heuristic
combiner mapped its single strongest pattern to 0.444, below a 0.5 threshold,
so any attack matching exactly one pattern scored as benign: **0 of 60** real
attacks caught, while the code read as perfectly reasonable. A unit test on a
single-pattern attack found it. The multi-pattern tests never could.

Two more are written up in [`docs/02-rag-evaluation.md`](docs/02-rag-evaluation.md):
an evaluation that was scoring its own confidence model circularly, and a
temperature-scaling step that made calibration measurably worse.

---

## How it fits together

```
question
   |
   v
[ input rail ]------ injection / jailbreak / length / PII ---> blocked
   |
   v
[ retrieve ]-------- BM25 + optional dense, RRF fusion ------> nothing found
   |
   v
[ context rail ]---- scan retrieved passages for payloads ---> drop passage
   |
   v
[ gateway ]--------- auth, budget, cache, route, retry ------> 401/402/429
   |                 every call metered and audited
   v
[ output rail ]----- citations, groundedness, PII egress ----> abstain
   |
   v
[ decision ]-------- calibrated confidence vs threshold -----> abstain
   |
   v
answer + citations + confidence
```

Each stage can stop the request, and each stops it for a different reason. The
last one is the interesting case: an answer can pass every safety check and
still be one the system should not assert. Separating "safe" from "confident"
is what lets a product hedge instead of choosing between bluffing and refusing.

```
src/groundwork/
  config.py          every threshold, quota, price, and gate floor
  rag/               corpus, embeddings, index, retriever, prompts, confidence
  guardrails/        detectors, classifier, input and output rails
  gateway/           providers, cache, budgets, router, HTTP server
  evaluation/        metrics, calibration, guardrail scoring, release gate
  security/          authz, tamper-evident audit log, PII redaction
  telemetry/         structured logging, drift monitoring
```

---

## Documentation

| Document | Contents |
|---|---|
| [01 Architecture](docs/01-architecture.md) | Request path, module boundaries, why the pieces are split this way |
| [02 RAG evaluation](docs/02-rag-evaluation.md) | What is measured, why those metrics, the five bugs the harness caught |
| [03 Guardrails](docs/03-guardrails.md) | Threat model per rail, measured effectiveness, what each rail cannot do |
| [04 LLM gateway](docs/04-llm-gateway.md) | Routing, caching, budgets, degradation, cost accounting |
| [05 Secure deployment](docs/05-secure-deployment.md) | Trust boundaries, secrets, audit, container hardening, incident response |
| [06 Results](docs/06-results.md) | Full measured output with commentary |
| [Threat model](deploy/THREAT-MODEL.md) | Assets, adversaries, controls, accepted risks |

---

## Honest limitations

Stated plainly, because a project about measuring things should not oversell
its own numbers.

- **The offline reader is extractive**, so exact-match against SQuAD's minimal
  gold spans is near zero by construction. Containment, EM, and token F1 are
  all reported, with the mismatch explained rather than hidden.
- **The embedder is a hashing bag-of-words**, not a semantic model. It is the
  honest floor a real embedding model has to beat, and the ablation is set up
  to test exactly that.
- **Confidence AUC is 0.612.** Distinguishing SQuAD v2's answerable from
  unanswerable questions is genuinely hard, by design, and a lexical system
  does it only somewhat better than chance. That weak signal is why a 60%
  precision promise costs most of the coverage, and the threshold sweep shows
  the whole trade rather than one flattering point on it.
- **PII recall is 0.57 over the entity types the detectors target**, and the
  corpus annotates 56 types they mostly do not. Regex catches structured
  identifiers and misses free-text names. That is why redaction is treated as
  defence in depth and not as the control that makes egress safe.
- **The single-node gateway** holds cache, budgets, and rate limits in
  process. The interfaces are the ones Redis gives you, so the swap is
  mechanical, but as written this does not scale past one instance.

---

## Licence

MIT. The datasets carry their own licences, recorded in `data/manifest.json`.
