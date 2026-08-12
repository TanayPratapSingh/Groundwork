# RAG evaluation

How this system is measured, why those metrics were chosen, and the five bugs
the harness caught that code review did not.

---

## 1. The problem with evaluating RAG

Most RAG evaluations measure the easy half. They ask questions the corpus can
answer and check whether the answer is right. That tells you nothing about the
failure that actually damages trust: a fluent, confident, well-cited answer to
a question the corpus cannot answer at all.

Measuring that requires questions with **no answer**, labelled as such by
someone other than you. Writing them yourself does not work, because you will
unconsciously write ones that are obviously unanswerable, and the system will
look great at declining them.

SQuAD v2.0 solves this. 5,945 of its 11,873 dev questions were written by
crowdworkers instructed to produce questions that *look* answerable from a
given paragraph while having no answer in it. They are adversarial by
construction, written by people who were not building this system.

That single property is what makes everything downstream possible:

| Needed for | Provided by |
|---|---|
| Retrieval labels | The paragraph each question was written from, exactly |
| Answer labels | Human-marked gold spans, several per question |
| Abstention labels | The `is_impossible` flag |
| Calibration labels | Whether answering *would have been* correct |

---

## 2. What is measured

### Stage 1: retrieval

Run with no model calls at all, so it completes in seconds over thousands of
questions and can sit in a pre-commit hook.

| Metric | Why |
|---|---|
| recall@k | Did the gold passage make it into the context at all? Upper bound on everything downstream |
| MRR | Distinguishes gold-at-rank-1 from gold-at-rank-5, which recall@5 scores identically. This is what a reranker moves |
| nDCG@10 | Comparable with published retrieval results |

All three retrieval modes are scored every run over the same questions and the
same index, so the comparison is an ablation rather than three separate claims.

**Blind spot, stated because a metric without one is a metric being oversold:**
these assume exactly one correct passage per question. A question answerable
from three passages scores as a miss on two of them, so on a corpus with
redundancy these are a lower bound.

### Stage 2: answers

| Metric | Value | What it means |
|---|---|---|
| Containment | 0.127 | Answerable questions where a gold span appears in the delivered answer |
| Containment ceiling | 0.581 | The same, ignoring the abstain decision: what a perfect policy would get |
| Exact match | 0.000 | Official SQuAD EM |
| Token F1 | 0.141 | Official SQuAD token overlap |

**EM near zero is expected and is not a system failure.** SQuAD's gold answers
are minimal spans ("France", "1918"). This pipeline returns cited sentences.
A completely correct sentence containing "France" scores 0 on EM and about
0.25 on F1, because six of its seven tokens were never in the gold span.

So containment is the primary metric: did the correct fact reach the user. EM
and F1 are reported anyway, for comparability and as a check, because
containment alone can be gamed by answering at length and the two moving in
opposite directions would show that immediately.

The **ceiling** matters as much as the headline. Containment 0.127 against a
ceiling of 0.581 says the generation is fine and the decision policy is
throwing answers away. Without that number, a containment drop is ambiguous
between "generation regressed" and "the policy got more cautious", and those
have completely different fixes.

### Stage 3: abstention as a classifier

Treating "should decline" as the positive class turns the interesting
behaviour into a confusion matrix.

| | Value |
|---|---|
| Correct abstention on unanswerable | 0.868 |
| False answer rate on unanswerable | 0.132 |
| Answer rate on answerable | 0.153 |

The system declines 87% of genuinely unanswerable questions, and pays for it
by declining a lot of answerable ones too. That trade is the product decision,
and it is set explicitly rather than emerging by accident (section 4).

### Stage 4: calibration

A calibrated system is one where "0.8 confident" means right about 80% of the
time. That is stronger and more useful than accuracy, because it is what makes
a threshold mean anything.

| Metric | Value |
|---|---|
| ECE (15 bins, 501 held-out questions) | 0.044 |
| MCE | 0.366 |
| Brier | 0.222 |
| AUC | 0.612 |

**ECE must never be read alone.** A model that outputs 0.5 for everything and
is right half the time has near-zero ECE and no discriminative value at all.
Quoted next to AUC 0.612 the pair is honest: the confidence is well calibrated
and only somewhat discriminative.

The reliability diagram is printed in the terminal after every run, next to
the numbers, because a calibration plot that lives in a notebook nobody opens
has never prevented a bad release.

---

## 3. The confidence model

Six features, logistic regression, fitted on real correctness labels.

The naive approach is to use the top retrieval score as confidence. It is a
bad estimator: not on a probability scale, blind to whether the answer used
the retrieved text, and unable to distinguish one strong hit from five
mediocre ones.

Features were chosen by measuring single-feature AUC against real labels over
812 answered questions, not by intuition:

| Feature | AUC | Kept |
|---|---|---|
| `query_coverage_answer` | 0.660 | yes |
| `query_coverage_top` | 0.608 | yes |
| `top_score` | 0.597 | yes |
| `query_coverage_context` | 0.575 | yes |
| `answer_length` | 0.540 | yes |
| `answer_type_match` | 0.526 | yes |
| `score_margin` | 0.497 | dropped |
| `answer_support` | 0.500 | dropped |
| `n_passages` | 0.500 | dropped |

`answer_support`, the fraction of answer terms found in the retrieved context,
is the instructive one. It reads like *the* obvious groundedness signal and it
scores exactly 0.500. The reader is extractive, so every answer is copied out
of the context and the feature is ~1.0 on every row, correct and incorrect
alike. It is a real check for a generative reader and pure noise for this one.

Fitted coefficients, which are readable precisely because the model is linear:

| Feature | Weight |
|---|---|
| `query_coverage_answer` | +1.69 |
| `query_coverage_top` | +0.96 |
| `top_score` | −0.66 |
| `query_coverage_context` | −0.29 |
| `answer_length` | −0.22 |
| `answer_type_match` | +0.12 |

The negative weight on `top_score` is worth noticing rather than smoothing
over: conditional on the other features, a high retrieval score is mildly
*negative* evidence. SQuAD v2's unanswerable questions were written against a
specific paragraph, so they retrieve that paragraph strongly. Retrieval
confidence is not answer confidence, and here the data says so directly.

---

## 4. Choosing the operating point

The threshold is chosen on the fit split by maximising coverage subject to a
precision floor:

> answer as many questions as possible, provided that when we answer we are
> right at least 60% of the time

The obvious alternative, maximising decision accuracy, was tried first and
**degenerates**. SQuAD v2 is half unanswerable, so a system that answers
nothing already scores 0.50, and with a confidence signal at AUC 0.61 the
accuracy-optimal cutoff abstains on nearly everything: it chose 0.48 and
delivered correct answers to 7% of answerable questions. That is the true
optimum for the metric and a useless product.

The precision floor is the one number a stakeholder should be asked to sign
off on. Everything else about the decision policy follows from it, and the
threshold sweep in the report shows the full curve so the trade is made
deliberately rather than discovered later.

---

## 5. The release gate

`scripts/run_eval.py` exits non-zero when the gate fails, so it drops into CI
without a wrapper.

Two kinds of check, and both are needed:

- **Absolute floors.** Without them, a build that was always mediocre keeps
  shipping.
- **Regression against the recorded baseline.** Without it, quality slides
  indefinitely at 2% a release, with every individual release passing.

The floors were set from the first complete measured run, roughly one margin
below what was achieved. They started as round numbers written before any
measurement existed, which made the first three runs fail on a containment
target nothing in the system could reach.

**The rule from that point on: floors only ever move up.** They are ratcheted
when a change improves a metric and never relaxed to turn a red build green. A
gate that can be lowered by the person whose change it just failed is not a
gate. If a legitimate change makes a metric worse, that trade is argued in the
pull request and the number changes in the same commit, in the open.

---

## 6. Five bugs the harness caught

Every one of these passed code review and would have shipped.

### The context rail flagged 100% of a clean corpus

The trained injection classifier is fitted on short user queries. Pointed at
800-character Wikipedia prose, every passage looked anomalous, so the pipeline
retrieved correctly and then discarded everything it found. Every question got
"no relevant passages".

Retrieved documents are now scanned by the pattern detector only, which has no
training distribution to drift from. **A classifier is valid on the
distribution it was fitted on**, and "user turn" and "corpus document" are not
the same distribution.

### The evaluation was scoring its own confidence model circularly

The pipeline made abstain decisions using the confidence model, and the
harness then fitted that model from the results, which means it fitted on data
the model itself had filtered.

Fixed by separating candidate generation from the decision. `Answer` carries a
`candidate_text` that is always populated, the harness runs with
`apply_decision=False`, fits confidence on one split, and only then chooses an
operating point. Confidences recorded during the run are recomputed with the
fitted model before any decision is scored, because otherwise the policy is
evaluated against the prior it replaced.

### Temperature scaling made calibration worse

The textbook step, fitted on held-out data, moved held-out ECE from 0.053 to
0.084. Selecting on NLL, then on ECE, then on ECE with a relative-improvement
guard all failed the same way, because the problem was never the objective: at
~295 validation points and a baseline ECE around 0.05, **the optimal
temperature is not identifiable**. Validation confidently preferred T=0.70 and
the test split preferred T=1.0 every time.

Scaling is now applied only when the model is visibly miscalibrated to begin
with (ECE > 0.08) *and* scaling buys a large absolute improvement. It stays
available for the case it is actually for, an early build here scored ECE
0.94, and stops firing on noise. Identity is the default, not the fallback.

### Hybrid retrieval was worse than sparse alone

Documented in the README and in `config.py`. Fusing a weak dense retriever
with a strong sparse one cost 5.7 points of recall@5.

### The gate was guarding a code path no request took

It read `retrieval["hybrid"]` while the pipeline had already been switched to
`sparse`. It now reads whichever mode is configured as the default, and so
does the baseline writer, or the first run after a default change reports a
phantom regression.

---

## 7. Running it

```bash
python3 scripts/run_eval.py                  # 1,200-question stratified sample
python3 scripts/run_eval.py -n 400           # faster inner loop
python3 scripts/run_eval.py --full           # all 11,873 questions
python3 scripts/run_eval.py --set-baseline   # record this run as the baseline
python3 scripts/run_eval.py --no-guardrails  # RAG metrics only
```

Sampling is stratified and seeded, so a sampled run is comparable across
commits. That is the only thing that makes a regression gate on a sample mean
anything.

Outputs land in `artifacts/`: `results.json` for machines, `RESULTS.md` for
people, `baseline.json` for the next run to compare against.
