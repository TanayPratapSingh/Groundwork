# Guardrails

Every rail here has a published precision, recall, and false-positive rate
against a labelled corpus somebody else wrote. A guardrail with no measured
false-positive rate is a liability dressed as a control.

---

## 1. The shape

**Input rails decide whether to call the model. Output rails decide whether to
return what it said.** They are separate because they fail differently.

An input rail is a classifier over adversarial text. It will be evaded,
because the attacker chooses the input and can iterate against it for free.

An output rail checks a property of the answer against the retrieved context.
It does not depend on guessing the attacker's phrasing, so it still holds when
the input rail has been beaten.

Neither is trusted to be sufficient. That layering is the entire design.

```
user text ──> [ length ] ──> [ injection ] ──> [ jailbreak ] ──> [ PII ] ──> model
                                                                              │
retrieved passages ──> [ context injection ] ─────────────────────────────────┤
                                                                              v
                        answer <── [ PII egress ] <── [ groundedness ] <── [ citations ]
```

---

## 2. Detectors: two kinds, on purpose

Each threat gets two independent detectors, combined by taking the maximum
score.

**Heuristic**: hand-written patterns, no training data, fully inspectable,
fires the same way on an attack invented tomorrow as on one from a corpus.

**Trained**: hashed n-grams plus logistic regression, fitted on a labelled
corpus, catches phrasings nobody thought to write a pattern for.

The reason both are kept is visible in the measurements, and it is not the
reason you would guess:

| Detector | deepset test | Gandalf (never seen) |
|---|---|---|
| Heuristic only | 0.083 | **0.553** |
| Trained + heuristic | 0.733 | **0.884** |

On the corpus the classifier was fitted on, the heuristic looks useless: 0.083
recall, because deepset's attacks are largely German and these patterns are
English. On Gandalf, 1,000 real attempts by players trying to beat a live
guarded system, the same heuristic catches 0.553 on its own. Patterns have no
training distribution to drift from.

A single detector would have looked excellent right up until traffic shifted.

### Why not a transformer classifier

An input rail runs on the critical path of every request, before the model
call, and competes with the model call for the user's patience. A rail that
adds 300ms of GPU inference to save 900ms of a refused generation is a bad
trade at p50 and a worse one at p99. This classifier scores a prompt in
microseconds on a CPU, which is what lets it be always-on.

The cost is measured rather than hidden: word n-grams generalise to paraphrase
poorly, so a lexically novel attack gets past. That is exactly why the rails
are layered and why the output rail verifies grounding independently.

---

## 3. Measured effectiveness

From `python3 scripts/run_eval.py`, on the test splits of corpora the
detectors were fitted on and on one they have never seen.

| Rail | Variant | Corpus | Precision | Recall | FPR | AUC |
|---|---|---|---|---|---|---|
| Injection | heuristic | deepset test | 1.000 | 0.083 | 0.000 | 0.567 |
| Injection | trained + heuristic | deepset test | 0.863 | 0.733 | 0.125 | 0.911 |
| Injection | heuristic | Gandalf, held out | 1.000 | 0.553 | n/a | |
| Injection | trained + heuristic | Gandalf, held out | 1.000 | 0.884 | n/a | |
| Jailbreak | heuristic | jailbreak test | 1.000 | 0.489 | 0.000 | 0.813 |
| Jailbreak | trained + heuristic | jailbreak test | 0.984 | 0.899 | 0.016 | 0.977 |

Gandalf is attack-only, so recall is meaningful there and FPR is undefined.

### The false-positive rate that actually matters

The FPR above is measured against the benign class of an attack corpus, which
is mostly short topical fragments. Production traffic is not that.

So it is also measured against **600 real SQuAD questions**, human-written
with no adversarial intent, as a stand-in for legitimate traffic:

| Rail | FPR on real user questions |
|---|---|
| Injection | **0.050** |
| Jailbreak | **0.000** |

This measurement exists because of a failure. The injection rail was first
tuned against only its own corpus's benign class and then rejected 6% of real
SQuAD questions in the end-to-end run. **A rail tuned on one benign
distribution tells you nothing about its false-positive rate on another, and
production traffic is always the other one.** Those 600 questions are now part
of threshold selection, and tightening the constraint from 10% to 5% halved
the rate with no loss of attack recall.

Threshold selection is a product decision expressed as a constraint: maximise
recall subject to a false-positive ceiling. Optimising F1 instead would
happily buy a few points of recall at a 25% false-positive rate, and a rail
that rejects a quarter of legitimate questions is a rail that gets switched
off within the week.

---

## 4. Indirect injection: the one that gets exploited

The user turn is adversarial input everyone remembers to check. A document
somebody added to the corpus eighteen months ago is adversarial input arriving
through a channel nobody is watching. In RAG, that is the variant that gets
exploited.

`InputRail.check_context` scans every retrieved passage before it reaches the
prompt. Two design choices:

**A poisoned passage is dropped, not blocked.** One bad document should
degrade the answer, never deny service for the whole query. Blocking turns a
corpus-integrity problem into an outage.

**Context scanning is heuristic-only.** The trained classifier is fitted on
short user queries; pointed at 800-character prose it flagged 100% of a clean
Wikipedia corpus as poisoned. This is documented at the line of code, in
`rails.py`, and has a regression test.

Prompt-level defences are also in place, and are not trusted. The system
prompt fences retrieved text and states it is data, never instructions. That
is necessary and nowhere near sufficient, which is why the scan exists and why
the output rail verifies citations independently of whether anything fired.

---

## 5. Output rails

Checked in order of how much they matter.

### Citation validity

Every `[Sn]` must refer to a passage that was actually supplied. A fabricated
citation is worse than no citation: it is a claim of provenance that will
survive being copied into a document someone relies on. A fabricated citation
abstains the answer outright.

### Groundedness

What fraction of the answer's content sentences share enough vocabulary with
the retrieved context to be plausibly derived from it. Lexical, therefore
cheap, therefore always-on.

Reported as a coverage proxy and not as a truth judgment. It catches wholesale
confabulation. It will not catch a subtly wrong date, and claiming otherwise
would be the same overselling this project exists to avoid.

Three bands: above target, passes; between target and a hard floor, passes but
is surfaced as reduced confidence rather than silently returned as solid;
below the floor, abstains.

**Groundedness alone is not relevance.** The demo made this concrete: a
question about Europe's largest inland port returned the correct Duisburg
sentence followed by an unrelated true sentence about the population of
Fresno. Both were properly cited and perfectly grounded. Relevance is a
separate property and needs enforcing separately, which the reader now does.

### PII egress

Retrieved documents can contain personal data the question never mentioned.
Redacting only the input misses that path entirely.

---

## 6. PII detection, and what a low score can mean

Recall against ai4privacy's human-annotated spans:

| Entity | Recall | Spans |
|---|---|---|
| EMAIL | 1.000 | 84 |
| IBAN | 1.000 | 31 |
| SSN | 0.576 | 33 |
| IP_ADDRESS | 0.562 | 146 |
| PHONE | 0.388 | 67 |
| DATE_OF_BIRTH | 0.387 | 62 |
| CREDIT_CARD | 0.175 | 63 |
| **Overall** | **0.570** | **486** |

**The credit-card number is the interesting one, and it is not a detector
failure.** Card matches are Luhn-validated, because a 16-digit order reference
is not a credit card and a detector that cannot tell the difference produces
enough false positives that people switch it off. Exactly **17.5% of the
corpus's annotated card numbers pass Luhn**, which is precisely the measured
recall. The detector is not missing cards; the corpus's synthetically
generated cards are not valid cards.

That is worth stating because the instinct on seeing 0.175 is to loosen the
detector, which would have traded a real precision guarantee for a better
number against bad labels. **Validate your evaluation data before believing a
low score.**

The corpus annotates 56 entity types. These detectors target nine. Structured
identifiers are caught well, free-text names and addresses are not caught at
all, and the overall figure is quoted against the targeted subset with that
caveat attached in the report itself.

The design consequence matters more than the numbers: because recall on names
is weak, **redaction is defence in depth, not the control that makes it safe
to send arbitrary user text to a third party.** The control carrying that
weight is not sending the text, which is why `Redactor` has a `block` mode and
why the deployment guide recommends it for regulated tenants.

---

## 7. What these rails cannot do

- **Stop a determined novel attack.** Lexical detectors are evaded by
  paraphrase. Layering, an output rail that does not depend on phrasing, and a
  gateway that caps blast radius are the response.
- **Verify factual correctness.** Groundedness measures whether a claim traces
  to a passage, not whether the passage is right or the inference from it is
  valid.
- **Redact free-text names.** See above.
- **Survive being the only control.** Every rail here assumes it sits behind
  authentication, budgets, an audit trail, and a corpus somebody curated.

Failure messages are deliberately identical across reasons. A refusal that
explains which rail fired is a refusal that teaches an attacker how to route
around it; the specific reason goes to the audit log instead.
