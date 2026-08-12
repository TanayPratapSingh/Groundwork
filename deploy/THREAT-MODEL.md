# Threat model

Assets, adversaries, controls, and the risks that are accepted rather than
mitigated. The accepted-risk section is the one worth reading: a threat model
without one is a list of things that went well.

---

## 1. Assets

| Asset | Why it is worth taking |
|---|---|
| Provider API credentials | Direct financial loss, at attacker-chosen volume |
| Tenant API keys | Impersonation, and a tenant's own budget to spend |
| The corpus | May be proprietary; poisoning it steers every future answer |
| User questions | Frequently contain personal or commercially sensitive data |
| The system prompt | Reveals the guardrail surface, which makes evasion cheaper |
| The audit log | The record of what happened; valuable to alter, not to read |
| Model spend | An availability and cost target in its own right |

---

## 2. Adversaries

**Opportunistic user.** Curious, occasionally malicious, no persistence. Tries
"ignore previous instructions" and asks for the system prompt. Handled by the
input rails.

**Motivated attacker.** Iterates against the rails, which are deterministic
and free to probe. Will eventually find a phrasing that gets through, because
lexical classifiers are evadable by construction. The response is layering,
egress control, and blast-radius limits, not a better classifier.

**Corpus contributor.** Anyone who can add a document. The most dangerous
adversary here and usually not modelled at all: their payload arrives through
a trusted channel, months before it is used, and is surfaced by the system's
own retrieval.

**Compromised tenant.** A leaked key. Bounded by that tenant's quota,
attributable through the audit log, revocable by removing one digest.

**Insider with log access.** The reason the audit log carries no prompt text
and structured logs are scrubbed. Read access to observability should not
equal read access to user data.

---

## 3. Threats and controls

### T1. Direct prompt injection

*Overwrite the operator's instructions from the user turn.*

Input rail, both heuristic and trained: 0.733 recall on the deepset test
split, 0.884 on the held-out Gandalf corpus, 5.0% false positives on real
benign questions. The output rail catches consequences the input rail misses,
since it checks a property of the answer rather than the phrasing of the
attack.

**Residual: high.** This will be evaded. Everything downstream assumes it.

### T2. Indirect injection via the corpus

*Plant instructions in a document the retriever will one day surface.*

`InputRail.check_context` scans every retrieved passage before it reaches the
prompt, heuristic-only because the trained classifier is fitted on queries and
misfires badly on prose. Poisoned passages are dropped, not blocked, so one
bad document degrades an answer rather than denying service. The system prompt
fences retrieved text as data.

**Residual: medium.** Depends on corpus governance, which is outside this
codebase. Who can add a document, and is it reviewed?

### T3. Data exfiltration

*Get the system to send data somewhere the attacker controls.*

The exfiltration pattern is in the injection detector. The real control is the
`NetworkPolicy`: default-deny egress with private ranges excluded, including
the cloud metadata endpoint. The service has no outbound tools, no fetch, and
no code execution, so there is no capability to hijack.

**Residual: medium.** The `ipBlock` rule cannot express hostnames, so it stops
lateral movement and metadata access but not exfiltration to an arbitrary
public HTTPS host. An egress proxy with a hostname allowlist closes that, and
is recommended in the manifest.

### T4. Credential theft

*Obtain provider or tenant keys.*

Keys stored as SHA-256 digests, compared with `hmac.compare_digest`, never
logged, and surfaced only as a 12-character digest prefix. Provider keys
arrive from the environment and are never written to disk. Example keys are
refused unless explicitly enabled. Read-only root filesystem, non-root user,
all capabilities dropped.

**Residual: low** in the app, **medium** in the platform. Kubernetes Secrets
are base64, not encryption; a real secrets manager is the fix and is called
out in `secrets.yaml`.

### T5. Cost exhaustion

*Burn the budget through volume or oversized inputs.*

Per-tenant daily caps and per-minute rate limits, both checked **before** the
call. Input length capped. Cost estimation is pessimistic, assuming full
output. Over-budget tenants degrade to a cheaper tier instead of failing.

**Residual: low.** This is the best-covered threat here, because it is the one
with an unambiguous, checkable invariant.

### T6. PII leakage to a third party

*Personal data in a question, or in a retrieved passage, reaching the provider.*

Redaction on input and on output egress. Measured, not assumed: 1.00 recall on
emails and IBANs, 0.57 overall across targeted types, and effectively zero on
free-text names.

**Residual: high, and stated as such.** Regex redaction does not make it safe
to send arbitrary user text to a third party. The control that carries that
weight is not sending it, which is what `Redactor(mode="block")` is for.

### T7. Audit tampering

*Alter the record of what happened.*

SHA-256 hash chain, verifiable in one call, resuming correctly across process
restarts. `fsync` on every append, so a crash between write and flush does not
lose the record.

**Residual: medium.** Chaining makes tampering detectable, not impossible: an
attacker with write access can rewrite the whole file consistently. Shipping
records off-host, or anchoring the chain head somewhere external, is the fix.

### T8. Model-output-driven harm

*A confident, well-cited, wrong answer.*

Groundedness checking, citation validation against supplied passages, and a
calibrated confidence with an explicit precision floor. Measured: ECE 0.044,
and 0.868 correct abstention on genuinely unanswerable questions.

**Residual: medium.** Groundedness is lexical. It catches confabulation and
will not catch a subtly wrong date drawn from a real passage.

---

## 4. Accepted risks

Stated plainly, because these are the ones that will bite.

1. **Lexical guardrails are evadable.** Accepted because the alternative, a
   transformer on the critical path of every request, costs more latency than
   it saves and would get switched off. Mitigated by layering and by limiting
   what a successful bypass can reach.

2. **Egress is not hostname-restricted.** Accepted for a reference deployment;
   an egress proxy is documented as the production answer.

3. **Single instance.** Cache, budgets, and rate limits are in-process, so
   horizontal scaling would break the guarantees. Accepted deliberately, with
   the Redis-shaped interfaces already in place.

4. **PII detection misses names and addresses.** Accepted, because the
   alternative, an NER model on every request, has its own recall ceiling and
   real latency cost. Mitigated by `block` mode for regulated tenants.

5. **The corpus is trusted to be curated.** This codebase scans retrieved text
   but does not govern who can add a document. Corpus governance is a
   prerequisite, not a feature here.

6. **No per-user authorisation inside a tenant.** Every key for a tenant sees
   the whole corpus. Document-level access control would be a substantial
   addition to the retriever and is out of scope.

---

## 5. What would change the picture

If this system gained tools, the analysis changes materially. Retrieval and
generation have no side effects; a tool that can write, send, or fetch turns
every injection finding from "wrong answer" into "unauthorised action", and
T1's high residual risk stops being tolerable.

The same applies to multi-tenancy of the corpus. Today one corpus is shared by
all tenants, so retrieval leaks nothing across them. Per-tenant corpora would
make the retriever an access-control boundary, and it is not built to be one.
