# Secure deployment

Trust boundaries, secrets, audit, container hardening, and what to do when
something goes wrong.

The full asset and adversary analysis is in
[`deploy/THREAT-MODEL.md`](../deploy/THREAT-MODEL.md); this document is the
operational half.

---

## 1. Trust boundaries

Four of them, and the mistake is almost always assuming there are three.

```
   [ user ]
       │  untrusted: anything typed here is adversarial input
       v
   [ gateway ]───────────────────────> [ model provider ]
       │                                 third party: everything you send
       │                                 leaves your control
       v
   [ retrieval corpus ]
       untrusted: a document added months ago is
       adversarial input on a channel nobody watches
```

The corpus is the boundary people forget. Retrieved text is *data*, never
instructions, and it is scanned before it reaches a prompt. See
[03-guardrails](03-guardrails.md) section 4.

The provider boundary matters just as much. Everything sent to a third-party
API has left your infrastructure. Redaction reduces what crosses that line and
does not make crossing it safe, which is why regulated tenants should be
running the redactor in `block` mode rather than `replace`.

---

## 2. Secrets

**API keys are stored hashed.** The keystore holds SHA-256 digests, never key
material. A dump of the key table does not let you call the API.

**Comparison is constant time.** `==` on secrets leaks length and prefix
through timing; `hmac.compare_digest` does not. Every stored digest is
compared on every authentication, so neither the number of configured keys nor
how early a match occurs is observable from response latency.

**Keys are never logged.** Only a 12-character digest prefix reaches
telemetry: enough to identify which key was used, useless for replaying it.

**Example keys fail closed.** The placeholders in `.env.example` are refused
unless `GROUNDWORK_ALLOW_DEV_KEYS=1` is set explicitly, so a copied-and-
forgotten config denies access rather than granting it under a placeholder
credential. There is a test for this.

Generate real keys with 256 bits of entropy, never a memorable string:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

For production, `GROUNDWORK_API_KEYS` in the environment is not the answer. It
is fine for a single service and has no rotation, no expiry, and no per-key
revocation. Use a secrets manager, mount at runtime, rotate on a schedule, and
revoke on exposure.

---

## 3. The audit log

Every model call, guardrail decision, and authorisation failure appends one
JSON line. Each record carries the SHA-256 of the previous record, so the file
is a hash chain.

```bash
python3 -c "
from groundwork.security.audit import AuditLog
ok, n, bad = AuditLog().verify()
print(f'verified={ok} records={n} first_bad={bad}')"
```

**Why chain it.** The audit log is what you reach for when someone asks what
the system did on 12 March, and a log that can be quietly edited answers that
question with whatever the editor preferred. Chaining does not *prevent*
tampering: an attacker with write access can rewrite the whole file. It makes
tampering **detectable**, which is what turns a log into evidence.

The chain resumes across process restarts by reading the last record. A
restart that reset it to genesis would silently split the log into
unverifiable segments, which is worse than not chaining at all.

**What is deliberately not recorded**: prompt text, retrieved passages, model
output, API keys, and the content of anything redacted. The log stores what
happened, who did it, and what it cost. Recording the text would recreate in
the log exactly the exposure the guardrails exist to prevent, usually in a
system with longer retention and wider access than the database.

The same principle is enforced in structured logging: `telemetry.logging.scrub`
drops known-sensitive keys before anything is emitted, keeping sizes. Knowing a
prompt was 4,000 characters is operationally useful and reveals nothing. This
exists because the realistic failure is not somebody deliberately logging a
key; it is a `logger.info("request: %s", payload)` added while debugging that
survives into production.

---

## 4. Container

`deploy/Dockerfile` is multi-stage and the runtime stage:

- runs as a non-root user with no shell
- mounts the root filesystem read-only, with `tmpfs` for scratch
- drops every Linux capability
- sets `no-new-privileges`
- pins the base image by digest, not by tag
- ships no compiler and no package manager
- declares a `HEALTHCHECK` against `/health`

`deploy/k8s/` adds a `NetworkPolicy` that denies all egress except DNS and
HTTPS to the model provider. That is the control that turns a successful
prompt injection from an exfiltration into a failed connection.

Resource limits are set. An LLM service without a memory limit is one large
retrieved context away from an OOM that takes the node with it.

---

## 5. Deploying

```bash
cp .env.example .env                    # then put real keys in it
docker compose -f deploy/docker-compose.yml up --build
curl -s localhost:8000/health
curl -s -X POST localhost:8000/v1/ask \
  -H "X-API-Key: $GROUNDWORK_KEY" -H 'Content-Type: application/json' \
  -d '{"question":"What is the largest inland port in Europe?"}'
```

Pre-flight checklist:

- [ ] Real keys generated, `.env` not committed (it is in `.gitignore`)
- [ ] `GROUNDWORK_ALLOW_DEV_KEYS` unset
- [ ] Tenant quotas set to real budgets in `config.py`
- [ ] `python3 scripts/run_eval.py` exits 0
- [ ] `artifacts/baseline.json` committed so CI has something to compare to
- [ ] Egress policy applied and verified from inside the container
- [ ] Audit log on persistent storage, not the container filesystem
- [ ] Redactor set to `block` for any regulated tenant
- [ ] Alerts wired to drift, abstention rate, and per-tenant spend

---

## 6. What to monitor

Distribution drift, not accuracy. Production has no labels, so accuracy cannot
be monitored there; the inputs can be.

`telemetry.drift` computes PSI and the KS statistic against a reference
window:

| PSI | Reading |
|---|---|
| < 0.10 | stable |
| 0.10 to 0.25 | moderate shift, investigate |
| ≥ 0.25 | significant, treat fitted thresholds as stale |

Watch `top_score`, `score_margin`, `query_coverage`, `confidence`, and
`groundedness`.

**The specific signature to alert on is confidence drifting up while
groundedness drifts down.** That is a system that has started bluffing, and it
will not show up in error rates, latency, or any health check.

Also page on: abstention rate moving more than a few points in a day
(threshold or corpus problem), guardrail fire rate spiking (attack, or a rail
that broke), per-tenant spend approaching cap, and audit chain verification
failing (which is an incident on its own, whatever else is true).

Both statistics stay silent below 20 samples. An alert nobody can trust trains
people to ignore all of them.

---

## 7. Incident response

**Suspected key compromise.** Remove the digest from the keystore and redeploy;
that revokes immediately. Then grep the audit log for that `key_id` to scope
what was accessed, and rotate every key issued to the same tenant.

**Successful prompt injection.** The audit log gives you the tenant and the
timestamp but not the prompt, by design, so scope from the guardrail decisions
and the retrieval trace. Add a pattern to the heuristic detector and a fixture
to `tests/test_guardrails.py` in the same change, so the fix has a regression
test before it has a deploy.

**Poisoned corpus document.** `check_context` drops rather than blocks, so
service continues while you remove the document and rebuild the index. Search
the audit log for the affected period to find which answers cited it.

**Runaway spend.** Budgets cap it automatically. Lower `daily_usd` for the
tenant and redeploy; the check is pre-flight, so it takes effect on the next
call rather than after the next bill.

**Audit chain broken.** Treat as a compromise until proven otherwise. The
first bad hash is returned by `verify()` and localises the tampering to the
record after which the chain diverges.
