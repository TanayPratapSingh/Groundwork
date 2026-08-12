"""Security: PII detection, key handling, and the audit hash chain."""

from __future__ import annotations

import json

import pytest

from conftest import requires_attacks
from groundwork.security.audit import AuditLog
from groundwork.security.authz import AuthError, KeyStore, generate_key, hash_key
from groundwork.security.redaction import EntityType, Redactor, detect, luhn_valid
from groundwork.telemetry.logging import scrub


# -- PII -------------------------------------------------------------------

def test_email_is_detected():
    spans = detect("Contact ada@example.com for details")
    assert any(s.entity is EntityType.EMAIL for s in spans)


def test_luhn_filters_non_cards():
    assert luhn_valid("4532015112830366")
    assert not luhn_valid("1234567812345678")


def test_a_16_digit_order_number_is_not_a_card():
    """The false positive that gets a redaction rail switched off."""
    assert not any(s.entity is EntityType.CREDIT_CARD
                   for s in detect("Your order number is 1234567812345678"))


def test_api_keys_are_detected():
    assert any(s.entity is EntityType.API_KEY
               for s in detect("use sk-abcdefghijklmnopqrstuvwxyz123456 to authenticate"))


def test_overlapping_spans_are_resolved_once():
    spans = detect("Card 4532015112830366 on file")
    starts = [s.start for s in spans]
    assert len(starts) == len(set(starts))


def test_redaction_replaces_and_preserves_the_rest():
    result = Redactor(mode="replace").apply("Email ada@example.com about photosynthesis")
    assert "ada@example.com" not in result.text
    assert "photosynthesis" in result.text
    assert result.entity_counts["EMAIL"] == 1


def test_block_mode_empties_the_text():
    result = Redactor(mode="block").apply("Email ada@example.com")
    assert result.text == "" and result.redacted


def test_flag_mode_records_without_modifying():
    result = Redactor(mode="flag").apply("Email ada@example.com")
    assert "ada@example.com" in result.text and result.redacted


def test_span_serialisation_never_echoes_the_value():
    """Logging what you redacted recreates the exposure you just prevented."""
    span = detect("Email ada@example.com")[0]
    assert "ada@example.com" not in json.dumps(span.as_dict())


# -- keys ------------------------------------------------------------------

def test_generated_keys_are_long_and_unique():
    keys = {generate_key() for _ in range(50)}
    assert len(keys) == 50
    assert all(len(k) >= 32 for k in keys)


def test_keystore_authenticates_a_valid_key():
    store = KeyStore()
    store.add("secret-key", "analytics")
    assert store.authenticate("secret-key").tenant == "analytics"


def test_keystore_rejects_an_invalid_key():
    store = KeyStore()
    store.add("secret-key", "analytics")
    with pytest.raises(AuthError):
        store.authenticate("wrong-key")


def test_keystore_rejects_a_missing_key():
    with pytest.raises(AuthError):
        KeyStore().authenticate(None)


def test_raw_keys_are_never_stored():
    store = KeyStore()
    store.add("plaintext-secret", "analytics")
    assert "plaintext-secret" not in json.dumps(store._by_digest)


def test_key_id_is_short_and_not_reversible():
    store = KeyStore()
    store.add("secret-key", "analytics")
    principal = store.authenticate("secret-key")
    assert len(principal.key_id) == 12
    assert principal.key_id != "secret-key"


def test_example_dev_keys_are_refused_by_default(monkeypatch):
    """A copied-and-forgotten .env must fail closed, not grant access."""
    monkeypatch.setenv("GROUNDWORK_API_KEYS", "dev-key-analytics:analytics")
    monkeypatch.delenv("GROUNDWORK_ALLOW_DEV_KEYS", raising=False)
    assert len(KeyStore.from_env()) == 0


# -- audit -----------------------------------------------------------------

def test_audit_chain_verifies(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    for i in range(5):
        log.append("call", "analytics", "ok", index=i)
    ok, checked, bad = log.verify()
    assert ok and checked == 5 and bad is None


def test_tampering_is_detected(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(3):
        log.append("call", "analytics", "ok", index=i)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[1]["outcome"] = "denied"   # rewrite history
    path.write_text("\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"))
                              for r in records) + "\n")

    ok, _, bad = AuditLog(path).verify()
    assert not ok and bad is not None


def test_chain_resumes_across_restarts(tmp_path):
    """A restart that reset the chain would split the log into unverifiable
    segments without any error."""
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append("call", "a", "ok")
    AuditLog(path).append("call", "a", "ok")   # fresh instance, same file
    ok, checked, _ = AuditLog(path).verify()
    assert ok and checked == 2


def test_audit_records_carry_no_prompt_text(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("call", "analytics", "ok", model="m", cost_usd=0.01, input_tokens=100)
    record = log.read()[0]
    assert set(record["detail"]) <= {"model", "cost_usd", "input_tokens"}


# -- log scrubbing ---------------------------------------------------------

def test_scrub_removes_sensitive_values_but_keeps_shape():
    scrubbed = scrub({"tenant": "analytics", "api_key": "secret", "question": "who?"})
    assert scrubbed["tenant"] == "analytics"
    assert "secret" not in str(scrubbed["api_key"])
    assert "who?" not in str(scrubbed["question"])


@requires_attacks
def test_pii_recall_on_the_real_corpus_is_reported_not_assumed():
    """Structured identifiers should be caught; the overall number is a
    published figure in the eval report, not a hidden assumption."""
    from groundwork.evaluation.guardrail_eval import evaluate_pii

    result = evaluate_pii(verbose=False)
    if result.get("status") != "ok":
        pytest.skip("PII corpus unavailable")
    assert result["per_entity"].get("EMAIL", {}).get("recall", 0) > 0.8
    assert 0.0 <= result["overall_recall"] <= 1.0
