"""A tamper-evident audit log.

Every model call, guardrail decision, and authorisation failure appends one
JSON line. Each record carries the SHA-256 of the previous record, so the file
is a hash chain: altering or deleting an entry after the fact breaks
verification at that point and at every point after it.

Why chain it at all. The audit log is the artefact you reach for when someone
asks "what did the system actually do on 12 March", and an audit log that can
be quietly edited answers that question with whatever the editor preferred.
Chaining does not prevent tampering (an attacker with write access can rewrite
the whole file), it makes tampering *detectable*, which is what turns the log
into evidence.

What is deliberately not recorded: prompt text, retrieved passages, model
output, API keys, and the content of any PII that was redacted. The log stores
what happened, who did it, and what it cost. Recording the text would recreate
in the log exactly the exposure the guardrails exist to prevent.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ARTIFACT_DIR

GENESIS = "0" * 64


@dataclass
class AuditRecord:
    event: str
    tenant: str
    outcome: str
    detail: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def canonical(self, previous_hash: str) -> str:
        """Serialise deterministically.

        `sort_keys` is load-bearing: dict ordering must not influence the hash,
        or a record re-serialised by a different Python version verifies as
        tampered.
        """
        return json.dumps({
            "ts": round(self.timestamp, 6),
            "event": self.event,
            "tenant": self.tenant,
            "outcome": self.outcome,
            "detail": self.detail,
            "prev": previous_hash,
        }, sort_keys=True, separators=(",", ":"))


class AuditLog:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or (ARTIFACT_DIR / "audit.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._last_hash = self._recover_tail()

    def _recover_tail(self) -> str:
        """Resume the chain from an existing file rather than restarting it.

        A process restart that reset the chain to GENESIS would silently split
        the log into unverifiable segments.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return GENESIS
        last = None
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    last = line
        if last is None:
            return GENESIS
        try:
            return json.loads(last)["hash"]
        except (json.JSONDecodeError, KeyError):
            # A truncated final line means the file is already inconsistent.
            # Continuing from GENESIS would hide that; raising surfaces it.
            raise RuntimeError(
                f"audit log {self.path} has a malformed final record; "
                "chain integrity cannot be established"
            )

    def append(self, event: str, tenant: str, outcome: str, **detail) -> str:
        record = AuditRecord(event=event, tenant=tenant, outcome=outcome, detail=detail)
        with self._lock:
            payload = record.canonical(self._last_hash)
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            line = json.dumps({
                "ts": round(record.timestamp, 6),
                "event": record.event,
                "tenant": record.tenant,
                "outcome": record.outcome,
                "detail": record.detail,
                "prev": self._last_hash,
                "hash": digest,
            }, sort_keys=True, separators=(",", ":"))
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())  # survive a crash between write and flush
            self._last_hash = digest
            return digest

    # -- verification -----------------------------------------------------

    def verify(self) -> tuple[bool, int, str | None]:
        """Recompute the chain. Returns (ok, records_checked, first_bad_hash)."""
        if not self.path.exists():
            return True, 0, None

        previous, checked = GENESIS, 0
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                entry = json.loads(line)
                record = AuditRecord(
                    event=entry["event"], tenant=entry["tenant"],
                    outcome=entry["outcome"], detail=entry["detail"],
                    timestamp=entry["ts"],
                )
                expected = hashlib.sha256(
                    record.canonical(entry["prev"]).encode("utf-8")
                ).hexdigest()
                if entry["prev"] != previous or expected != entry["hash"]:
                    return False, checked, entry["hash"]
                previous = entry["hash"]
                checked += 1
        return True, checked, None

    def read(self, limit: int | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            records = [json.loads(line) for line in fh if line.strip()]
        return records[-limit:] if limit else records
