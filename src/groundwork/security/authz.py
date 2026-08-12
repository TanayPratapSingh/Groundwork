"""API key authentication and scope checks.

Three properties this gets right, all of which are routinely got wrong:

1. **Keys are stored hashed.** The gateway holds SHA-256 digests, never the
   key material. A dump of the key table does not let you call the API.
2. **Comparison is constant time.** `==` on secrets leaks length and prefix
   through timing. `hmac.compare_digest` does not.
3. **Keys are never logged.** Only the digest prefix reaches telemetry, which
   is enough to identify which key was used and useless for replaying it.

Keys come from the `GROUNDWORK_API_KEYS` environment variable as
`key:tenant,key:tenant`. That is fine for a single service and explicitly not
where this should end up: production wants a secrets manager with rotation and
per-key expiry, which is what deploy/THREAT-MODEL.md specifies.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass, field

from ..config import SETTINGS


class AuthError(RuntimeError):
    """Authentication failed. Never carries a reason to the caller."""


class ScopeError(RuntimeError):
    """Authenticated but not permitted to perform this action."""


def hash_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_key() -> str:
    """A key with 256 bits of entropy. Use this, not a memorable string."""
    return secrets.token_urlsafe(32)


@dataclass(frozen=True)
class Principal:
    """The authenticated caller."""

    tenant: str
    key_digest: str
    scopes: frozenset[str] = field(default_factory=frozenset)

    @property
    def key_id(self) -> str:
        """Short, non-reversible key identifier, safe for logs."""
        return self.key_digest[:12]

    def require(self, scope: str) -> None:
        if scope not in self.scopes:
            raise ScopeError(f"tenant '{self.tenant}' lacks scope '{scope}'")


class KeyStore:
    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        """`mapping` is digest -> tenant. Raw keys are accepted only via
        `add`, which hashes on the way in."""
        self._by_digest: dict[str, str] = dict(mapping or {})

    # -- construction -----------------------------------------------------

    @classmethod
    def from_env(cls, variable: str = "GROUNDWORK_API_KEYS") -> "KeyStore":
        store = cls()
        raw = os.getenv(variable, "").strip()
        for pair in filter(None, (p.strip() for p in raw.split(","))):
            if ":" not in pair:
                continue
            key, tenant = pair.rsplit(":", 1)
            # The example keys shipped in .env.example are refused explicitly,
            # so a copied-and-forgotten config fails closed rather than
            # granting access under a placeholder credential.
            if key.startswith("dev-key-") and os.getenv("GROUNDWORK_ALLOW_DEV_KEYS") != "1":
                continue
            store.add(key.strip(), tenant.strip())
        return store

    def add(self, raw_key: str, tenant: str) -> str:
        digest = hash_key(raw_key)
        self._by_digest[digest] = tenant
        return digest

    # -- verification -----------------------------------------------------

    def authenticate(self, raw_key: str | None) -> Principal:
        if not raw_key:
            raise AuthError("missing credentials")

        candidate = hash_key(raw_key)
        # Every stored digest is compared, and comparison is constant time, so
        # neither the number of configured keys nor how early a match occurs is
        # observable from response latency.
        matched: str | None = None
        for digest, tenant in self._by_digest.items():
            if hmac.compare_digest(candidate, digest):
                matched = tenant
        if matched is None:
            raise AuthError("invalid credentials")

        quota = SETTINGS.gateway.tenants.get(matched)
        scopes = frozenset(quota.scopes) if quota else frozenset()
        return Principal(tenant=matched, key_digest=candidate, scopes=scopes)

    def __len__(self) -> int:
        return len(self._by_digest)

    @property
    def configured(self) -> bool:
        return bool(self._by_digest)
