"""Structured logging with a redaction filter in the path.

One line of JSON per event, and a filter that scrubs known-sensitive keys
before anything is emitted. The filter exists because the realistic failure is
not somebody deliberately logging an API key, it is a `logger.info("request:
%s", payload)` added while debugging that survives into production and quietly
writes prompts and credentials to a log aggregator with a different retention
policy and a wider audience than the database.

Defence in depth: the audit log never receives text in the first place
(security/audit.py), and this filter catches the paths that do.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

SENSITIVE_KEYS = frozenset({
    "api_key", "apikey", "authorization", "token", "secret", "password",
    "prompt", "question", "answer", "text", "content", "context", "messages",
    "system", "passage", "passages",
})

REDACTED = "[redacted]"


def scrub(value: Any, depth: int = 0) -> Any:
    """Recursively drop sensitive values, keeping structure and sizes.

    Sizes are kept rather than deleted: knowing a prompt was 4,000 characters
    is operationally useful and reveals nothing about what it said.
    """
    if depth > 6:
        return "[truncated]"
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key.lower() in SENSITIVE_KEYS:
                out[key] = f"{REDACTED}({len(str(item))} chars)"
            else:
                out[key] = scrub(item, depth + 1)
        return out
    if isinstance(value, (list, tuple)):
        return [scrub(v, depth + 1) for v in value[:20]]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload["fields"] = scrub(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "groundwork", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def log_event(logger: logging.Logger, message: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, message, extra={"fields": fields})
