"""HTTP surface for the gateway.

FastAPI is an optional dependency. The whole system works without it; this
module is the deployment shape, not the core.

Two things here are security decisions rather than plumbing:

**Errors are generic to the caller and specific in the audit log.** A 401 that
says "unknown key" versus "key expired" turns the endpoint into an oracle for
probing valid credentials. The operator gets the detail; the caller gets a
status code.

**The API key is read from a header, never a query parameter.** Query strings
land in access logs, proxy logs, browser history, and Referer headers.
"""

from __future__ import annotations

import os
from typing import Any

try:
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    FASTAPI_AVAILABLE = False
    BaseModel = object  # type: ignore

from ..config import DEFAULT_TIER, SETTINGS
from ..rag.index import HybridIndex
from ..rag.pipeline import RagPipeline
from ..security.authz import AuthError, KeyStore, Principal, ScopeError
from ..telemetry.logging import get_logger, log_event
from .router import Gateway, GatewayError

logger = get_logger("groundwork.server")


if FASTAPI_AVAILABLE:

    class AskRequest(BaseModel):
        question: str = Field(min_length=1, max_length=SETTINGS.guardrails.max_input_chars)
        tier: str = Field(default=DEFAULT_TIER)
        top_k: int | None = Field(default=None, ge=1, le=20)

    class AskResponse(BaseModel):
        answer: str
        decision: str
        confidence: float
        groundedness: float
        citations: list[str]
        sources: list[dict]
        latency_ms: float

    def create_app(index: HybridIndex | None = None, gateway: Gateway | None = None) -> "FastAPI":
        app = FastAPI(
            title="Groundwork",
            version="0.1.0",
            description="Retrieval-grounded QA with guardrails, budgets, and an audit trail.",
        )

        # Loaded once at startup rather than per request. Rebuilding an index
        # inside a handler is the classic way a p99 becomes a p50.
        app.state.index = index or HybridIndex.load()
        app.state.gateway = gateway or Gateway(keystore=KeyStore.from_env())
        app.state.pipeline = RagPipeline(app.state.index, gateway=app.state.gateway)

        def principal_from(x_api_key: str | None = Header(default=None)) -> Principal:
            try:
                return app.state.gateway.authenticate(x_api_key)
            except GatewayError:
                # Deliberately uniform. The audit log already recorded which
                # key was presented and why it failed.
                raise HTTPException(status_code=401, detail="unauthorized")

        @app.get("/health")
        def health() -> dict[str, Any]:
            """Liveness only. Deliberately unauthenticated and deliberately
            free of version, tenant, or corpus detail: a health endpoint is
            the most-scanned URL you expose."""
            return {"status": "ok"}

        @app.get("/ready")
        def ready() -> dict[str, Any]:
            """Readiness: can this instance actually serve a request?"""
            return {
                "status": "ready" if len(app.state.index) else "no index",
                "documents": len(app.state.index),
                "provider": app.state.gateway.provider.name,
            }

        @app.post("/v1/ask", response_model=AskResponse)
        def ask(request: AskRequest, principal: Principal = Depends(principal_from)) -> Any:
            try:
                principal.require("query")
            except ScopeError:
                raise HTTPException(status_code=403, detail="forbidden")

            try:
                answer = app.state.pipeline.answer(
                    request.question,
                    tenant=principal.tenant,
                    tier=request.tier,
                    top_k=request.top_k,
                )
            except GatewayError as exc:
                # Budget and rate-limit statuses are passed through: unlike
                # auth failures these are the caller's own quota, so telling
                # them is useful and leaks nothing.
                raise HTTPException(status_code=exc.status, detail=str(exc))

            log_event(logger, "answered", tenant=principal.tenant,
                      decision=answer.decision.value,
                      confidence=round(answer.confidence, 3),
                      latency_ms=round(answer.latency_ms, 1))

            payload = answer.as_dict()
            return {
                "answer": payload["answer"],
                "decision": payload["decision"],
                "confidence": payload["confidence"],
                "groundedness": payload["groundedness"],
                "citations": payload["citations"],
                "sources": payload["sources"],
                "latency_ms": payload["latency_ms"],
            }

        @app.get("/v1/usage")
        def usage(principal: Principal = Depends(principal_from)) -> dict[str, Any]:
            """A tenant sees only its own ledger."""
            return app.state.gateway.budgets.ledger(principal.tenant).as_dict()

        @app.exception_handler(Exception)
        def unhandled(request: Request, exc: Exception) -> JSONResponse:
            # Stack traces are for the operator's log, never the response body.
            logger.exception("unhandled error")
            return JSONResponse(status_code=500, content={"detail": "internal error"})

        return app

else:  # pragma: no cover

    def create_app(*_args, **_kwargs):
        raise RuntimeError(
            "FastAPI is not installed. Install the optional extra:\n\n"
            "    pip install 'groundwork[serve]'\n"
        )


def run() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.getenv("GROUNDWORK_HOST", "127.0.0.1"),
        port=int(os.getenv("GROUNDWORK_PORT", "8000")),
    )
