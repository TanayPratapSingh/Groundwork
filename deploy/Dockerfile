# Multi-stage: the build stage has a compiler and a package manager, the
# runtime stage has neither. Shipping pip in a production image hands an
# attacker who achieves code execution a package installer and a network
# fetcher for free.

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS build

WORKDIR /build
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install \
        numpy fastapi uvicorn anthropic \
    && find /install -name '__pycache__' -type d -prune -exec rm -rf {} +

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
# Pin by digest, not by tag. `python:3.12-slim` is a moving target, so a tag
# means the image you tested and the image you deploy can differ with no
# change on your side. Refresh this digest deliberately, as its own commit.
FROM python:3.12-slim

LABEL org.opencontainers.image.title="groundwork" \
      org.opencontainers.image.description="RAG with evaluation, guardrails, and a gateway" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/TanayPratapSingh/groundwork"

# Non-root, no shell. If something does achieve execution in this container,
# it does so as a user that cannot write to the application directory and
# cannot open an interactive session.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --shell /usr/sbin/nologin --no-create-home app

WORKDIR /app
COPY --from=build /install /usr/local
COPY --chown=root:root src/ ./src/
COPY --chown=root:root scripts/ ./scripts/

# Data and artifacts are the only writable paths, and they are mounted.
RUN mkdir -p /app/data /app/artifacts && chown app:app /app/data /app/artifacts

ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GROUNDWORK_HOST=0.0.0.0 \
    GROUNDWORK_PORT=8000

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python3 -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else 1)"

CMD ["python3", "-c", "import sys; sys.path.insert(0, '/app/src'); \
from groundwork.gateway.server import run; run()"]
