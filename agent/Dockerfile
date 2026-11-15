FROM python:3.13-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev


FROM python:3.13-slim-bookworm

# Kaleido renders chart PNGs through a local Chromium.
RUN apt-get update \
    && apt-get install -y --no-install-recommends chromium fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 agent

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    BROWSER_PATH=/usr/bin/chromium \
    CALLBACK_HOST=0.0.0.0 \
    CALLBACK_PORT=8181

WORKDIR /app
COPY --from=build --chown=agent:agent /app /app
RUN mkdir -p /app/data /app/logs && chown agent:agent /app/data /app/logs

USER agent
EXPOSE 8181

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8181/health', timeout=4)"

CMD ["ont-agent-bot"]
