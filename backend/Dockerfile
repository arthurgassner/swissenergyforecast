# --- BUILD ---
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder
WORKDIR /src

# Install python dependencies
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=backend/pyproject.toml,target=backend/pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --project backend

# --- RUN ---
FROM python:3.14-slim
WORKDIR /src

# Install curl, for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Install the OpenMP shared library -- used by lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/.venv /src/.venv
COPY backend .

# Put the .venv on the PATH
ENV PATH="/src/.venv/bin:$PATH"

CMD ["fastapi", "run", "app/main.py", "--port", "8000"]