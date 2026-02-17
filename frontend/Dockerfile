# Build stage
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder

# Install all dependencies
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=frontend/pyproject.toml,target=frontend/pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --project frontend

WORKDIR /app
COPY . .

# Rely on cache to build static site
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run --project frontend mkdocs build -f frontend/mkdocs.yml

# Run stage
FROM nginx:alpin
# NOTE: mkdocs defaults to the site folder
COPY --from=builder /app/frontend/site /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]