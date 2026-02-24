# --- BUILD ---
FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim AS builder
WORKDIR /src

# Install all dependencies
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=frontend/pyproject.toml,target=frontend/pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --project frontend

COPY . .

# Rely on cache to build static site
RUN --mount=type=cache,target=/root/.cache/uv \
    uv run --project frontend mkdocs build -f frontend/mkdocs.yml

# --- RUN ---
FROM nginx:alpine
WORKDIR /src

# Install curl, for healthcheck
RUN apk add --no-cache curl

# NOTE: mkdocs defaults to the site/ folder
COPY --from=builder /src/frontend/site /usr/share/nginx/html
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]