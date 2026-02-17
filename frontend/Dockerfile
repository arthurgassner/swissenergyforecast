FROM ghcr.io/astral-sh/uv:python3.14-alpine AS builder

# TODO optimize the build step

WORKDIR /app
COPY . .

RUN uv sync 
RUN uv run mkdocs build

FROM nginx:alpine

# NOTE: mkdocs defaults to the site folder
COPY --from=builder /app/site /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]