
## Dev

### Install

- Install `uv`
- Run `uv sync --project backend --dev` TODO
- Run `uv run pre-commit install` TODO

### Run

To run locally environment, run `docker compose up --watch --build`.

> [!NOTE]
> This will run both the frontend and backend.
> The backend will auto-reload/rebuild on any code changes (see `compose.override.yaml` for details)
> **FIXME**: The frontend will NOT auto-reload/rebuild on any code changes (see `compose.override.yaml` for details)

### Test

To run the tests, run `uv run pytest`.

## Prod

`docker compose --file compose.yaml --file compose.prod.yaml up --build`