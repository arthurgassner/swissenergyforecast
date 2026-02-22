import sys
import time

from app.core.config import get_settings
from app.routers.forecast import router as forecast_router
from app.routers.loads import router as entsoe_loads_router
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

logger.remove()
logger.add(sys.stderr, colorize=True)  # Force colorization, as Docker strips them otherwise
logger.add(get_settings().LOGS_FILEPATH, level="INFO", rotation="10 MB", retention="365 days")

app = FastAPI(title="[Swiss Energy Forcasting] ML Backend")

# CORS configuration
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def middleware(request: Request, call_next):
    start_time_s = time.perf_counter()

    response = await call_next(request)

    elapsed_time_s = time.perf_counter() - start_time_s
    logger.info(
        f"from {request.client.host}:{request.client.port} | {request.method} {request.url}: {response.status_code} {elapsed_time_s * 1e3:.0f}ms"
    )

    response.headers["X-Process-Time"] = str(elapsed_time_s)

    return response


@app.get("/")
async def get_root():
    return {"message": "Welcome to the swissenergyforecast-backend!"}


app.include_router(entsoe_loads_router)
app.include_router(forecast_router)
