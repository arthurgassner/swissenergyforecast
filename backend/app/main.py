import sys

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.core.config import get_settings
from app.routers.entsoe_loads import router as entsoe_loads_router
from app.routers.forecasts import router as forecasts_loads_router
from app.routers.forecast import router as forecast_router

logger.remove()
logger.add(sys.stderr, colorize=True)  # Force colorization, as Docker strips them otherwise
logger.add(get_settings().LOGS_FILEPATH, level="INFO", rotation="10 MB", retention="365 days")

app = FastAPI(title="[Swiss Energy Forcasting] ML Backend")

# CORS configuration
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.middleware("http")
async def middleware(request: Request, call_next):
    logger.info(f"Received {request.method} on {request.url} from {request.client.host}:{request.client.port}")
    response = await call_next(request)
    return response


@app.get("/")
async def get_root():
    return {"message": "Welcome to the swissenergyforecast-backend!"}


app.include_router(entsoe_loads_router)
app.include_router(forecasts_loads_router)
app.include_router(forecast_router)