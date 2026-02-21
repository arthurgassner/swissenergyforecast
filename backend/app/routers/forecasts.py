import os
from datetime import datetime, timedelta
from random import sample

import joblib
import pandas as pd
from entsoe import EntsoePandasClient
from fastapi import APIRouter, BackgroundTasks
from loguru import logger

from app.core.config import get_settings
from app.core.model import Model
from app.services import (
    data_cleaning_service,
    feature_extraction_service,
)

router = APIRouter()

@router.get("/forecasts/fetch/latest/predictions")
async def get_forecasts_fetch_latest_predictions():
    # Load latest forecast
    timestamps, predicted_24h_later_load = [], []
    if get_settings().YHAT_FILEPATH.is_file():
        yhat = pd.read_pickle(get_settings().YHAT_FILEPATH)
        timestamps = yhat.index.tolist()
        predicted_24h_later_load = yhat["predicted_24h_later_load"].fillna("NaN").tolist()

    latest_forecasts = {
        "timestamps": timestamps,
        "predicted_24h_later_load": predicted_24h_later_load,
    }

    logger.info(
        f"Ready to send back: {len(latest_forecasts['timestamps'])} timestamps [{min(latest_forecasts['timestamps'])} -> {max(latest_forecasts['timestamps'])}]"
    )

    return latest_forecasts


@router.get("/forecasts/fetch/latest/ts")
async def get_fetch_latest_forecast_ts():
    if not get_settings().YHAT_FILEPATH.is_file():
        logger.warning("No forecast has been created. Sending back -1")
        return {"latest_forecast_ts": -1}

    creation_ts = os.path.getctime(get_settings().YHAT_FILEPATH)  # since epoch
    logger.info(
        f"Ready to send back the creation timestamp of {get_settings().YHAT_FILEPATH}: {creation_ts} ({datetime.fromtimestamp(creation_ts)})"
    )
    return {"latest_forecast_ts": creation_ts}