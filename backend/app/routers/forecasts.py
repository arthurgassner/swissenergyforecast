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