from datetime import datetime, timedelta
from random import sample

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends
from loguru import logger

from app.clients.db_client import DBCLient
from app.clients.entsoe_client import ENTSOEClient
from app.core.config import Settings, get_settings
from app.core.dependencies import get_db_client, get_entsoe_client
from app.core.model import Model
from app.schemas.forecast import MAPE, Forecast
from app.services import (
    data_cleaning_service,
    feature_extraction_service,
)

router = APIRouter()

@router.put("/forecast/latest")
async def put_forecast_latest(entsoe_client: ENTSOEClient = Depends(get_entsoe_client), db_client: DBCLient = Depends(get_db_client), settings: Settings = Depends(get_settings)) -> Forecast:
    # TODO move to background task
    
    # Fetch latest loads/forecasts from ENTSOE
    lastest_load_and_forecast_df = entsoe_client.fetch_latest_load_and_forecast()
    await db_client.save_bronze(lastest_load_and_forecast_df) # Dump latest load/forecast to disk 

    # Measure the ENTSO-E's performance
    y, yhat = lastest_load_and_forecast_df["Actual Load"], lastest_load_and_forecast_df["Forecasted Load"]
    entsoe_mapes = MAPE.compute_mapes(y=y, yhat=yhat, timedelta_strs=['1h', '24h', '1w', '4w'])

    # Clean the data
    lastest_load_and_forecast_df = data_cleaning_service.clean(lastest_load_and_forecast_df)
    await db_client.save_silver(lastest_load_and_forecast_df) 

    # Enrich with features
    lastest_load_and_forecast_df = feature_extraction_service.enrich(lastest_load_and_forecast_df)
    await db_client.save_gold(lastest_load_and_forecast_df) 

    # Walk-forward validate the model
    latest_load_ts = await db_client.fetch_latest_load_ts()

    # Compute the MAPE over the past week/month.
    # To avoid heavy computations, we estimate it only measuring on 10% of the timestamps for week and month
    # (i.e. 17 timestamps of the 170 timestamps in a week, and 50/500 in a month)
    # We have seen that this is a pretty good estimate
    query_timestamps = Model.get_hourly_timestamps(start=latest_load_ts - timedelta(hours=23), end=latest_load_ts)
    query_timestamps += Model.get_hourly_timestamps(start=latest_load_ts - timedelta(weeks=1), end=latest_load_ts - timedelta(hours=23), n_sample=17)
    query_timestamps += Model.get_hourly_timestamps(start=latest_load_ts - timedelta(weeks=4), end=latest_load_ts - timedelta(weeks=1), n_sample=50)
    model = Model(n_estimators=settings.MODEL_N_ESTIMATORS)
    walkforward_yhat = model.train_predict(Xy=lastest_load_and_forecast_df, query_timestamps=query_timestamps)

    y = lastest_load_and_forecast_df.reindex(walkforward_yhat.index)["24h_later_load"]
    our_mapes = MAPE.compute_mapes(y=y, yhat=walkforward_yhat, timedelta_strs=['1h', '24h', '1w', '4w'])

    # Train-predict
    query_timestamps = Model.get_hourly_timestamps(start=latest_load_ts + timedelta(hours=1), end=latest_load_ts + timedelta(hours=24))
    yhat = model.train_predict(Xy=lastest_load_and_forecast_df, query_timestamps=query_timestamps)

    forecast = Forecast(entsoe_mapes=entsoe_mapes, our_mapes=our_mapes)
    await db_client.save_latest_forecast(forecast)
    logger.success(f"Forecast computed:\n{forecast}")
    return forecast

@router.get("/forecast/latest")
async def get_forecast_latest(db_client: DBCLient = Depends(get_db_client)) -> Forecast:
    return await db_client.fetch_latest_forecast()