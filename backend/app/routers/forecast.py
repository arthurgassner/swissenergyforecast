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
from app.schemas.forecast import Forecast
from app.services import (
    data_cleaning_service,
    data_loading_service,
    feature_extraction_service,
    performance_measure_service,
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
    mape = performance_measure_service.compute_mape(y=y, yhat=yhat, timedelta_strs=['1h', '24h', '1w', '4w'])
    await db_client.save_entsoe_mape(mape) 
    logger.info(f"ENTSO-E MAPE: {mape}")

    # Clean the data
    lastest_load_and_forecast_df = data_cleaning_service.clean(lastest_load_and_forecast_df)
    await db_client.save_silver(lastest_load_and_forecast_df) 

    # Enrich with features
    lastest_load_and_forecast_df = feature_extraction_service.enrich(lastest_load_and_forecast_df)

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
    await db_client.save_walkforward_yhat(walkforward_yhat)

    # TODO is this contact then split needed ?...
    y_and_yhat = pd.concat([lastest_load_and_forecast_df[["24h_later_load"]], walkforward_yhat], axis=1, join="inner")
    y, yhat = y_and_yhat["24h_later_load"], y_and_yhat["predicted_24h_later_load"]
    mape = performance_measure_service.compute_mape(y=y, yhat=yhat, timedelta_strs=['1h', '24h', '1w', '4w'])
    await db_client.save_our_model_mape(mape)
    logger.info(f"Our model's MAPE: {mape}")

    # Train-predict
    query_timestamps = [pd.Timestamp(latest_load_ts) + timedelta(hours=i) for i in range(1, 25)]
    our_model_yhat = model.train_predict(Xy=lastest_load_and_forecast_df, query_timestamps=query_timestamps)
    await db_client.save_our_model_yhat(our_model_yhat)

    return Forecast()
