from datetime import datetime
from datetime import timedelta
from random import sample
from uuid import UUID

import pandas as pd
from app.clients.db_client import DBCLient
from app.clients.entsoe_client import ENTSOEClient
from app.core.config import Settings
from app.core.config import get_settings
from app.core.dependencies import get_db_client
from app.core.dependencies import get_entsoe_client
from app.core.model import Model
from app.schemas.forecast import MAPE
from app.schemas.forecast import Forecast
from app.services import data_cleaning_service
from app.services import feature_extraction_service
from fastapi import APIRouter
from fastapi import BackgroundTasks
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import status
from loguru import logger

router = APIRouter()


@router.put("/forecast/latest")
async def put_forecast_latest(
    entsoe_client: ENTSOEClient = Depends(get_entsoe_client),
    db_client: DBCLient = Depends(get_db_client),
    settings: Settings = Depends(get_settings),
) -> Forecast:
    # TODO move to background task

    # Fetch latest loads/forecasts from ENTSOE
    lastest_load_and_forecast_df = await entsoe_client.fetch_latest_load_and_forecast()
    await db_client.save_bronze(lastest_load_and_forecast_df)  # Dump latest load/forecast to disk

    # Clean the data
    lastest_load_and_forecast_df = data_cleaning_service.clean(lastest_load_and_forecast_df)
    await db_client.save_silver(lastest_load_and_forecast_df)

    # Enrich with features
    lastest_load_and_forecast_df = feature_extraction_service.enrich(lastest_load_and_forecast_df)
    await db_client.save_gold(lastest_load_and_forecast_df)

    # Walk-forward validate the model
    latest_load_ts = lastest_load_and_forecast_df.dropna(subset=("24h_later_load")).index.max()

    # Compute the MAPE over the past week/month.
    # To avoid heavy computations, we estimate it only measuring on 10% of the timestamps for week and month
    # (i.e. 17 timestamps of the 170 timestamps in a week, and 50/500 in a month)
    # We have seen that this is a pretty good estimate
    query_timestamps = Model.get_hourly_timestamps(start=latest_load_ts - timedelta(hours=23), end=latest_load_ts)
    query_timestamps += Model.get_hourly_timestamps(
        start=latest_load_ts - timedelta(weeks=1), end=latest_load_ts - timedelta(hours=23), n_sample=17
    )
    query_timestamps += Model.get_hourly_timestamps(
        start=latest_load_ts - timedelta(weeks=4), end=latest_load_ts - timedelta(weeks=1), n_sample=50
    )
    model = Model(n_estimators=settings.MODEL_N_ESTIMATORS)
    walkforward_yhat = model.train_predict(Xy=lastest_load_and_forecast_df, query_timestamps=query_timestamps)

    # Estimate custom model performance
    y = lastest_load_and_forecast_df["24h_later_load"].reindex(walkforward_yhat.index)
    custom_mapes = MAPE.compute_mapes(y=y, yhat=walkforward_yhat, timedelta_strs=["1h", "24h", "1w", "4w"])

    # Train-predict
    query_timestamps = Model.get_hourly_timestamps(
        start=latest_load_ts + timedelta(hours=1), end=latest_load_ts + timedelta(hours=24)
    )
    yhat = model.train_predict(Xy=lastest_load_and_forecast_df, query_timestamps=query_timestamps)

    forecast = Forecast(day_later_predicted_loads=yhat.to_list(), timestamps=yhat.index.to_list(), mapes=custom_mapes)
    await db_client.save_latest_forecast(forecast)
    logger.success(f"Forecast computed:\n{forecast}")
    return forecast


@router.get("/forecast/custom/latest")
async def get_forecast_custom_latest(db_client: DBCLient = Depends(get_db_client)) -> Forecast:
    latest_forecast = await db_client.fetch_latest_forecast()

    if latest_forecast is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No forecasts found in the database")

    return latest_forecast


@router.get("/forecast/custom/{forecast_id}")
async def get_forecast_custom_forecast_id(forecast_id: UUID, db_client: DBCLient = Depends(get_db_client)) -> Forecast:
    raise NotImplementedError()


@router.get("/forecast/entsoe")
async def get_forecast_entsoe(
    days: int = Query(0, ge=0, description="How many days to look back"),
    hours: int = Query(0, ge=0, description="How many hours to look back"),
    db_client: DBCLient = Depends(get_db_client),
) -> Forecast:

    # Load past loads
    df = await db_client.load_silver()
    if df is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No ENTSO-E loads found in the database")

    # Figure out since when the records should be sent
    start_ts = df.index.max() - timedelta(days=days, hours=hours)
    df = df[df.index > start_ts]

    # Measure the ENTSO-E performance
    mapes = MAPE.compute_mapes(y=df["24h_later_load"], yhat=df["24h_later_forecast"], timedelta_strs=["1h", "24h", "1w", "4w"])

    return Forecast(
        timestamps=df["24h_later_forecast"].index.to_list(),
        day_later_predicted_loads=df["24h_later_forecast"].to_list(),
        mapes=mapes,
    )
