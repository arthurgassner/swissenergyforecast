from datetime import datetime
from datetime import timedelta
from random import sample
from uuid import UUID

import pandas as pd
from app.clients.db_client import DBClient
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
from fastapi.responses import ORJSONResponse
from loguru import logger

router = APIRouter()


@router.put("/forecast/custom", response_class=ORJSONResponse)
async def put_forecast_custom(
    entsoe_client: ENTSOEClient = Depends(get_entsoe_client),
    db_client: DBClient = Depends(get_db_client),
    settings: Settings = Depends(get_settings),
) -> Forecast:
    """Fetch ENTSO-E's latest data and compute a new forecast for its next 24h."""
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
    await db_client.save_latest_mapes(custom_mapes)
    [logger.success(f"- {e}") for e in custom_mapes]  # TODO rework

    # Train-predict the next 24h
    start_ts = latest_load_ts + timedelta(hours=1)
    query_timestamps = Model.get_hourly_timestamps(start=start_ts, end=start_ts + timedelta(hours=24))
    yhat = model.train_predict(Xy=lastest_load_and_forecast_df, query_timestamps=query_timestamps)

    forecast = Forecast(day_later_predicted_loads=yhat.to_list(), timestamps=yhat.index.to_list())
    await db_client.save_latest_forecast(forecast)
    return forecast


@router.get("/forecast/custom", response_class=ORJSONResponse)
async def get_forecast_custom(db_client: DBClient = Depends(get_db_client)) -> Forecast:
    """Fetch the latest-computed custom-forecast, for the next 24h"""
    latest_forecast = await db_client.fetch_latest_forecast()

    if latest_forecast is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No forecasts found in the database")

    return latest_forecast


@router.get("/forecast/entsoe", response_class=ORJSONResponse)
async def get_forecast_entsoe(db_client: DBClient = Depends(get_db_client)) -> Forecast:
    """Fetch the latest-fetched ENTSO-E-forecast, for the next 24h"""
    # Load past loads
    df = await db_client.load_silver()
    if df is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No ENTSO-E forecasts found in the database")

    # Only consider future records
    cutoff_ts = pd.Timestamp.now(tz=df.index.tz).floor("h") - pd.Timedelta(days=1)
    forecast_df = df.loc[df.index > cutoff_ts, "24h_later_forecast"]

    return Forecast(timestamps=forecast_df.index.to_list(), day_later_predicted_loads=forecast_df.to_list())


@router.get("/forecast/custom/range", response_class=ORJSONResponse)
async def get_forecast_custom_range(
    start_ts: int = Query(..., description="Start Unix timestamp (seconds)"),
    end_ts: int = Query(..., description="End Unix timestamp (seconds)"),
    db_client: DBClient = Depends(get_db_client),
) -> Forecast:
    """Fetch custom model's forecasts between two specific timestamps."""
    raise NotImplementedError()


@router.get("/forecast/entsoe/range", response_class=ORJSONResponse)
async def get_forecast_entsoe_range(
    start_ts: int = Query(..., description="Start Unix timestamp (seconds)"),
    end_ts: int = Query(..., description="End Unix timestamp (seconds)"),
    db_client: DBClient = Depends(get_db_client),
) -> Forecast:
    """Fetch ENTSO-E forecasts between two specific timestamps."""

    # Load past loads
    df = await db_client.load_silver()
    if df is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No ENTSO-E forecasts found in the database")

    # Convert UNIX ts -> datetime
    try:
        start_ts = pd.to_datetime(start_ts, unit="s", utc=True)
        end_ts = pd.to_datetime(end_ts, unit="s", utc=True)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid timestamp format: {str(e)}")

    # Offset by 1day, since we care about the forecasted value AT each timestamp
    start_ts -= timedelta(days=1)
    end_ts -= timedelta(days=1)

    # Only keep forecasts within start;end
    mask = (df.index >= start_ts) & (df.index <= end_ts)
    forecast_df = df.loc[mask, "24h_later_forecast"]

    if forecast_df.empty:
        return Forecast()

    return Forecast(timestamps=forecast_df.index.to_list(), day_later_predicted_loads=forecast_df.to_list())


@router.get("/forecast/custom/mapes")
async def get_forecast_custom_mapes(db_client: DBClient = Depends(get_db_client)) -> list[MAPE]:
    """Return the past performances of the latest-computed custom-model"""
    latest_mapes = await db_client.fetch_latest_mapes()

    if latest_mapes is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No MAPEs found in the database")

    return latest_mapes


@router.get("/forecast/entsoe/mapes")
async def get_forecast_entsoe_mapes(db_client: DBClient = Depends(get_db_client)) -> list[MAPE]:
    """Return the past performances of the latest-fetched ENTSO-E-model"""
    # Load past loads
    df = await db_client.load_silver()
    if df is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No ENTSO-E forecasts found in the database")

    # Measure the ENTSO-E past performance
    mapes = MAPE.compute_mapes(y=df["24h_later_load"], yhat=df["24h_later_forecast"], timedelta_strs=["1h", "24h", "1w", "4w"])
    return mapes


@router.get("/forecast/custom/{forecast_id}", response_class=ORJSONResponse)
async def get_forecast_custom_forecast_id(forecast_id: UUID, db_client: DBClient = Depends(get_db_client)) -> Forecast:
    raise NotImplementedError()
