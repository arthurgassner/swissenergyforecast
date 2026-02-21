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
    # Fetch latest loads/forecasts from ENTSOE
    lastest_load_and_forecast_df = entsoe_client.fetch_latest_load_and_forecast()
    await db_client.save_bronze(lastest_load_and_forecast_df) # Dump latest load/forecast to disk 

    # Measure the ENTSO-E's performance
    mape = performance_measure_service.compute_mape(
        y_true_col="Actual Load", y_pred_col="Forecasted Load", data=lastest_load_and_forecast_df,
        timedelta_strs=['1h', '24h', '1w', '4w'],
    )
    await db_client.save_entsoe_mape(mape) 
    logger.info(f"ENTSO-E MAPE: {mape}")

    # Clean the data
    lastest_load_and_forecast_df = data_cleaning_service.clean(lastest_load_and_forecast_df)
    await db_client.save_silver(lastest_load_and_forecast_df) 

    # Enrich with features
    lastest_load_and_forecast_df = feature_extraction_service.enrich(lastest_load_and_forecast_df)

    # Walk-forward validate the model
    latest_load_ts = await db_client.fetch_latest_load_ts()

    # TODO clean up this part
    # Figure out ranges to timestamps to test on
    past_24h_ts = latest_load_ts - timedelta(hours=23)
    past_1w_ts = latest_load_ts - timedelta(weeks=1)
    past_4w_ts = latest_load_ts - timedelta(weeks=4)

    past_24h_timestamps = pd.date_range(start=past_24h_ts, end=latest_load_ts, freq="h").to_list()
    past_1w_timestamps = pd.date_range(start=past_1w_ts, end=past_24h_ts, freq="h").to_list()
    past_4w_timestamps = pd.date_range(start=past_4w_ts, end=past_1w_ts, freq="h").to_list()

    # Estimate the MAPE off 10% (17 and 50) of the points for the past week/month
    # To avoid heavy computations
    model = Model(n_estimators=settings.MODEL_N_ESTIMATORS)
    query_timestamps = past_24h_timestamps + sample(past_1w_timestamps, 17) + sample(past_4w_timestamps, 50)
    walkforward_y = lastest_load_and_forecast_df[["24h_later_load"]]
    walkforward_yhat = model.train_predict(Xy=lastest_load_and_forecast_df, query_timestamps=query_timestamps) # TODO move to async ?..
    await db_client.save_walkforward_yhat(walkforward_yhat)

    mape = performance_measure_service.compute_mape(
        y_true_col="24h_later_load", y_pred_col="predicted_24h_later_load",
        data=pd.concat([walkforward_y, walkforward_yhat], axis=1, join="inner"),
        timedelta_strs=['1h', '24h', '1w', '4w'],
    )
    await db_client.save_our_model_mape(mape) 

    # Train-predict
    query_timestamps = [pd.Timestamp(latest_load_ts) + timedelta(hours=i) for i in range(1, 25)]
    our_model_yhat = model.train_predict(Xy=lastest_load_and_forecast_df, query_timestamps=query_timestamps)
    await db_client.save_our_model_yhat(our_model_yhat)

    return Forecast()
