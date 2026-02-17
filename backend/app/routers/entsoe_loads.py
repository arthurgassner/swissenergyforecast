import pandas as pd
from fastapi import APIRouter
from loguru import logger

from app.core.config import get_settings
from app.schemas.entsoe_loads_fetch_latest import (
    EntsoeLoadsFetchLatestRequest,
    EntsoeLoadsFetchLatestResponse,
)

router = APIRouter()


@router.post("/entsoe-loads/fetch/latest")
async def post_entsoe_loads_fetch_latest(request: EntsoeLoadsFetchLatestRequest) -> EntsoeLoadsFetchLatestResponse:
    # Load past loads
    silver_df = pd.read_pickle(get_settings().SILVER_DF_FILEPATH)

    # Figure out till when the records should be sent
    end_ts = silver_df.index.max()
    cutoff_ts = end_ts - request.delta_time

    # Only keep the data till
    silver_df = silver_df[silver_df.index > cutoff_ts]

    response = EntsoeLoadsFetchLatestResponse(
        timestamps=silver_df.index.tolist(),
        day_later_loads=silver_df["24h_later_load"].astype(float).fillna("NaN").tolist(),
        day_later_forecasts=silver_df["24h_later_forecast"].astype(float).fillna("NaN").tolist(),
    )

    if len(response.timestamps):
        logger.info(f"Ready to send back: {len(response.timestamps)} timestamps between {cutoff_ts} -> {end_ts}")
    else:
        logger.warning(
            f"Ready to send empty dict: {len(response.timestamps)} timestamps between {cutoff_ts} -> {end_ts}"
        )

    return response
