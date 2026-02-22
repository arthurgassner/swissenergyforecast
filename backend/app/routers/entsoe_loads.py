from datetime import timedelta
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status

from fastapi.responses import ORJSONResponse
import pandas as pd
from fastapi import APIRouter, Depends, Query
from loguru import logger

from app.clients.db_client import DBCLient
from app.core.config import get_settings
from app.core.dependencies import get_db_client
from app.schemas.entsoe_loads import (
    ENTSOELoads,
)

router = APIRouter()


@router.get("/loads", response_class=ORJSONResponse)
async def get_loads(
    days: int = Query(0, ge=0, description="How many days to look back"),
    hours: int = Query(0, ge=0, description="How many hours to look back"),
    db_client: DBCLient = Depends(get_db_client)
    ) -> ENTSOELoads:

    # Load past loads
    silver_df = await db_client.load_silver()
    if silver_df is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No ENTSO-E loads found in the database")

    # Figure out till when the records should be sent
    end_ts = silver_df.index.max()
    cutoff_ts = end_ts - timedelta(days=days, hours=hours)

    # Only keep the data till
    silver_df = silver_df[silver_df.index > cutoff_ts]

    return ENTSOELoads(timestamps=silver_df.index.tolist(), day_later_loads=silver_df["24h_later_load"].astype(float).tolist())
