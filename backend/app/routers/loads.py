from datetime import timedelta

from app.clients.db_client import DBClient
from app.core.dependencies import get_db_client
from app.schemas.entsoe_loads import ENTSOELoads
from fastapi import APIRouter
from fastapi import Depends
from fastapi import HTTPException
from fastapi import Query
from fastapi import status
from fastapi.responses import ORJSONResponse

router = APIRouter()


# TODO: rework to support /range format
# NOTE: ORJSON is faster than standard JSON (rust), and allows for NaN -> null typecasting
@router.get("/loads", response_class=ORJSONResponse)
async def get_loads(
    days: int = Query(0, ge=0, description="How many days to look back"),
    hours: int = Query(0, ge=0, description="How many hours to look back"),
    db_client: DBClient = Depends(get_db_client),
) -> ENTSOELoads:

    # Load past loads
    df = await db_client.load_silver()
    if df is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No ENTSO-E loads found in the database")

    # Figure out since when the records should be sent
    start_ts = df.index.max() - timedelta(days=days, hours=hours) - timedelta(days=1)
    df = df[df.index > start_ts]

    return ENTSOELoads(timestamps=df.index.tolist(), day_later_loads=df["24h_later_load"].tolist())
