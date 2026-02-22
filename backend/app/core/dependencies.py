from fastapi import Depends
from entsoe.entsoe import EntsoePandasClient

from app.clients.db_client import DBCLient
from app.clients.entsoe_client import ENTSOEClient
from app.core.config import Settings, get_settings


async def _get_entsoe_pandas_client(settings: Settings = Depends(get_settings)) -> EntsoePandasClient:
    return EntsoePandasClient(api_key=settings.ENTSOE_API_KEY)

async def get_entsoe_client(entsoe_pandas_client: EntsoePandasClient = Depends(_get_entsoe_pandas_client), settings: Settings = Depends(get_settings)) -> ENTSOEClient:
    return ENTSOEClient(entsoe_pandas_client=entsoe_pandas_client, settings=settings)

async def get_db_client(settings: Settings = Depends(get_settings)) -> ENTSOEClient:
    return DBCLient(settings=settings)