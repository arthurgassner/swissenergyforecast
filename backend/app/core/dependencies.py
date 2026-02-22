from app.clients.db_client import DBClient
from app.clients.entsoe_client import ENTSOEClient
from app.core.config import Settings, get_settings
from entsoe.entsoe import EntsoePandasClient
from fastapi import Depends


async def _get_entsoe_pandas_client(settings: Settings = Depends(get_settings)) -> EntsoePandasClient:
    return EntsoePandasClient(api_key=settings.ENTSOE_API_KEY)


async def get_entsoe_client(
    entsoe_pandas_client: EntsoePandasClient = Depends(_get_entsoe_pandas_client), settings: Settings = Depends(get_settings)
) -> ENTSOEClient:
    return ENTSOEClient(entsoe_pandas_client=entsoe_pandas_client, settings=settings)


async def get_db_client(settings: Settings = Depends(get_settings)) -> ENTSOEClient:
    return DBClient(settings=settings)
