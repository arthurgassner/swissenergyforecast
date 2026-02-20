from fastapi import Depends

from app.clients.db_client import DBCLient
from app.clients.entsoe_client import ENTSOEClient
from app.core.config import Settings, get_settings


async def get_entsoe_client(settings: Settings = Depends(get_settings)) -> ENTSOEClient:
    return ENTSOEClient(api_key=settings.ENTSOE_API_KEY)


async def get_db_client(settings: Settings = Depends(get_settings)) -> ENTSOEClient:
    return DBCLient(settings=settings)