from pathlib import Path

import joblib
from loguru import logger
import pandas as pd

from app.core.config import Settings
from app.schemas.forecast import MAPE, Forecast


class DBCLient:
    """Dummy DBClient, to prepare for porting the codebase to a proper DB."""
    # TODO move to psql

    def __init__(self, settings: Settings) -> None:
        self._bronze_filepath = settings.BRONZE_DF_FILEPATH
        self._silver_filepath = settings.SILVER_DF_FILEPATH
        self._gold_filepath = settings.GOLD_DF_FILEPATH
        self._latest_forecast_filepath = settings.LATEST_FORECAST_FILEPATH

        # Ensure the filepaths can be reached
        self._bronze_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._silver_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._gold_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._latest_forecast_filepath.parent.mkdir(parents=True, exist_ok=True)

    async def save_bronze(self, df: pd.DataFrame) -> None:
        """Dump df to the bronze filepath."""
        df.to_pickle(self._bronze_filepath)

    async def save_silver(self, df: pd.DataFrame) -> None:
        """Dump df to the silver filepath."""
        df.to_pickle(self._silver_filepath)

    async def save_gold(self, df: pd.DataFrame) -> None:
        """Dump df to the gold filepath."""
        df.to_pickle(self._gold_filepath)

    async def save_latest_forecast(self, forecast: Forecast) -> None:
        """Dump forecast to the latest forecast's filepath."""
        joblib.dump(forecast, self._latest_forecast_filepath)

    async def load_bronze(self) -> pd.DataFrame:
        """Load df from the bronze filepath."""
        return pd.read_pickle(self._bronze_filepath)

    async def load_silver(self) -> pd.DataFrame:
        """Load df from the silver filepath."""
        return pd.read_pickle(self._silver_filepath)

    async def load_gold(self) -> pd.DataFrame:
        """Load df from the gold filepath."""
        return pd.read_pickle(self._gold_filepath)
    
    async def fetch_latest_forecast(self) -> Forecast:
        return joblib.load(self._latest_forecast_filepath)

    async def fetch_latest_load_ts(self) -> pd.Timestamp:
        """Fetch the timestamp of the latest load."""
        gold_df = await self.load_gold()
        return gold_df.dropna(subset=("24h_later_load")).index.max()