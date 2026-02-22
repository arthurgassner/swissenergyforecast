from pathlib import Path

import joblib
import pandas as pd
from app.core.config import Settings
from app.schemas.forecast import MAPE, Forecast
from loguru import logger


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

    async def load_bronze(self) -> pd.DataFrame | None:
        """Load df from the bronze filepath."""
        if not self._bronze_filepath.is_file():
            return None

        return pd.read_pickle(self._bronze_filepath)

    async def load_silver(self) -> pd.DataFrame | None:
        """Load df from the silver filepath."""
        if not self._silver_filepath.is_file():
            return None

        return pd.read_pickle(self._silver_filepath)

    async def load_gold(self) -> pd.DataFrame | None:
        """Load df from the gold filepath."""
        if not self._gold_filepath.is_file():
            return None

        return pd.read_pickle(self._gold_filepath)

    async def fetch_latest_forecast(self) -> Forecast | None:
        if not self._latest_forecast_filepath.is_file():
            return None

        return joblib.load(self._latest_forecast_filepath)
