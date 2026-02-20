from pathlib import Path

import joblib
import pandas as pd

from app.core.config import Settings


class DBCLient:
    """Dummy DBClient, to prepare for porting the codebase to a proper DB."""
    # TODO move to psql

    def __init__(self, settings: Settings) -> None:
        self._bronze_filepath = settings.BRONZE_DF_FILEPATH
        self._silver_filepath = settings.SILVER_DF_FILEPATH
        self._gold_filepath = settings.GOLD_DF_FILEPATH
        self._entsoe_mape_filepath = settings.ENTSOE_MAPE_FILEPATH
        self._our_model_mape_filepath = settings.OUR_MODEL_MAPE_FILEPATH
        self._walkforward_yhat_filepath = settings.WALKFORWARD_YHAT_FILEPATH
        self._our_model_yhat_filepath = settings.YHAT_FILEPATH

        # Ensure the filepaths can be reached
        self._bronze_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._silver_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._gold_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._entsoe_mape_filepath = settings.ENTSOE_MAPE_FILEPATH
        self._our_model_mape_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._walkforward_yhat_filepath.parent.mkdir(parents=True, exist_ok=True)
        self._our_model_yhat_filepath.parent.mkdir(parents=True, exist_ok=True)

    async def save_bronze(self, df: pd.DataFrame) -> None:
        """Dump df to the bronze filepath."""
        df.to_pickle(self._bronze_filepath)

    async def save_silver(self, df: pd.DataFrame) -> None:
        """Dump df to the silver filepath."""
        df.to_pickle(self._silver_filepath)

    async def save_gold(self, df: pd.DataFrame) -> None:
        """Dump df to the gold filepath."""
        df.to_pickle(self._gold_filepath)

    async def save_entsoe_mape(self, mape_df: pd.DataFrame) -> dict[str, float]:
        """Dump mape_df to the ENTSO-E MAPE filepath."""
        mape = {
            "1h": mape_df.mape.iloc[0],
            "24h": mape_df.mape.iloc[1],
            "7d": mape_df.mape.iloc[2],
            "4w": mape_df.mape.iloc[3],
        }
        joblib.dump(mape, self._entsoe_mape_filepath)
        return mape

    async def save_our_model_mape(self, mape_df: pd.DataFrame) -> dict[str, float]:
        """Dump mape_df to our model's MAPE filepath."""
        mape = {
            "1h": mape_df.mape.iloc[0],
            "24h": mape_df.mape.iloc[1],
            "7d": mape_df.mape.iloc[2],
            "4w": mape_df.mape.iloc[3],
        }
        joblib.dump(mape, self._our_model_mape_filepath)
        return mape
    
    async def save_walkforward_yhat(self, yhat: pd.Series) -> None:
        """Dump yhat to the walkforward yhat's filepath."""
        yhat.to_pickle(self._our_model_yhat_filepath)

    async def save_our_model_yhat(self, yhat: pd.Series) -> None:
        """Dump yhat to our model yhat's filepath."""
        yhat.to_pickle(self._our)
    
    async def load_bronze(self) -> pd.DataFrame:
        """Load df from the bronze filepath."""
        return pd.read_pickle(self._bronze_filepath)

    async def load_silver(self) -> pd.DataFrame:
        """Load df from the silver filepath."""
        return pd.read_pickle(self._silver_filepath)

    async def load_gold(self) -> pd.DataFrame:
        """Load df from the gold filepath."""
        return pd.read_pickle(self._gold_filepath)
    
    async def fetch_latest_load_ts(self) -> pd.Timestamp:
        """Fetch the timestamp of the latest load."""
        gold_df = await self.load_gold()
        return gold_df.dropna(subset=("24h_later_load")).index.max()