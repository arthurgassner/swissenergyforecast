from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ROOT_FOLDERPATH: Path = Path(__file__).resolve().parent.parent.parent
    DATA_FOLDERPATH: Path = ROOT_FOLDERPATH / "data"
    BRONZE_DF_FILEPATH: Path = DATA_FOLDERPATH / "bronze" / "df.pickle"
    SILVER_DF_FILEPATH: Path = DATA_FOLDERPATH / "silver" / "df.pickle"
    GOLD_DF_FILEPATH: Path = DATA_FOLDERPATH / "gold" / "df.pickle"
    WALKFORWARD_YHAT_FILEPATH: Path = DATA_FOLDERPATH / "walkforward_yhat.pickle"
    YHAT_FILEPATH: Path = DATA_FOLDERPATH / "yhat.pickle"
    OUR_MODEL_MAPE_FILEPATH: Path = DATA_FOLDERPATH / "our_model_mape.joblib"
    ENTSOE_MAPE_FILEPATH: Path = DATA_FOLDERPATH / "entsoe_mape.joblib"
    LOGS_FILEPATH: Path = DATA_FOLDERPATH / "logs" / ".log"
    ENTSOE_API_KEY: str 
    MODEL_N_ESTIMATORS: int

@lru_cache
def get_settings() -> Settings:
    return Settings(_env_file='.env') # TODO rely on a environment-selected dev file
