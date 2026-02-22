from datetime import datetime
from unittest.mock import MagicMock, Mock

import numpy as np
import pandas as pd
import pytest

from app.clients.entsoe_client import ENTSOEClient
from app.core.config import Settings, get_settings

@pytest.fixture
def settings() -> Settings:
    """Mocked settings, for testing."""
    return get_settings()

@pytest.fixture
def entsoe_client(settings: Settings) -> ENTSOEClient:
    """Mocked ENTSOEClient"""
    
    mocked_entsoe_pandas_client = MagicMock()

    def dynamic_load_side_effect(country_code, start, end, **kwargs) -> pd.DataFrame:
        """Generates a DataFrame with hourly frequency between start and end."""
        if start > pd.Timestamp.now(tz="Europe/Zurich"):
            empty_index = pd.DatetimeIndex([], dtype="datetime64[us, Europe/Zurich]")
            return pd.DataFrame(columns=["Forecasted Load", "Actual Load"], dtype=float, index=empty_index)

        # Generate mock data proportional to the length of the index
        index = pd.date_range(start=start, end=end, freq="h",  tz="Europe/Zurich", unit='us', inclusive="left")
        return pd.DataFrame({
            "Forecasted Load": np.random.uniform(100.0, 500.0, size=len(index)),
            "Actual Load": np.random.uniform(100.0, 500.0, size=len(index))
        }, index=index)
    
    # Mocking 'query_load_and_forecast' method
    mocked_entsoe_pandas_client.query_load_and_forecast.side_effect = dynamic_load_side_effect
    
    return ENTSOEClient(mocked_entsoe_pandas_client, settings)