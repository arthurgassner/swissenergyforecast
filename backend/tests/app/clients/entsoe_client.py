import os
from datetime import datetime, timedelta

import pandas as pd
from entsoe import EntsoePandasClient
import pytest

from app.core.config import get_settings
from app.clients.entsoe_client import ENTSOEClient


def test__split_yearly():
    """Ensure time intervals are split as expected."""

    # Given 
    start_ts = pd.Timestamp("2014-01-01 00:00", tz="Europe/Zurich")
    end_ts = pd.Timestamp("2016-01-02 00:00", tz="Europe/Zurich")
    
    # When
    start_end_timestamps = ENTSOEClient._split_yearly(start_ts=start_ts, end_ts=end_ts)

    # Then
    assert len(start_end_timestamps) == 3
    assert start_end_timestamps[0][0] == start_ts
    assert start_end_timestamps[-1][1] == end_ts

    first_split_timedelta = (start_end_timestamps[0][1] - start_end_timestamps[0][0])
    assert first_split_timedelta == pd.Timedelta(days=365) # Since the start_ts is the earliest ts in 2014

    last_split_timedelta = (start_end_timestamps[-1][1] - start_end_timestamps[-1][0])
    assert last_split_timedelta == pd.Timedelta(days=1) # Since end_ts is 1 day after the start of the year


def test__split_yearly__end_before_start():
    """Ensure time intervals raises a ValueError when the end is before the start"""

    # Given 
    start_ts = pd.Timestamp("2014-01-02 00:00", tz="Europe/Zurich")
    end_ts = pd.Timestamp("2014-01-01 00:00", tz="Europe/Zurich")
    
    # When - Then
    with pytest.raises(ValueError):
         ENTSOEClient._split_yearly(start_ts=start_ts, end_ts=end_ts)