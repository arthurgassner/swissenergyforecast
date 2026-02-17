import os
from datetime import datetime, timedelta

import pandas as pd
from entsoe import EntsoePandasClient
import pytest

from app.core.config import get_settings
from app.services import data_loading_service


def test__split_yearly():
    """Ensure time intervals are split as expected."""

    # Given 
    start_ts = pd.Timestamp("2014-01-01 00:00", tz="Europe/Zurich")
    end_ts = pd.Timestamp("2016-01-02 00:00", tz="Europe/Zurich")
    
    # When
    start_end_timestamps = data_loading_service._split_yearly(start_ts=start_ts, end_ts=end_ts)

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
        data_loading_service._split_yearly(start_ts=start_ts, end_ts=end_ts)


def test__query_load_and_forecast__future_ts():
    """Querying the ENTSO-E API with a timestamp 48h in the future should result in an empty df."""

    # given
    entsoe_client = EntsoePandasClient(api_key=get_settings().ENTSOE_API_KEY)
    future_ts = pd.Timestamp(datetime.now(), tz="Europe/Zurich") + timedelta(hours=48)

    # when
    fetched_df = data_loading_service._query_load_and_forecast(
        entsoe_client=entsoe_client, start_ts=future_ts, end_ts=future_ts + timedelta(hours=24)
    )

    # then
    expected_df = pd.DataFrame(
        columns=["Forecasted Load", "Actual Load"],
        dtype=float,
        index=pd.DatetimeIndex([], dtype="datetime64[ns, Europe/Zurich]"),
    )
    assert (expected_df == fetched_df).all().all()  # same values
    assert all(c1 == c2 for c1, c2 in zip(expected_df.columns, fetched_df.columns))  # same column names
    assert (expected_df.dtypes == fetched_df.dtypes).all()  # same dtypes
    assert (expected_df.index == fetched_df.index).all()  # same index


def test__query_load_and_forecast__24h_ago_ts():
    """Querying the ENTSO-E API with a timestamp 24h ago."""

    # given
    entsoe_client = EntsoePandasClient(api_key=get_settings().ENTSOE_API_KEY)
    now_ts = pd.Timestamp(datetime.now(), tz="Europe/Zurich")

    # when
    fetched_df = data_loading_service._query_load_and_forecast(entsoe_client=entsoe_client, start_ts=now_ts - timedelta(hours=24))

    # then

    # data
    assert len(fetched_df.columns) == 2  # 2 columns
    assert fetched_df.columns[0] == "Forecasted Load" and fetched_df.columns[1] == "Actual Load"
    # data is hourly, so we should not have more than 49 -- 49 if hour change happened in the last 48h -- datapoints
    assert len(fetched_df) <= 49
    # all datapoints after now should be NaN
    mask = fetched_df.index > now_ts
    assert fetched_df.loc[mask, "Actual Load"].isna().all()

    # index
    assert isinstance(fetched_df.index, pd.DatetimeIndex)
    assert fetched_df.index.is_monotonic_increasing
    assert fetched_df.index.is_unique

    # dtypes
    assert (fetched_df.dtypes == "float64").all()  # correct dtype
    assert fetched_df.index.dtype == "datetime64[ns, Europe/Zurich]"  # correct timezone


def test__query_load_and_forecast__specitic_ts():
    """Querying the ENTSO-E API with a timestamp whose load/forecast is known."""

    # given
    entsoe_client = EntsoePandasClient(api_key=get_settings().ENTSOE_API_KEY)

    # when
    fetched_df = data_loading_service._query_load_and_forecast(
        entsoe_client=entsoe_client,
        start_ts=pd.Timestamp("20240701 00:30", tz="Europe/Zurich"),
        end_ts=pd.Timestamp("20240701 01:30", tz="Europe/Zurich"),
    )

    # then

    # data
    assert len(fetched_df.columns) == 2  # 2 columns
    assert fetched_df.columns[0] == "Forecasted Load" and fetched_df.columns[1] == "Actual Load"

    # data is hourly, so we should have exactly one datapoint
    assert len(fetched_df) == 1
    # And no NaN, as that data should be known
    assert fetched_df["Actual Load"].isna().sum() == 0
    # And the data should match the historically-known data, as seen on the ENTSO-E website
    # Forecasted Load [6.1.A] 01:00 - 02:00 07.10.2024
    assert fetched_df["Forecasted Load"].iloc[0] == 5693
    # Actual Load [6.1.A] 01:00 - 02:00 07.10.2024 --> Note that this can be updated by ENTSO-E
    assert fetched_df["Actual Load"].iloc[0] == 4994

    # index
    assert isinstance(fetched_df.index, pd.DatetimeIndex)
    assert fetched_df.index.is_monotonic_increasing
    assert fetched_df.index.is_unique

    # dtypes
    assert (fetched_df.dtypes == "float64").all()  # correct dtype
    assert fetched_df.index.dtype == "datetime64[ns, Europe/Zurich]"  # correct timezone
