from datetime import datetime, timedelta

from entsoe.entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from human_readable import precise_delta
from loguru import logger
import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential, wait_fixed



class ENTSOEClient:
    def __init__(self, api_key: str) -> None:
        self._entsoe_pandas_client = EntsoePandasClient(api_key=api_key)

    @staticmethod
    def _split_yearly(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
        """Split a time interval (start_ts, end_ts) into an ordered list of yearly intervals, i.e.
        [(start_ts, start_ts + 1year), (start_ts + 1year, start_ts + 2years), ..., (start_ts + nyears, end_ts)]

        Args:
            start_ts (pd.Timestamp): Starting timestamp
            end_ts (pd.Timestamp): End timestamp

        Returns:
            list[tuple[pd.Timestamp, pd.Timestamp]]: Ordered list of yearly intervals.
        """
        
        if start_ts > end_ts:
            error_str = f"start_ts ({start_ts}) must be <= end_ts ({end_ts})"
            logger.error(error_str)
            raise ValueError(error_str)
        
        start_end_timestamps = []
        curr_start_ts = start_ts
        curr_end_ts = min(end_ts, curr_start_ts + timedelta(days=365))
        while curr_end_ts < end_ts:
            start_end_timestamps.append((curr_start_ts, curr_end_ts))
            curr_start_ts = curr_end_ts
            curr_end_ts = min(end_ts, curr_start_ts + timedelta(days=365))
        start_end_timestamps.append((curr_start_ts, end_ts))
        return start_end_timestamps
    
    @staticmethod
    def _raise_if_unexpected_format(df: pd.DataFrame) -> None:
        """Raise ValueError if df is formatted differently than expected from the ENTSO-E API, i.e. a pd.DataFrame with
            - columns: ('Forcasted Load', 'Actual Load')
            - dtypes: float64
            - index: DateTimeIndex with dtype datetime64[us, Europe/Zurich]
        """
        # Ensure df is actually a dataframe
        if type(df) is not pd.DataFrame:
            error_str = f"ENSTO-E-sourced pd.DataFrame is not a pd.DataFrame; it is: {type(df)}"
            logger.error(error_str)
            raise ValueError(error_str)

        # Ensure columns are as expected
        expected_columns = ('Forecasted Load', 'Actual Load')
        if len(df.columns) != 2 or any(expected_columns != df.columns):
            error_str = f"ENSTO-E-sourced pd.DataFrame's columns should be: {expected_columns} but were: {df.columns}"
            logger.error(error_str)
            raise ValueError(error_str)
        

        # Ensure dtypes are as expected
        expected_dtype = 'float64'
        if any(df.dtypes != expected_dtype):
            error_str = f"ENSTO-E-sourced pd.DataFrame's dtypes should all be {expected_dtype} but were: {df.dtypes}"
            logger.error(error_str)
            raise ValueError(error_str)
        
        # Ensure index is as expected
        excepted_index_dtype = "datetime64[us, Europe/Zurich]"
        if type(df.index) is not pd.DatetimeIndex or df.index.dtype != excepted_index_dtype:
            error_str = f"ENSTO-E-sourced pd.DataFrame's index should be a pd.DateTimeIndex with dtype {excepted_index_dtype} but was: {type(df.index)} (index.dtype: {df.index.dtype})"
            logger.error(error_str)
            raise ValueError(error_str)

    
    @retry(retry=retry_if_exception_type(requests.ConnectionError), stop=stop_after_attempt(10), wait=wait_fixed(5))
    def _query_load_and_forecast(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp | None = None) -> pd.DataFrame:
        """Query the ENTSO-E API for the load and forecast data from `start_ts` to `end_ts`."""

        human_delta_str = precise_delta(end_ts - start_ts, minimum_unit="seconds")
        logger.info(f"Asking the ENTSO-E API for load/forecast data between {start_ts} -> {end_ts} ({human_delta_str})")

        try:
            load_and_forecast_df = self._entsoe_pandas_client.query_load_and_forecast(country_code="CH", start=start_ts, end=end_ts)
            ENTSOEClient._raise_if_unexpected_format(load_and_forecast_df)
        
        except NoMatchingDataError: # No data found for the requested time span
            logger.warning(f"No data available between {start_ts} -> {end_ts} ({human_delta_str})")
            
            # empty dataframe
            load_and_forecast_df = pd.DataFrame(columns=["Forecasted Load", "Actual Load"], dtype=float, index=pd.DatetimeIndex([], dtype="datetime64[us, Europe/Zurich]"))
        
        except requests.ConnectionError as e:
            logger.warning(f"Thrown {e}.")
            raise e

        return load_and_forecast_df

    def query_load_and_forecast(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        """Query the ENTSO-E API for the load and forecast data from `start_ts` to `end_ts`, breaking it down into yearly-queries.
        
        It seems that the ENTSO-E API tends to terminate the connection when asking for 10 years of data.
        Hence, the data is fetched year-by-year -- as it seems to lower the odds of aborted connections.

        Args:
            start_ts (pd.Timestamp): Starting ts (tz="Europe/Zurich") of the requested data
            end_ts (pd.Timestamp, optional): Ending ts (tz="Europe/Zurich") of the requested data

        Returns:
            pd.DataFrame: Fetched data.
                            - columns: ('Forcasted Load', 'Actual Load')
                            - dtypes: float64
                            - index: datetime64[us, Europe/Zurich]
                            Empty dataframe if no data could be found
        """

        # Split up the query into yearly queries
        start_end_timestamps = ENTSOEClient._split_yearly(start_ts, end_ts)

        # Send each yearly-query to the ENTSO-E API
        # TODO run them all in parallel
        # TODO async ?
        load_and_forecast_dfs = []
        for curr_start_ts, curr_end_ts in start_end_timestamps:
            load_and_forecast_df = self._query_load_and_forecast(curr_start_ts, curr_end_ts)
            load_and_forecast_dfs.append(load_and_forecast_df)

        return pd.concat(load_and_forecast_dfs)
    
    def fetch_latest_load_and_forecast(self) -> pd.DataFrame:
        """Query the ENTSO-E API for the load & forecast data, from 01.01.2014 to now+24h."""
        # start_ts = pd.Timestamp("2014-01-01 00:00", tz="Europe/Zurich")
        start_ts = pd.Timestamp("2025-01-01 00:00", tz="Europe/Zurich") # TODO REMOVE
        end_ts = pd.Timestamp(datetime.now() + timedelta(hours=24), tz="Europe/Zurich")
        return self.query_load_and_forecast(start_ts, end_ts)
