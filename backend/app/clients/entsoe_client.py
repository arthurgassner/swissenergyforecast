import asyncio
from datetime import datetime, timedelta

import pandas as pd
import requests
from app.core.config import Settings
from entsoe.entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from human_readable import precise_delta
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed


class ENTSOEClient:
    def __init__(self, entsoe_pandas_client: EntsoePandasClient, settings: Settings) -> None:
        self._entsoe_pandas_client = entsoe_pandas_client
        self.max_concurrent_requests = settings.MAX_CONCURRENT_REQUESTS

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
        expected_columns = ("Forecasted Load", "Actual Load")
        if len(df.columns) != 2 or any(expected_columns != df.columns):
            error_str = f"ENSTO-E-sourced pd.DataFrame's columns should be: {expected_columns} but were: {df.columns}"
            logger.error(error_str)
            raise ValueError(error_str)

        # Ensure dtypes are as expected
        expected_dtype = "float64"
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
    def _query_load_and_forecast(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        """Query the ENTSO-E API for the load and forecast data from `start_ts` to `end_ts`."""

        human_delta_str = precise_delta(end_ts - start_ts, minimum_unit="seconds")
        logger.info(f"Asking the ENTSO-E API for load/forecast data between {start_ts} -> {end_ts} ({human_delta_str})")

        try:
            load_and_forecast_df = self._entsoe_pandas_client.query_load_and_forecast(
                country_code="CH", start=start_ts, end=end_ts
            )
            ENTSOEClient._raise_if_unexpected_format(load_and_forecast_df)

        except NoMatchingDataError:  # No data found for the requested time span
            logger.warning(f"No data available between {start_ts} -> {end_ts} ({human_delta_str})")

            # empty dataframe
            load_and_forecast_df = pd.DataFrame(
                columns=["Forecasted Load", "Actual Load"],
                dtype=float,
                index=pd.DatetimeIndex([], dtype="datetime64[us, Europe/Zurich]"),
            )

        except requests.ConnectionError as e:
            logger.warning(f"Thrown {e}.")
            raise e

        return load_and_forecast_df

    async def _semaphore_query(self, semaphore: asyncio.Semaphore, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        """Helper to run the synchronous API call in a thread with a semaphore."""
        async with semaphore:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._query_load_and_forecast, start_ts, end_ts)

    async def query_load_and_forecast(self, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
        """Query the ENTSO-E API for the load and forecast data from `start_ts` to `end_ts`, breaking it down into yearly-queries.

        It seems that the ENTSO-E API tends to terminate the connection when asking for 10 years of data.
        Hence, the data is fetched year-by-year -- as it seems to lower the odds of aborted connections.

        Args:
            start_ts (pd.Timestamp): Starting ts (tz="Europe/Zurich") of the requested data
            end_ts (pd.Timestamp): Ending ts (tz="Europe/Zurich") of the requested data
            max_concurrent_requests (int, optional): How many concurrent request to the ENTSO-E API can be done

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
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        coroutines = [self._semaphore_query(semaphore, start_ts, end_ts) for start_ts, end_ts in start_end_timestamps]
        load_and_forecast_dfs = await asyncio.gather(*coroutines, return_exceptions=True)

        # Filter returned exceptions, if any
        raised_exceptions = [(idx, e) for idx, e in enumerate(load_and_forecast_dfs) if isinstance(e, Exception)]
        if raised_exceptions:
            for idx, e in raised_exceptions:
                logger.error(f"Couldn't fetch ENTSO-E data from {start_end_timestamps[idx][0]} -> {start_end_timestamps[idx][1]}")
            raise RuntimeError(f"Fetching data from ENTSO-E API raised: {raised_exceptions}")

        return pd.concat(load_and_forecast_dfs)

    async def fetch_latest_load_and_forecast(self) -> pd.DataFrame:
        """Query the ENTSO-E API for the load & forecast data, from 01.01.2014 to now+24h."""
        start_ts = pd.Timestamp("2014-01-01 00:00", tz="Europe/Zurich")
        end_ts = pd.Timestamp(datetime.now() + timedelta(hours=24), tz="Europe/Zurich")
        return await self.query_load_and_forecast(start_ts, end_ts)
