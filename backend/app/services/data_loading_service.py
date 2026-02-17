"""Service responsible for downloading the data from the ENTSO-E servers."""

import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError
from human_readable import precise_delta
from loguru import logger

def _split_yearly(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Split a time interval (start_ts, end_ts) into an ordered list of yearly intervals.

    Args:
        start_ts (pd.Timestamp): Starting timestamp
        end_ts (pd.Timestamp): End timestamp

    Returns:
        list[tuple[pd.Timestamp, pd.Timestamp]]: Ordered list of yearly intervals.
    """
    if start_ts > end_ts:
        logger.error(f"start_ts ({start_ts}) must be <= end_ts ({end_ts})")
        raise ValueError
    
    start_end_timestamps = []
    curr_start_ts = start_ts
    curr_end_ts = min(end_ts, curr_start_ts + timedelta(days=365))
    while curr_end_ts < end_ts:
        start_end_timestamps.append((curr_start_ts, curr_end_ts))
        curr_start_ts = curr_end_ts
        curr_end_ts = min(end_ts, curr_start_ts + timedelta(days=365))
    start_end_timestamps.append((curr_start_ts, end_ts))
    return start_end_timestamps

def _query_load_and_forecast(
    entsoe_client: EntsoePandasClient,
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp | None = None,
    max_retries: int = 10,
) -> pd.DataFrame:
    """Query the ENTSO-E API for the load and forecast data from `start_ts` to `end_ts`, breaking it down into yearly-queries.

    It seems that the ENTSO-E API tends to terminate the connection when asking for 10 years of data.
    Hence, the data is fetched year-by-year -- as it seems to lower the odds of aborted connections.

    Args:
        entsoe_client (EntsoePandasClient): ENTSO-E client
        start_ts (pd.Timestamp): Starting ts (tz="Europe/Zurich") of the requested data
        end_ts (pd.Timestamp, optional): Ending ts (tz="Europe/Zurich") of the requested data, default to 24h away from now.
        max_retries (int): Max amount of retries, as the ENTSO-E API tends to abort the connection.

    Returns:
        pd.DataFrame: Fetched data.
                        - columns: ('Forcasted Load', 'Actual Load')
                        - dtypes: float64
                        - index: datetime64[ns, Europe/Zurich]
                        Empty dataframe if no data could be found
    """
    if end_ts is None:
        end_ts = pd.Timestamp(datetime.now() + timedelta(hours=24), tz="Europe/Zurich")

    # Split up the query into yearly queries
    start_end_timestamps = _split_yearly(start_ts, end_ts)

    # Send each yearly-query to the ENTSO-E API
    fetched_dfs = []
    for curr_start_ts, curr_end_ts in start_end_timestamps:
        logger.info(
            f"Asking the ENTSO-E API for load/forecast data between {curr_start_ts} -> {curr_end_ts} ({precise_delta(curr_end_ts - curr_start_ts, minimum_unit="seconds")})"
        )
        n_retries = 0
        while n_retries < max_retries:
            try:
                fetched_df = entsoe_client.query_load_and_forecast(
                    country_code="CH", start=curr_start_ts, end=curr_end_ts
                )
                break
            except NoMatchingDataError:
                logger.warning(
                    f"No data available between {curr_start_ts} -> {curr_end_ts} ({precise_delta(curr_end_ts - curr_start_ts, minimum_unit="seconds")})"
                )
                fetched_df = pd.DataFrame(  # empty dataframe
                    columns=["Forecasted Load", "Actual Load"],
                    dtype=float,
                    index=pd.DatetimeIndex([], dtype="datetime64[ns, Europe/Zurich]"),
                )
                break
            except requests.ConnectionError as e:
                n_retries += 1
                if not n_retries < max_retries:
                    raise e
                logger.warning(f"Thrown {e}. Retrying {n_retries}/{max_retries}...")
            time.sleep(1)  # Wait time between requests [s]
        fetched_dfs.append(fetched_df)

    return pd.concat(fetched_dfs)


def fetch_df(entsoe_client: EntsoePandasClient, out_df_filepath: Path) -> None:
    """Fetch the forecast/load data from the ENTSO-E API, and dump it to disk.

    Args:
        entsoe_client (EntsoePandasClient): ENTSO-E client
        out_df_filepath (Path): Filepath where the dataframe (.pickle) should be stored.
    """

    # Fetch loads and forecasts
    fetched_df = _query_load_and_forecast(entsoe_client, start_ts=pd.Timestamp("2014-01-01 00:00", tz="Europe/Zurich"))

    # Dump to output df
    # Ensure the folderpath exists
    out_df_filepath.parent.mkdir(parents=True, exist_ok=True)
    fetched_df.to_pickle(out_df_filepath)
