"""Service responsible for extracted features out of the cleaned data."""

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


def _n_hours_ago_load(df: pd.DataFrame, n_hours: int) -> pd.Series:
    """For each timestamps in the index, compute the load n_hours ago

    Assume that each row's index is the current timestamp.
    That is, when we say "timedelta ago from now", we mean "timedelta ago from this timestamp".

    Args:
        df (pd.DataFrame): Dataframe containing the `24h_later_load`, whose index refers to "now" when saying "24h later".
        n_hours (int): How many hours ago is the load of interest ?

    Returns:
        pd.Series: Series whose index is the same as `df`, and whose values are the loads n_hours ago from their index.
    """

    assert "24h_later_load" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.freq == "h"

    return df["24h_later_load"].shift(24 + n_hours)


def _rolling_window(df: pd.DataFrame, n_hours: int, stat: Callable) -> pd.Series:
    """For each timestamps in the index, compute the `stat` over the date comprised between that timestamp and `timedelta` ago.

    Args:
        df (pd.DataFrame): Dataframe containing the `24h_later_load`, whose index refers to "now" when saying "24h later".
        n_hours (int): Over how many hours should we compute the statistics
        stat (Callable): Statistic function

    Returns:
        pd.Series: Series whose index is the same as `df`, and whose values are the statistics computed over `n_hours` hours.
    """

    assert "24h_later_load" in df.columns
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.freq == "h"

    # Compute the last-hour load for each row
    last_hour_loads = _n_hours_ago_load(df, n_hours=1)
    return last_hour_loads.rolling(window=n_hours, min_periods=1).apply(stat)


def extract_features(df: pd.DataFrame, out_df_filepath: Path) -> None:
    """Extract the features.

    Args:
        df (pd.DataFrame): Dataframe whose features must be extracted (.pickle)
        out_df_filepath (Path): Filepath where to dump the extracted features (.pickle)
    """

    # Enrich the df with the datetime attributes
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["day"] = df.index.day
    df["hour"] = df.index.hour
    df["weekday"] = df.index.weekday

    # Enrich each row with previous loads: 1h-ago, 2h-ago, 3h-ago, 24h-ago, 7days-ago
    df["1h_ago_load"] = _n_hours_ago_load(df, n_hours=1)
    df["2h_ago_load"] = _n_hours_ago_load(df, n_hours=2)
    df["3h_ago_load"] = _n_hours_ago_load(df, n_hours=3)
    df["24h_ago_load"] = _n_hours_ago_load(df, n_hours=24)
    df["7d_ago_load"] = _n_hours_ago_load(df, n_hours=7 * 24)

    # Enrich the df with statistics
    df["8h_min"] = _rolling_window(df, n_hours=8, stat=np.nanmin)
    df["8h_max"] = _rolling_window(df, n_hours=8, stat=np.nanmax)
    df["8h_median"] = _rolling_window(df, n_hours=8, stat=np.nanmedian)

    df["24h_min"] = _rolling_window(df, n_hours=24, stat=np.nanmin)
    df["24h_max"] = _rolling_window(df, n_hours=24, stat=np.nanmax)
    df["24h_median"] = _rolling_window(df, n_hours=24, stat=np.nanmedian)

    df["7d_min"] = _rolling_window(df, n_hours=7 * 24, stat=np.nanmin)
    df["7d_max"] = _rolling_window(df, n_hours=7 * 24, stat=np.nanmax)
    df["7d_median"] = _rolling_window(df, n_hours=7 * 24, stat=np.nanmedian)

    # Dump to output df
    out_df_filepath.parent.mkdir(parents=True, exist_ok=True)  # Ensure the folderpath exists
    df.to_pickle(out_df_filepath)
