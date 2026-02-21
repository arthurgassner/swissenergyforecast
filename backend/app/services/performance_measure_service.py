"""Service responsible for measuring the performance of a time-series prediction model."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def compute_mape(y: pd.Series, yhat: pd.Series, timedelta_strs: list[str]) -> dict[str, float]:
    """Measure the Mean Absolute Percentage Error (MAPE) between the ground-truth and a prediction,
    for each timedelta->latest-ts-in-data.index

    Any rows containing NaNs are not considered.
    If no rows could be considered to compute a MAPE, np.nan is set as the MAPE.

    Args:
        y (pd.Series): Ground-truth's Series, with pd.DatetimeIndex index
        yhat (pd.Series): Prediction's Series, with pd.DatetimeIndex index
        timedelta_strs (list[str]): Timedeltas from which we should compute the MAPE, starting at the latest ts in `data`.
                                    Each timedelta_str must be in the format expected by pd.to_timedelta, e.g. '7w1d2h' or '1d' 

    Returns:
        dict[timedelta_str]: Series containing the MAPE values -- under 'mape' -- for each timedelta.
                    The index of row corresponds to the starting timestamp from which the MAPE was computed.
                    The first row corresponds to the first timedelta passed.
    """

    # Check the input is as we expect
    assert isinstance(y, pd.Series)
    assert isinstance(yhat, pd.Series)
    assert y.index.is_unique
    assert yhat.index.is_unique
    assert isinstance(y.index, pd.DatetimeIndex)
    assert isinstance(yhat.index, pd.DatetimeIndex)
    assert len(y) == len(yhat)
    assert (y.index == yhat.index).all()

    # Only keep rows with both a y and yhat value
    nan_mask = y.isna() | yhat.isna()
    y = y[~nan_mask].sort_index()
    yhat = yhat[~nan_mask].sort_index()

    # Latest ts in the Series
    last_load_ts = y.index.max()

    timedelta_str_to_mape = {}
    for timedelta_str in sorted(timedelta_strs):
        td = pd.to_timedelta(timedelta_str)
        curr_starting_ts = last_load_ts - td
        curr_data_mask = y.index >= curr_starting_ts

        curr_mape = np.nan
        if len(curr_data_mask):
            curr_mape = mean_absolute_percentage_error(y_true=y[curr_data_mask], y_pred=yhat[curr_data_mask]) * 100
        timedelta_str_to_mape[timedelta_str] = curr_mape

    return timedelta_str_to_mape 