"""Service responsible for measuring the performance of a time-series prediction model."""

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def compute_mape(y_true_col: str, y_pred_col: str, data: pd.DataFrame, timedelta_strs: list[str]) -> dict[str, float]:
    """Measure the Mean Absolute Percentage Error (MAPE) between the ground-truth and a prediction,
    for each timedelta->latest-ts-in-data.index

    Any rows containing NaNs are not considered.
    If no rows could be considered to compute a MAPE, np.nan is set as the MAPE.

    Args:
        y_true_col (str): Ground-truth's column name in `data`
        y_pred_col (str): Prediction's column name in `data`
        data (pd.DataFrame): Dataframe containing the ground-truth and prediction, with a pd.DatetimeIndex
        timedelta_strs (list[str]): Timedeltas from which we should compute the MAPE, starting at the latest ts in `data`.
                                    Each timedelta_str must be in the format expected by pd.to_timedelta, e.g. '7w1d2h' or '1d' 
                                

    Returns:
        dict[timedelta_str]: Series containing the MAPE values -- under 'mape' -- for each timedelta.
                    The index of row corresponds to the starting timestamp from which the MAPE was computed.
                    The first row corresponds to the first timedelta passed.
    """

    # Check the input is as we expect
    assert isinstance(data.index, pd.DatetimeIndex)
    assert data.index.is_unique
    assert y_true_col in data.columns
    assert y_pred_col in data.columns

    data = data.dropna(subset=(y_true_col, y_pred_col)).sort_index()
    max_ts = data.index.max()

    timedelta_str_to_mape = {}
    for timedelta_str in sorted(timedelta_strs):
        td = pd.to_timedelta(timedelta_str)
        curr_starting_ts = max_ts - td
        curr_data = data[data.index >= curr_starting_ts]

        curr_mape = np.nan
        if len(curr_data):
            curr_mape = mean_absolute_percentage_error(y_true=curr_data[y_true_col], y_pred=curr_data[y_pred_col]) * 100
        timedelta_str_to_mape[timedelta_str] = curr_mape

    return timedelta_str_to_mape 