"""Service responsible for measuring the performance of a time-series prediction model."""

from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error


def compute_mape(
    y_true_col: str,
    y_pred_col: str,
    data: pd.DataFrame,
    timedeltas: list[timedelta],
) -> pd.Series:
    """Measure the Mean Absolute Percentage Error (MAPE) between the ground-truth and a prediction,
    for each period between the latest ts in data.index and spanning timedelta.
    Any rows containing NaNs are not considered.
    If no rows could be considered to compute a MAPE, np.nan is set as the MAPE.

    Args:
        y_true_col (str): Ground-truth's column name in `data`
        y_pred_col (str): Prediction's column name in `data`
        data (pd.DataFrame): Dataframe containing the ground-truth and prediction, with a pd.DatetimeIndex
        timedeltas (list[timedelta]): Timedelta from which we should compute the MAPE,
                                        starting at the latest ts in `data`.

    Returns:
        pd.Series: Series containing the MAPE values -- under 'mape' -- for each timedelta.
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

    starting_ts_to_mape = {}
    for timedelta in sorted(timedeltas):
        curr_starting_ts = max_ts - timedelta
        curr_data = data[data.index >= curr_starting_ts]

        curr_mape = np.nan
        if len(curr_data):
            curr_mape = (
                mean_absolute_percentage_error(
                    y_true=curr_data[y_true_col],
                    y_pred=curr_data[y_pred_col],
                )
                * 100
            )
        starting_ts_to_mape[curr_starting_ts] = curr_mape

    return pd.DataFrame(
        {
            "mape": starting_ts_to_mape.values(),
        },
        index=pd.DatetimeIndex(starting_ts_to_mape.keys()),
    )
