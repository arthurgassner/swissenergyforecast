from datetime import timedelta

import numpy as np
import pandas as pd

from app.schemas.forecast import MAPE

def test_compute_mapes__perfect_prediction():
    """Check that the MAPE of a perfect prediction is 0.0."""

    # Given a df of the expected format
    index = pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240102 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240103 23:45", tz="Europe/Zurich"),
            ]
        )
    y = pd.Series([101.0, 202.0, 303.0], index=index)
    timedelta_strs = ["3D", "4D", "5D", "6D"]

    # when
    mapes = MAPE.compute_mapes(y, y, timedelta_strs=timedelta_strs)

    # then
    assert len(mapes) == len(timedelta_strs)  # 4 timedelta_strs where given
    assert all(e.score == 0.0 for e in mapes) # The predictions are always without error

def test_compute_mapes__mape_changes():
    """Check that the MAPE changes with the increased deltatimes."""

    # Given a df of the expected format
    index = pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240102 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240103 23:45", tz="Europe/Zurich"),
            ]
        )
    y = pd.Series([101.0, 202.0, 303.0], index=index)
    yhat = pd.Series([101.0, 101.0, 303.0], index=index)
    timedelta_strs = ["1h", "24h", "2D"]

    # when
    mapes = MAPE.compute_mapes(y, yhat, timedelta_strs=timedelta_strs)

    # then
    assert len(mapes) == len(timedelta_strs)  # As many rows as given timedeltas
    assert np.isclose(mapes[0].score, 0.0, atol=0.1)
    assert np.isclose(mapes[1].score, 25.0, atol=0.1)
    assert np.isclose(mapes[2].score, 16.6, atol=0.1)
