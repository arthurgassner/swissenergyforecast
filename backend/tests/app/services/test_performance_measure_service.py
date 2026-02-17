from datetime import timedelta

import numpy as np
import pandas as pd

from app.services import performance_measure_service


def test_compute_mape__perfect_prediction():
    """Check that the MAPE of a perfect prediction is 0.0."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [101.0, 202.0, 303.0],
            "Actual Load": [101.0, 202.0, 303.0],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240102 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240103 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when
    timedeltas = [timedelta(days=3 + i) for i in range(5)]
    mape_df = performance_measure_service.compute_mape(
        "Actual Load",
        "Forecasted Load",
        data=df,
        timedeltas=timedeltas,
    )

    # then
    assert len(mape_df) == len(timedeltas)  # 5 timedeltas where given
    assert (mape_df.mape == 0.0).all()  # The predictions are always without error


def test_compute_mape__mape_changes():
    """Check that the MAPE changes with the increased deltatimes."""

    # Given a df of the expected format
    df = pd.DataFrame(
        {
            "Forecasted Load": [101.0, 101.0, 303.0],
            "Actual Load": [101.0, 202.0, 303.0],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("20240101 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240102 23:45", tz="Europe/Zurich"),
                pd.Timestamp("20240103 23:45", tz="Europe/Zurich"),
            ]
        ),
    )

    # when
    timedeltas = [timedelta(hours=1), timedelta(hours=24), timedelta(days=2)]
    mape_df = performance_measure_service.compute_mape(
        "Actual Load",
        "Forecasted Load",
        data=df,
        timedeltas=timedeltas,
    )

    # then
    assert len(mape_df) == len(timedeltas)  # As many rows as given timedeltas
    assert np.isclose(mape_df.mape.iloc[0], 0.0, atol=0.1)
    assert np.isclose(mape_df.mape.iloc[1], 25.0, atol=0.1)
    assert np.isclose(mape_df.mape.iloc[2], 16.6, atol=0.1)
