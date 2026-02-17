import numpy as np
import pandas as pd

from app.core.model import Model


def test__train_predict__missing_query_ts():
    """Check whether Xy.index missing the query_ts yields a np.nan prediction."""

    # given
    model = Model(n_estimators=10)
    Xy = pd.DataFrame(
        {"feature1": [], "feature2": [], "24h_later_load": []},
        index=pd.DatetimeIndex([]),
    )
    query_ts = pd.Timestamp("2024-10-01 00:00", tz="Europe/Zurich")

    # when-then
    assert np.isnan(model._train_predict(Xy, query_ts))


def test__train_predict__expected_prediction():
    """Check whether a model's prediction is of the expected shape."""

    # given
    model = Model(n_estimators=10)
    Xy = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 1],
            "feature2": [1, 2, 3, 1],
            "feature3": [1, 2, 3, 1],
            "24h_later_load": [1, 2, 3, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 01:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 02:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 03:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 04:00", tz="Europe/Zurich"),
            ]
        ),
    )
    query_ts = pd.Timestamp("2024-01-01 04:00", tz="Europe/Zurich")

    # when
    yhat = model._train_predict(Xy, query_ts)

    # then
    assert yhat == 2.0  # as the model overfit the training set


def test_train_predict__expected_prediction():
    """Check whether a model's predictions are of the expected shape."""

    # given
    model = Model(n_estimators=10)
    Xy = pd.DataFrame(
        {
            "feature1": [1, 2, 3, 4, 1, 2],
            "feature2": [1, 2, 3, 4, 1, 2],
            "feature3": [1, 2, 3, 4, 1, 2],
            "24h_later_load": [1, 2, 3, 4, np.nan, np.nan],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 01:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 02:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 03:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 04:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 05:00", tz="Europe/Zurich"),
                pd.Timestamp("2024-01-01 06:00", tz="Europe/Zurich"),
            ]
        ),
    )
    query_timestamps = [
        pd.Timestamp("2024-01-01 04:00", tz="Europe/Zurich"),
        pd.Timestamp("2024-01-01 05:00", tz="Europe/Zurich"),
        pd.Timestamp("2024-01-01 06:00", tz="Europe/Zurich"),
    ]
    # when
    yhat = model.train_predict(Xy, query_timestamps)

    # then
    assert isinstance(yhat, pd.DataFrame)
    assert len(yhat.columns) == 1
    assert "predicted_24h_later_load" in yhat.columns
    assert yhat["predicted_24h_later_load"].dtype == "float64"
    assert len(yhat) == len(query_timestamps)
