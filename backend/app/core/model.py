from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


class Model:
    """Class responsible for handling the training, inference and testing of a time-series prediction model."""

    def __init__(self, n_estimators: int) -> None:
        """Create a Model object, which encapsulates an LGBMRegressor with `n_estimators` estimators

        Args:
            n_estimators (int): Amount of estimators of the LGBMRegressor
        """
        # Create untrained-model
        self._model = lgb.LGBMRegressor(n_estimators=n_estimators, force_row_wise=True, verbose=-1)

    def _train_predict(self, Xy: pd.DataFrame, query_ts: pd.Timestamp) -> float:
        """Train a model on all the features whose index is BEFORE query_ts,
        and run an inference on the features EXACTLY AT query_ts.

        Args:
            Xy (pd.DataFrame): Dataframe containing the (features, target), where the target is '24h_later_load'
            query_ts (pd.Timestamp): Timestamp whose inference we are interested in

        Returns:
            float: Predicted value for '24h_later_load',
                   np.nan if the features were missing from Xy
        """

        assert isinstance(Xy.index, pd.DatetimeIndex)
        assert Xy.index.is_unique
        assert Xy.index.is_monotonic_increasing

        # Extract the serving Xy
        if not query_ts in Xy.index:
            logger.warning(f"Query timestamp {query_ts} is missing from Xy's index. Cannot run prediction.")
            return np.nan

        X_serving = Xy.loc[[query_ts]].drop(columns=["24h_later_load"])

        # Prepare training data
        Xy = Xy.dropna(subset=("24h_later_load"))
        Xy = Xy[Xy.index < query_ts]  # Only train on data strictly before the ts
        X, y = Xy.drop(columns=["24h_later_load"]), Xy["24h_later_load"]

        # Train the model
        self._model.fit(X, y)

        # Predict
        return float(self._model.predict(X_serving)[0])

    def train_predict(self, Xy: pd.DataFrame, query_timestamps: list[pd.Timestamp]) -> pd.Series:
        """Train one model per query_ts in `query_timestamps`.
        Each model will only be training on the features in Xy available strictly BEFORE said query_ts.
        The features EXACTLY AT the query_ts will be used to predict the `24h_later_load`.

        Args:
            Xy (pd.DataFrame): Dataframe containing the (features, target), where the target is '24h_later_load'
            query_timestamps (list[pd.Timestamp]): Timestamps whose inference we are interested in

        Returns:
            pd.Series: Dataframe with the predicted values under the column 'predicted_24h_later_load'.
                       The index corresponds to the query_timestamps.
        """

        predicted_values = []
        for query_ts in tqdm(query_timestamps):
            predicted_values.append(self._train_predict(Xy, query_ts))

        return pd.DataFrame({"predicted_24h_later_load": predicted_values}, index=pd.DatetimeIndex(query_timestamps))