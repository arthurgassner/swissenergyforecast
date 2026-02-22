from datetime import datetime, timezone
from typing import Any, Type

from human_readable import precise_delta
from loguru import logger
import pandas as pd
from pydantic import BaseModel, Field, computed_field, model_validator
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

class MAPE(BaseModel):
    score: float | None = Field(default=None)
    timestamps: list[datetime] = Field(default_factory=list)
    y_true: list[float] = Field(default_factory=list)
    y_pred: list[float] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_same_amount_of_timestamps_and_predictions(self) -> 'MAPE':
        if len(self.y_pred) != len(self.timestamps):
            error_str = f"timestamps, y_true and y_pred should have the same amount of values, but #timestamps: {len(self.timestamps)}, #y_true: {len(self.y_true)} and #y_pred: {len(self.y_pred)}"
            logger.error(error_str)
            raise ValueError(error_str)
        return self
    
    def __format__(self, format_spec) -> str:
        return f"MAPE [{self.n_samples} timestamps over {self.span_str}]: {self.score:.2f}%"

    @computed_field
    @property
    def span_str(self) -> str:
        human_delta_str = precise_delta(self.end_ts - self.start_ts, minimum_unit="hours")
        return human_delta_str
    
    @computed_field
    @property
    def n_samples(self) -> int:
        return len(self.y_true)
    
    @computed_field
    @property
    def start_ts(self) -> datetime | None:
        if not self.timestamps:
            return None
        return min(self.timestamps)
    
    @computed_field
    @property
    def end_ts(self) -> datetime | None:
        if not self.timestamps:
            return None
        return max(self.timestamps)
    
    @staticmethod
    def compute_mapes(y: pd.Series, yhat: pd.Series, timedelta_strs: list[str]) -> list[MAPE]:
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
            list[MAPE]: Computed MAPEs
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

        mapes = []
        for timedelta_str in sorted(timedelta_strs):
            curr_starting_ts = last_load_ts - pd.to_timedelta(timedelta_str)
            curr_data_mask = y.index >= curr_starting_ts
            if not len(curr_data_mask):
                mapes.append(MAPE())
                continue

            y_true, y_pred = y[curr_data_mask], yhat[curr_data_mask]
            curr_score = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred) * 100
            mapes.append(MAPE(score=curr_score, timestamps=y.index[curr_data_mask], y_true=y_true, y_pred=y_pred))

        return mapes 

class Forecast(BaseModel):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    timestamps: list[datetime] = Field(default_factory=list)
    day_later_predicted_load: list[float] = Field(default_factory=list)
    entsoe_mapes: list[MAPE] = Field(default_factory=list)
    our_mapes: list[MAPE] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_same_amount_of_timestamps_and_predictions(self) -> 'Forecast':
        if len(self.y_pred) != len(self.timestamps):
            error_str = f"timestamps and y_pred should have the same amount of values, but #timestamps: {len(self.timestamps)} and #y_pred: {len(self.y_pred)}"
            logger.error(error_str)
            raise ValueError(error_str)
        return self

    def __format__(self, format_spec) -> str:
        formatted_str = "ENSTO-E MAPEs:\n"
        for e in self.entsoe_mapes:
            formatted_str += f"- {e}\n"

        formatted_str += "Our MAPEs:\n"
        for e in self.our_mapes:
            formatted_str += f"- {e}\n"

        return formatted_str