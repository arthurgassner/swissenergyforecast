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
    def _raise_if_unexpected_format(y: pd.Series) -> None:
        # Ensure y is actually a Series
        if type(y) is not pd.Series:
            error_str = f"y is not a pd.Series; it is: {type(y)}"
            logger.error(error_str)
            raise ValueError(error_str)            
        
        # Ensure index is unique
        if not y.index.is_unique:
            value_counts_df = y.index.value_counts()
            duplicate_mask = value_counts_df > 1
            error_str = f"y.index is not unique; {duplicate_mask.sum()}/{len(value_counts_df)} of its unique elements are duplicated: {value_counts_df[duplicate_mask]}"
            logger.error(error_str)
            raise ValueError(error_str)
        
        # Ensure index is DatetimeIndex
        if type(y.index) is not pd.DatetimeIndex:
            error_str = f"y.index is not a pd.DatetimeIndex; it is: {type(y.index)}"
            logger.error(error_str)
            raise ValueError(error_str)   

    
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
        MAPE._raise_if_unexpected_format(y)
        MAPE._raise_if_unexpected_format(yhat)
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
    mapes: list[MAPE] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_same_amount_of_timestamps_and_predictions(self) -> 'Forecast':
        if len(self.day_later_predicted_load) != len(self.timestamps):
            error_str = f"timestamps and day_later_predicted_load should have the same amount of values, but #timestamps: {len(self.timestamps)} and #day_later_predicted_load: {len(self.day_later_predicted_load)}"
            logger.error(error_str)
            raise ValueError(error_str)
        return self

    def __format__(self, format_spec) -> str:
        formatted_str = "MAPEs:\n"
        for e in self.mapes:
            formatted_str += f"- {e}\n"

        return formatted_str