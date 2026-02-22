from datetime import datetime

from loguru import logger
import pandas as pd
from pydantic import BaseModel, model_validator

class ENTSOELoads(BaseModel):
    timestamps: list[datetime]
    day_later_loads: list[float | None]

    @model_validator(mode='after')
    def check_same_amount_of_timestamps_and_predictions(self) -> 'ENTSOELoads':
        if len(self.day_later_loads) != len(self.timestamps):
            error_str = f"timestamps and day_later_loads should have the same amount of values, but #timestamps: {len(self.timestamps)} and #day_later_loads: {len(self.day_later_loads)}"
            logger.error(error_str)
            raise ValueError(error_str)
        return self
