"""This module contains all event and photon source related structures."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import numpy as np
from pydantic import BaseModel, NonNegativeInt

from ananke.models.geometry import OrientedLocatedObjects
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import (
    EventRecordSchema,
    HitSchema,
    NoiseRecordSchema,
    OrientedRecordSchema,
    RecordIdSchema,
    RecordSchema,
    SourceRecordSchema,
    TimedSchema, RecordStatisticsSchema,
)
from ananke.utils import percentile as percentile_func
import numpy.typing as npt
from pandera.typing import DataFrame


class RecordIds(DataFrameFacade):
    """General description of a record ids."""

    df: DataFrame[RecordIdSchema]

    def get_by_record(self, record_id: int) -> RecordIds:
        """Gets all sources by a record id.

        Args:
            record_id: ID of the record to get

        Returns:
            Sources of the record
        """
        return self.__class__(df=self.df[self.df["record_id"] == record_id])

    @property
    def record_ids(self) -> pd.Series:
        """Gets all the record ids of the current df."""
        return self.df["record_id"]


class TimeStatistics(BaseModel):
    count: NonNegativeInt
    min: float
    max: float


class RecordTimes(DataFrameFacade):
    """General description of intervals."""

    df: DataFrame[TimedSchema]

    @property
    def times(self) -> pd.Series:
        """Gets DataFrame with all times."""
        return self.df["time"]

    def add_time(self, time_difference: npt.ArrayLike) -> None:
        """Adds time to the data frame.

        Args:
            time_difference: time to add
        """
        self.df["time"] = self.df["time"] + time_difference

    def get_statistics(self, percentile: Optional[float] = None) -> TimeStatistics:
        """Returns the Statistics of the current hits.

        Args:
            percentile: Float between 0 and one to give percentile of included rows.

        Returns:
            TimeStatistics Object containing min, max, and count
        """
        count = len(self)
        if percentile is not None:
            if percentile < 0 or percentile > 1:
                raise ValueError('Percentiles can only be between 0 and 1.')
            beginning_percentile = 0.5 - percentile / 2.0
            ending_percentile = 0.5 + percentile / 2.0
            aggregations = [
                percentile_func(beginning_percentile, "min"),
                percentile_func(ending_percentile, "max"),
            ]
            count = int(np.round(count * percentile))
        else:
            aggregations = ["min", "max"]

        grouped_hits = self.df.agg({"time": aggregations})

        return TimeStatistics(
            count=count,
            min=grouped_hits.at['min', 'time'],
            max=grouped_hits.at['max', 'time']
        )


class Records(RecordIds, RecordTimes):
    """General description of a record for events or sources."""

    df: DataFrame[RecordSchema]


class RecordStatistics(Records):
    """General description of a record for events or sources."""

    df: DataFrame[RecordStatisticsSchema]


class OrientedRecords(OrientedLocatedObjects, Records):
    """General description of a record for events or sources."""

    df: DataFrame[OrientedRecordSchema]


class Sources(OrientedRecords):
    """Record for a photon source."""

    df: DataFrame[SourceRecordSchema]

    # TODO: Fix THis
    # angle_distribution: Optional[npt.ArrayLike] = None

    @property
    def number_of_photons(self) -> pd.Series:
        """Gets DataFrame with all numbers of photons."""
        return self.df["number_of_photons"]


class EventRecords(OrientedRecords):
    """Record of an event that happened."""

    df: DataFrame[EventRecordSchema]


class NoiseRecords(Records):
    """Record of an event that happened."""

    df: DataFrame[NoiseRecordSchema]


class Hits(Records):
    """Record of an event that happened."""

    df: DataFrame[HitSchema]
