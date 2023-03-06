"""This module contains all event and photon source related structures."""
from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from ananke.models.geometry import OrientedLocatedObjects
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import (
    EventRecordSchema,
    FullRecordSchema,
    HitSchema,
    NoiseRecordSchema,
    OrientedRecordSchema,
    RecordIdSchema,
    RecordIdsTypes_,
    RecordSchema,
    RecordStatisticsSchema,
    SourceSchema,
    TimedSchema,
)
from ananke.utils import percentile as percentile_func
from pandera.typing import DataFrame
from pydantic import BaseModel, NonNegativeInt


# TODO: Add typed schema


class RecordIds(DataFrameFacade):
    """General description of a record ids."""

    df: DataFrame[RecordIdSchema]

    def get_by_record_ids(
        self, record_ids: RecordIdsTypes_, invert: bool = False
    ) -> Optional[RecordIds]:
        """Gets all sources by a record id.

        Args:
            record_ids: ID(s) of the record to get
            invert: Get except passed record ids

        Returns:
            Records with given ids
        """
        if isinstance(record_ids, int):
            record_ids = [record_ids]

        if invert:
            new_df = self.df[~self.df["record_id"].isin(record_ids)]
        else:
            new_df = self.df[self.df["record_id"].isin(record_ids)]

        new_df.reset_index(drop=True, inplace=True)

        if new_df.empty:
            return None

        return self.__class__(df=new_df)

    @property
    def record_ids(self) -> pd.Series:
        """Gets all the record ids of the current df."""
        return self.df["record_id"]


# TODO: Fix everything here
class TimeStatistics(BaseModel):
    """Model for staticstical information of record."""

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
        self.df["time"] = self.df["time"] + np.array(time_difference)

    def get_statistics(self, percentile: Optional[float] = None) -> TimeStatistics:
        """Returns the Statistics of the current hits.

        Args:
            percentile: Float between 0 and one to give percentile of included rows.

        Returns:
            TimeStatistics Object containing min, max, and count
        """
        # TODO: Refractor get statistics
        count = len(self)
        if percentile is not None:
            if percentile < 0 or percentile > 1:
                raise ValueError("Percentiles can only be between 0 and 1.")
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
            min=grouped_hits.at["min", "time"],
            max=grouped_hits.at["max", "time"],
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

    df: DataFrame[SourceSchema]

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


class FullRecords(Records):
    """General description of a record for events or sources."""

    df: DataFrame[FullRecordSchema]


class Hits(Records):
    """Record of an event that happened."""

    df: DataFrame[HitSchema]
