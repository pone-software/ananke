"""This module contains all event and photon source related structures."""
from __future__ import annotations

import pandas as pd

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
    TimedSchema,
)
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


class RecordTimes(DataFrameFacade):
    """General description of intervals."""

    df: DataFrame[TimedSchema]

    @property
    def times(self) -> pd.Series:
        """Gets DataFrame with all times."""
        return self.df["time"]


class Records(RecordIds, RecordTimes):
    """General description of a record for events or sources."""

    df: DataFrame[RecordSchema]


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
