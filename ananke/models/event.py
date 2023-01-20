"""This module contains all event and photon source related structures."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Callable

import numpy.typing as npt
import pandas as pd
from pandera.typing import DataFrame

from ananke.models.detector import Detector
from ananke.models.geometry import OrientedLocatedObjects
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import (
    RecordSchema,
    SourceRecordSchema,
    EventRecordSchema,
    HitSchema,
    OrientedRecordSchema,
    RecordIdSchema,
    TimedSchema,
    NoiseRecordSchema,
)


class RecordIds(DataFrameFacade):
    """General description of a record ids."""
    df: DataFrame[RecordIdSchema] = RecordIdSchema.example(size=0)

    def get_by_record(self, record_id: int) -> RecordIds:
        """Gets all sources by a record id.

        Args:
            record_id: ID of the record to get

        Returns:
            Sources of the record
        """
        return self.__class__(df=self.df[self.df['record_id'] == record_id])


class RecordTimes(DataFrameFacade):
    """General description of intervals."""
    df: DataFrame[TimedSchema] = TimedSchema.example(size=0)

    @property
    def times(self) -> pd.Series:
        """Gets DataFrame with all times."""
        return self.df['time']


class Records(RecordIds, RecordTimes):
    """General description of a record for events or sources."""
    df: DataFrame[RecordSchema] = RecordSchema.example(size=0)


class OrientedRecords(OrientedLocatedObjects, Records):
    """General description of a record for events or sources."""
    df: DataFrame[OrientedRecordSchema] = OrientedRecordSchema.example(size=0)


class Sources(OrientedRecords):
    """Record for a photon source."""
    df: DataFrame[SourceRecordSchema] = SourceRecordSchema.example(size=0)

    # TODO: Fix THis
    # angle_distribution: Optional[npt.ArrayLike] = None

    @property
    def number_of_photons(self) -> pd.Series:
        """Gets DataFrame with all numbers of photons."""
        return self.df['number_of_photons']


class EventRecords(OrientedRecords):
    """Record of an event that happened."""
    df: DataFrame[EventRecordSchema] = EventRecordSchema.example(size=0)


class NoiseRecords(Records):
    """Record of an event that happened."""
    df: DataFrame[NoiseRecordSchema] = NoiseRecordSchema.example(size=0)


class Hits(RecordTimes, RecordIds):
    """Record of an event that happened."""
    df: DataFrame[HitSchema] = HitSchema.example(size=0)


@dataclass
class Collection:
    detector: Detector
    records: Records
    hits: Hits
    sources: Optional[Sources] = None

    @classmethod
    def concat(cls, collections_to_concat: List[Collection]) -> Collection:
        if len(collections_to_concat) == 0:
            raise ValueError('You have to pass at least one Collection object in list')
        sources_list = []
        records_list = []
        hits_list = []
        for collection_to_concat in collections_to_concat:
            if collection_to_concat.sources is not None:
                sources_list.append(collection_to_concat.sources)
            records_list.append(collection_to_concat.records)
            hits_list.append(collection_to_concat.hits)

        if len(sources_list) == 0:
            sources_list = None
        collection = Collection(
            detector=collections_to_concat[0].detector,
            sources=Sources.concat(sources_list),
            records=Records.concat(records_list),
            hits=Hits.concat(hits_list),
        )
        return collection
