"""This module contains all event and photon source related structures."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

import numpy.typing as npt
import pandas as pd
from pandera.typing import DataFrame

from ananke.models.detector import Detector
from ananke.models.geometry import OrientedLocatedObjects
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import RecordSchema, SourceRecordSchema, EventRecordSchema, HitSchema


class Records(OrientedLocatedObjects):
    """General description of a record for events or sources."""
    df: DataFrame[RecordSchema]


class SourceRecords(Records):
    """Record for a photon source."""
    df: DataFrame[SourceRecordSchema]

    # TODO: Fix THis
    #angle_distribution: Optional[npt.ArrayLike] = None

    @property
    def number_of_photons(self) -> pd.DataFrame:
        """Gets DataFrame with all numbers of photons."""
        return self.df[[
            'number_of_photons'
        ]]

    @property
    def times(self) -> pd.DataFrame:
        """Gets DataFrame with all times."""
        return self.df[[
            'time'
        ]]


class EventRecords(Records):
    """Record of an event that happened."""
    df: DataFrame[EventRecordSchema]


class Hits(DataFrameFacade):
    """Record of an event that happened."""
    df: DataFrame[HitSchema]


@dataclass
class Events:
    detector: Detector
    sources: SourceRecords
    events: EventRecords
    hits: Hits

    @classmethod
    def concat(cls, events_to_concat: List[Events]) -> Events:
        if len(events_to_concat) == 0:
            raise ValueError('You have to pass at least one Events object in list')
        events = Events(
            detector=events_to_concat[0].detector,
            sources=SourceRecords.concat([x.sources for x in events_to_concat]),
            events=EventRecords.concat([x.events for x in events_to_concat]),
            hits=Hits.concat([x.hits for x in events_to_concat]),
        )
        return events
