"""This module contains all event and photon source related structures."""
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import List, Optional

import pandas as pd

from ananke.models.detector import Detector
from ananke.models.geometry import OrientedLocatedObject
from ananke.models.interfaces import ScientificSequence, ScientificCollection


@dataclass
class Hit(ScientificSequence):
    """Description of a detector hit."""

    #: ID of the string
    string_id: int

    #: ID of the module
    module_id: int

    #: ID of the PMT
    pmt_id: int

    #: time of the hit
    time: float

    def to_pandas(self) -> pd.DataFrame:
        """Converts hit to Dataframe

        Returns:
            Dataframe with columns of attributes
        """
        hit_dict = asdict(self)
        return pd.DataFrame.from_dict(hit_dict)


@dataclass
class Record(OrientedLocatedObject):
    """General description of a record for events or sources."""

    #: Time of the record
    time: float

    def to_pandas(self) -> pd.DataFrame:
        """Converts Record to Dataframe

        Returns:
            Dataframe with columns of attributes
        """
        dataframe = super().to_pandas()

        dataframe = dataframe.assign(
            time=self.time,
        )

        # Hack as type cannot be defined here due to some order issues
        if hasattr(self, 'type'):
            dataframe = dataframe.assign(
                type=self.type
            )

        return dataframe


class SourceType(Enum):
    """Enum for photon source types."""

    STANDARD_CHERENKOV = auto()
    ISOTROPIC = auto()


@dataclass
class SourceRecord(Record):
    """Record for a photon source."""

    #: Total number of source photons
    number_of_photons: int

    #: Source record type
    type: SourceType = SourceType.STANDARD_CHERENKOV

    def to_pandas(self) -> pd.DataFrame:
        """Converts SourceRecord to Dataframe

        Returns:
            Dataframe with columns of attributes
        """
        dataframe = super().to_pandas()

        return dataframe.assign(
            number_of_photons=self.number_of_photons,
        )


class EventType(Enum):
    """Enum for different event types."""

    STARTING_TRACK = auto()
    CASCADE = auto()
    REALISTIC_TRACK = auto()


@dataclass
class EventRecord(Record):
    """Record of an event that happened."""

    #: Energy of the event
    energy: float

    #: Type of the event
    type: EventType

    #: Depending on the type an event can have a length
    length: Optional[float] = None

    #: Fennel particle id of the event
    particle_id: Optional[int] = None  # TODO: needed here?

    def to_pandas(self) -> pd.DataFrame:
        """Converts SourceRecord to Dataframe

        Returns:
            Dataframe with columns of attributes
        """
        dataframe = super().to_pandas()

        return dataframe.assign(
            energy=self.energy,
            length=self.length,
            particle_id=self.particle_id,
        )

class HitCollection(ScientificCollection[Hit]):
    """Collection containing Hits."""
    pass
class SourceRecordCollection(ScientificCollection[SourceRecord]):
    """Collection containing source records."""
    pass


@dataclass
class Event(ScientificSequence):
    """Class combining detector response with monte carlo truth sources and record."""

    #: ID of the event
    ID: int

    #: Detector of the event
    detector: Detector

    #: Monte carlo truth of the event
    event_record: EventRecord

    #: Detector response
    hits: HitCollection = field(
        default_factory=lambda: HitCollection()
        )

    #: Photon sources leading to hits
    sources: SourceRecordCollection = field(
        default_factory=lambda: SourceRecordCollection()
        )

    def to_pandas(self) -> pd.DataFrame:
        """Converts SourceRecord to Dataframe

        Returns:
            Dataframe with columns of attributes
        """
        dataframe = self.event_record.to_pandas()

        return dataframe.assign(
            ID=self.ID,
        )
class EventCollection(ScientificCollection[Event]):
    """Collection containing events."""
    pass
