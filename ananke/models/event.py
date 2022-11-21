"""This module contains all event and photon source related structures."""
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List

from ananke.data.detector import Detector
from ananke.models.geometry import OrientedLocatedObject


@dataclass
class Hit:
    """Description of a detector hit."""

    #: ID of the string
    string_id: int

    #: ID of the module
    module_id: int

    #: ID of the PMT
    pmt_id: int

    #: time of the hit
    time: float


@dataclass
class Record(OrientedLocatedObject):
    """General description of a record for events or sources."""

    #: Time of the record
    time: float

    #: Some Enum like Type Object
    type: Enum


class SourceType(Enum):
    """Enum for photon source types."""

    STANDARD_CHERENKOV = auto()
    ISOTROPIC = auto()


@dataclass
class SourceRecord(Record):
    """Record for a photon source."""

    #: Source record type
    type: SourceType


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


@dataclass
class Event:
    """Class combining detector response with monte carlo truth sources and record."""

    #: ID of the event
    ID: int

    #: Detector of the event
    detector: Detector

    #: Monte carlo truth of the event
    event_record: EventRecord

    #: Detector response
    hits: List[Hit] = field(default_factory=lambda: [])

    #: Photon sources leading to hits
    sources: List[SourceRecord] = field(default_factory=lambda: [])
