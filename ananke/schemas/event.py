"""Module containing all schemas for Events"""
from enum import Enum
from typing import Optional

import pandera as pa
from pandera.typing import Series

from ananke.schemas.geometry import OrientedLocatedObjectSchema


class EventType(Enum):
    """Enum for different event types."""

    STARTING_TRACK = 1
    CASCADE = 2
    REALISTIC_TRACK = 3


class SourceType(Enum):
    """Enum for photon source types."""

    STANDARD_CHERENKOV = 1
    ISOTROPIC = 2


class HitType(Enum):
    STARTING_TRACK = EventType.STARTING_TRACK
    CASCADE = EventType.CASCADE
    REALISTIC_TRACK = EventType.REALISTIC_TRACK
    NOISE = 4


class EventIdSchema(pa.SchemaModel):
    event_id: Series[int]


class HitSchema(EventIdSchema):
    string_id: Series[int]
    module_id: Series[int]
    pmt_id: Series[int]
    time: Series[float]
    type: Series[HitType]


class RecordSchema(OrientedLocatedObjectSchema, EventIdSchema):
    time: Series[float]
    type: Series[Enum]


class SourceRecordSchema(RecordSchema):
    number_of_photons: Series[int]
    type: Series[SourceType]


class EventRecordSchema(RecordSchema):
    energy: Series[float]
    type: Series[EventType]
    length: Optional[Series[float]]
