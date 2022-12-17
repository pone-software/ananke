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
    event_id: Series[int] = pa.Field(coerce=True)


class RecordSchema(EventIdSchema):
    time: Series[float] = pa.Field(coerce=True)


class HitSchema(RecordSchema):
    string_id: Series[int] = pa.Field(coerce=True)
    module_id: Series[int] = pa.Field(coerce=True)
    pmt_id: Series[int] = pa.Field(coerce=True)

class OrientedRecordSchema(OrientedLocatedObjectSchema, RecordSchema):
    pass


class SourceRecordSchema(OrientedRecordSchema):
    number_of_photons: Series[int] = pa.Field(coerce=True)
    type: Series[int] = pa.Field(isin=[x.value for x in SourceType])


class EventRecordSchema(OrientedRecordSchema):
    energy: Series[float] = pa.Field(coerce=True)
    type: Series[int] = pa.Field(isin=[x.value for x in EventType])
    particle_id: Series[int] = pa.Field(coerce=True)
    length: Optional[Series[float]] = pa.Field(coerce=True)
