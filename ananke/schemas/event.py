"""Module containing all schemas for Collection"""
from enum import Enum
from typing import Optional

import pandera as pa
from pandera.typing import Series

from ananke.schemas.geometry import OrientedLocatedObjectSchema


class EventType(Enum):
    """Enum for different event types."""

    STARTING_TRACK = 'starting_track'
    CASCADE = 'cascade'
    REALISTIC_TRACK = 'realistic_track'


class SourceType(Enum):
    """Enum for photon source types."""

    STANDARD_CHERENKOV = 'cherenkov'
    ISOTROPIC = 'isotropic'


class RecordIdSchema(pa.SchemaModel):
    record_id: Series[int] = pa.Field(coerce=True)


class TimedSchema(pa.SchemaModel):
    time: Series[float] = pa.Field(coerce=True)


class RecordSchema(RecordIdSchema, TimedSchema):
    type: Series[str]


class HitSchema(TimedSchema, RecordIdSchema):
    string_id: Series[int] = pa.Field(coerce=True)
    module_id: Series[int] = pa.Field(coerce=True)
    pmt_id: Series[int] = pa.Field(coerce=True)


class OrientedRecordSchema(OrientedLocatedObjectSchema, RecordSchema):
    pass


class SourceRecordSchema(OrientedRecordSchema):
    number_of_photons: Series[int] = pa.Field(coerce=True)
    type: Series[str] = pa.Field(isin=[x.value for x in SourceType])


class EventRecordSchema(OrientedRecordSchema):
    energy: Series[float] = pa.Field(coerce=True)
    type: Series[str] = pa.Field(isin=[x.value for x in EventType])
    particle_id: Series[int] = pa.Field(coerce=True)
    length: Optional[Series[float]] = pa.Field(coerce=True)
