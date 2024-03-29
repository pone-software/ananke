"""Module containing all schemas for Collection."""
from enum import Enum
from typing import List, Optional, Union

import pandas as pd
import pandera as pa

from ananke.schemas.geometry import OrientedLocatedObjectSchema
from pandera.typing import Series


class RecordType(Enum):
    """Enum for different records types."""

    STARTING_TRACK = 0
    CASCADE = 1
    REALISTIC_TRACK = 2
    ELECTRICAL = 20
    BIOLUMINESCENCE = 21


class EventType(Enum):
    """Enum for different event types."""

    STARTING_TRACK = RecordType.STARTING_TRACK.value
    CASCADE = RecordType.CASCADE.value
    REALISTIC_TRACK = RecordType.REALISTIC_TRACK.value


class NoiseType(Enum):
    """Enum for different noise types."""

    ELECTRICAL = RecordType.ELECTRICAL.value
    BIOLUMINESCENCE = RecordType.BIOLUMINESCENCE.value


class SourceType(Enum):
    """Enum for photon source types."""

    CHERENKOV = 0
    ISOTROPIC = 1


Types = Union[EventType, NoiseType, SourceType, RecordType]
TypesTypes_ = Optional[Union[List[Types], Types]]


class RecordIdSchema(pa.SchemaModel):
    """Schema for Dataframes having record ids."""

    record_id: Series[int] = pa.Field(coerce=True)


RecordIdsTypes_ = Optional[Union[int, List[int], pd.Series]]


class TimedSchema(pa.SchemaModel):
    """Schema for Dataframes having some time constraint."""

    time: Series[float] = pa.Field(coerce=True)
    duration: Optional[Series[float]] = pa.Field(coerce=True, nullable=True)


# TODO: Switch string to Category


class RecordSchema(RecordIdSchema, TimedSchema):
    """Schema for a timed record with an ID and type."""

    type: Series[int] = pa.Field(coerce=True, isin=[x.value for x in RecordType])


class RecordStatisticsSchema(RecordSchema):
    """Schema for a timed record with an ID and type."""

    hit_count: Optional[Series[pa.Int]] = pa.Field(coerce=True, ge=0, nullable=True)
    first_hit: Optional[Series[pa.Float]] = pa.Field(coerce=True, nullable=True)
    last_hit: Optional[Series[pa.Float]] = pa.Field(coerce=True, nullable=True)
    source_count: Optional[Series[pa.Int]] = pa.Field(coerce=True, ge=0, nullable=True)
    first_source: Optional[Series[pa.Float]] = pa.Field(coerce=True, nullable=True)
    last_source: Optional[Series[pa.Float]] = pa.Field(coerce=True, nullable=True)


class HitSchema(RecordSchema):
    """Schema for individual hits."""

    type: Series[int] = pa.Field(coerce=True, isin=[x.value for x in RecordType])
    string_id: Series[int] = pa.Field(coerce=True)
    module_id: Series[int] = pa.Field(coerce=True)
    pmt_id: Series[int] = pa.Field(coerce=True)


class OrientedRecordSchema(OrientedLocatedObjectSchema, RecordSchema):
    """Schema for records that have a location and orientation."""

    pass


class SourceSchema(OrientedRecordSchema):
    """Schema for photon sources."""

    number_of_photons: Series[int] = pa.Field(coerce=True)
    type: Series[int] = pa.Field(coerce=True, isin=[x.value for x in SourceType])


class NoiseRecordSchema(RecordSchema):
    """Schema for noise records."""

    type: Series[int] = pa.Field(coerce=True, isin=[x.value for x in NoiseType])


class EventRecordSchema(OrientedRecordSchema):
    """Schema for event records."""

    energy: Series[float] = pa.Field(coerce=True, nullable=True)
    type: Series[int] = pa.Field(isin=[x.value for x in EventType])
    particle_id: Series[int] = pa.Field(coerce=True, nullable=True)
    length: Optional[Series[float]] = pa.Field(coerce=True, nullable=True)


class FullRecordSchema(NoiseRecordSchema, EventRecordSchema):
    """Schema with all possible event and noise columns."""

    type: Series[int] = pa.Field(isin=[x.value for x in RecordType])
