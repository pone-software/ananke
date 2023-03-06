"""Module containing schemas for the exporters."""

import pandera as pa

from ananke.schemas.event import RecordIdSchema, RecordSchema, TimedSchema
from pandera.typing import Series


class GraphNetDetectorSchema(RecordIdSchema, TimedSchema):
    """Schema with all detector features of GraphNet P-One."""

    string_id: Series[int] = pa.Field(coerce=True)
    module_id: Series[int] = pa.Field(coerce=True)
    pmt_id: Series[int] = pa.Field(coerce=True)
    pmt_x: Series[float] = pa.Field(coerce=True)
    pmt_y: Series[float] = pa.Field(coerce=True)
    pmt_z: Series[float] = pa.Field(coerce=True)
    pmt_azimuth: Series[float] = pa.Field(coerce=True)
    pmt_zenith: Series[float] = pa.Field(coerce=True)
    time: Series[float] = pa.Field(coerce=True)


class GraphNetTruthSchema(RecordSchema):
    """Schema with all truth features of GraphNet P-One."""

    interaction_x: Series[float] = pa.Field(coerce=True)
    interaction_y: Series[float] = pa.Field(coerce=True)
    interaction_z: Series[float] = pa.Field(coerce=True)
    interaction_azimuth: Series[float] = pa.Field(coerce=True)
    interaction_zenith: Series[float] = pa.Field(coerce=True)
    energy: Series[float] = pa.Field(coerce=True)
    particle_id: Series[int] = pa.Field(coerce=True)
