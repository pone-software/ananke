"""Module containing all geometrical dataframe schemas"""

import pandera as pa
from pandera.typing import Series


class PolarSchema(pa.SchemaModel):
    norm: Series[float]
    phi: Series[float]


class SphericalSchema(PolarSchema):
    theta: Series[float]


class Vector2DSchema(pa.SchemaModel):
    x: Series[float] = pa.Field(coerce=True)
    y: Series[float] = pa.Field(coerce=True)


class Vector3DSchema(Vector2DSchema):
    z: Series[float] = pa.Field(coerce=True)


class LocatedObjectSchema(pa.SchemaModel):
    location_x: Series[float]
    location_y: Series[float]
    location_z: Series[float]


class OrientedLocatedObjectSchema(pa.SchemaModel):
    orientation_x: Series[float]
    orientation_y: Series[float]
    orientation_z: Series[float]