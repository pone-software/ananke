"""Module containing all geometrical dataframe schemas."""

import pandera as pa

from pandera.typing import Series


class PolarSchema(pa.SchemaModel):
    """Schema for the polar coordinates data frame."""

    norm: Series[float] = pa.Field(coerce=True, nullable=True)
    phi: Series[float] = pa.Field(coerce=True, nullable=True)


class SphericalSchema(PolarSchema):
    """Schema for the spherical coordinates data frame."""

    theta: Series[float] = pa.Field(coerce=True, nullable=True)


class Vector2DSchema(pa.SchemaModel):
    """Schema for a 2D vectors data frame."""

    x: Series[float] = pa.Field(coerce=True, nullable=True)
    y: Series[float] = pa.Field(coerce=True, nullable=True)


class Vector3DSchema(Vector2DSchema):
    """Schema for a 3D vectors data frame."""

    z: Series[float] = pa.Field(coerce=True, nullable=True)


class LocatedObjectSchema(pa.SchemaModel):
    """Schema for a located objects data frame."""

    location_x: Series[float] = pa.Field(coerce=True, nullable=True)
    location_y: Series[float] = pa.Field(coerce=True, nullable=True)
    location_z: Series[float] = pa.Field(coerce=True, nullable=True)


class OrientedLocatedObjectSchema(LocatedObjectSchema):
    """Schema for a located objects that are oriented data frame."""

    orientation_x: Series[float] = pa.Field(coerce=True, nullable=True)
    orientation_y: Series[float] = pa.Field(coerce=True, nullable=True)
    orientation_z: Series[float] = pa.Field(coerce=True, nullable=True)
