"""Module containing all the configuration classes."""
from enum import Enum
from typing import Union, Optional, Literal

from pydantic import BaseModel, Field, PositiveInt, PositiveFloat, NonNegativeFloat, confloat

class DetectorGeometries(str, Enum):
    """Possible detector geometries."""
    SINGLE = 'single'
    TRIANGULAR = 'triangular'
    GRID = 'grid'
    HEXAGONAL = 'hexagonal'
    RHOMBUS = 'rhombus'


class Position(BaseModel):
    """Configuration for a position."""

    #: X-position of the position.
    x: float

    #: Y-position of the position.
    y: float


class GeometryConfiguration(BaseModel):
    """Configuration for a detector geometry."""

    #: Start position of the current geometry
    start_position: Position = Position(x=0.0, y=0.0)

    type: Literal[DetectorGeometries.SINGLE]

class SingleGeometryConfiguration(GeometryConfiguration):

    #: Type of the expected detector geometry
    type: Literal[DetectorGeometries.SINGLE]


class LengthGeometryConfiguration(GeometryConfiguration):
    """Configuration for a length based geometry.

    Examples for this could be squares or triangles.
    """

    #: Start position of the current geometry

    #: types for LengthGeometryConfiguration
    type: Literal[DetectorGeometries.TRIANGULAR, DetectorGeometries.RHOMBUS]

    #: side length of the detector
    side_length: PositiveInt


class SidedGeometryConfiguration(GeometryConfiguration):
    """Configuration for geometries with multiple sides.

    Examples are grids and hexagonal architectures.
    """

    #: types for SidedGeometryConfiguration
    type : Literal[DetectorGeometries.GRID, DetectorGeometries.HEXAGONAL]

    #: Amounts for the number of modules per side
    number_of_strings_per_side: PositiveInt

    #: Distance between the individual strings
    distance_between_strings: NonNegativeFloat


class PMTConfiguration(BaseModel):
    """Configuration for a single PMT."""

    #: Efficiency of the PMT
    efficiency: confloat(ge=0, le=1) = 0.5

    #: Base Noise rate for the detector
    noise_rate: NonNegativeFloat = 16e-5

    #: Scale of gamma distribution for noise rate. If `0.0`, the fixed value is taken
    gamma_scale: NonNegativeFloat = 0.0


class ModuleConfiguration(BaseModel):
    """Configuration for a single Module"""

    #: radius of a given module
    radius: PositiveFloat


class StringConfiguration(BaseModel):
    """Configuration for a single string"""

    #: The distance of the first module to the flow
    z_offset: float = 0.0

    #: Number of modules on the string
    module_number: PositiveInt

    #: Distance between individual modules
    module_distance: PositiveFloat


class DetectorConfiguration(BaseModel):
    """Configuration for the detector builder to build detector."""

    #: Settings regarding the shape of the final detector
    geometry: Union[LengthGeometryConfiguration, SidedGeometryConfiguration, GeometryConfiguration] \
        = Field(..., discriminator="type")

    #: Configuration for the individual string
    string: StringConfiguration

    #: Configuration for the individual module
    module: ModuleConfiguration

    #: Configuration for the individual PMT
    pmt: PMTConfiguration

    #: Seed for the random generators used within detector
    seed: int = 1337