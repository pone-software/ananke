"""Contains all the classes for representing a detector."""
from dataclasses import dataclass, field
from typing import List

from ananke.data.configurations import DetectorConfiguration
from ananke.data.geometry import LocatedObject, OrientedLocatedObject, Vector3D
from ananke.data.interfaces import INumpyRepresentable


@dataclass
class PMT(OrientedLocatedObject):
    """Python class representing individual PMT."""

    #: Index of the current PMT
    ID: int


@dataclass
class Module(LocatedObject):
    """Python class representing individual module."""

    #: Index of the current module
    ID: int

    #: Number of modules per string
    builder_configuration: DetectorConfiguration

    @property
    def PMTs(self) -> List[PMT]:
        """List of PMTs based on the parameters."""
        return []


@dataclass
class String(LocatedObject):
    """Python class representing individual string."""

    #: Index of the current string
    ID: int

    #: Number of modules per string
    builder_configuration: DetectorConfiguration

    @property
    def modules(self) -> List[Module]:
        """List of modules based on the parameters."""
        return []


@dataclass
class Detector(INumpyRepresentable):
    """Python class representing detector."""

    #: configuration from the builder
    builder_configuration: DetectorConfiguration

    #: positions for the strings
    string_positions: List[Vector3D] = field(default_factory=lambda: [])

    @property
    def strings(self) -> List[String]:
        """All the detector strings based on string positions."""
        return []
