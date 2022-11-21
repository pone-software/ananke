"""This module contains all geometric dataclasses."""
from dataclasses import dataclass
from typing import Any, List

import numpy as np

from ananke.data.interfaces import INumpyRepresentable, NumpyRepresentable
from numpy import typing as npt


@dataclass
class Vector2D(NumpyRepresentable):
    """A 2D vector with interface to radial and cartesian coordinates."""

    #: X-component
    x: float

    #: Y-component
    y: float

    @property
    def phi(self) -> float:
        """Phi coordinate in radial units."""
        return float(np.arctan(self.y / self.x))

    @property
    def norm(self) -> float:
        """Returns L2-norm of 2D vector."""
        return float(np.linalg.norm(self))


@dataclass
class Vector3D(Vector2D):
    """A 3D vector with interface to radial and spherical coordinates."""

    #: Z-component
    z: float

    @property
    def theta(self) -> float:
        """Phi coordinate in radial units."""
        return float(np.arccos(self.z / self.norm))


@dataclass
class LocatedObject(INumpyRepresentable):
    """Object that has a location."""

    #: Location of the object
    location: Vector3D

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[np.float_]:
        """Numpy array representation of the location.

        Returns:
            Numpy array of location
        """
        return np.array(self.location, dtype=dtype)

    def __getitem__(self, index: Any) -> Any:
        """Passes on getitem of the location.

        Args:
            index: what do you want to get

        Returns:
            selected values of the location

        """
        return self.location[index]

    def __len__(self) -> int:
        """Length of the location.

        Returns:
            Length of location
        """
        return len(self.location)


@dataclass
class OrientedLocatedObject(LocatedObject):
    """Object that has a location and orientation."""

    #: Orientation of the object
    orientation: Vector3D

    @property
    def oriented_location(self) -> List[float]:
        """Gets a list with length 5 containing location and orientation.

        Specifically the elements are ordered as follows: x, y, z, phi, theta
        """
        return [
            self.location.x,
            self.location.y,
            self.location.z,
            self.orientation.phi,
            self.orientation.theta,
        ]

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[np.float_]:
        """Numpy array representation as stated in `oriented_location` property.

        Returns:
            Numpy array of `oriented_location`
        """
        return np.array(self.oriented_location, dtype=dtype)

    def __getitem__(self, index: Any) -> Any:
        """Passes on getitem of the `oriented_location` property.

        Args:
            index: what do you want to get

        Returns:
            selected values of the `oriented_location`

        """
        return self.oriented_location[index]

    def __len__(self) -> int:
        """Length of the `oriented_location` property.

        Returns:
            Length of `oriented_location`
        """
        return len(self.oriented_location)
