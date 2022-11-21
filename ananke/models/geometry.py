"""This module contains all geometric dataclasses."""
from dataclasses import dataclass
from typing import Any

import numpy as np

from ananke.models.interfaces import NumpyRepresentable
from numpy import typing as npt


@dataclass
class Vector2D(NumpyRepresentable):
    """A 2D vector with interface to radial and cartesian coordinates."""

    #: X-component
    x: float

    #: Y-component
    y: float

    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        return np.array([self.x, self.y], dtype=dtype)

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

    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        return np.append(super()._get_numpy_array(dtype=dtype), self.z)

    @property
    def theta(self) -> float:
        """Phi coordinate in radial units."""
        return float(np.arccos(self.z / self.norm))


@dataclass
class LocatedObject(NumpyRepresentable):
    """Object that has a location."""

    #: Location of the object
    location: Vector3D

    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        return np.array(self.location)


@dataclass
class OrientedLocatedObject(LocatedObject):
    """Object that has a location and orientation."""

    #: Orientation of the object
    orientation: Vector3D

    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        location_array = super()._get_numpy_array()
        return np.append(location_array, [self.orientation.phi, self.orientation.theta])
