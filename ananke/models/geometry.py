"""This module contains all geometric dataclasses."""
from __future__ import annotations

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

    def _get_scaling_factor_for_length(self, length: float) -> float:
        """Calculates the factor to scale by to get to length.

        Args:
            length: Length to calculate factor for

        Returns:
            Scaling factor based on vector norm

        """
        norm = self.norm
        return length / norm

    def scale_to_length(self, length: float) -> None:
        """Scaling the vector to a given length

        Args:
            length: length to scale the vector to
        """
        factor = self._get_scaling_factor_for_length(length)
        self.x *= factor
        self.y *= factor
        self._update_numpy_array()

    @classmethod
    def from_polar(cls, norm: float, phi: float) -> Vector2D:
        """Creates a 2D vector from polar coordinates

        Args:
            norm: norm of the vector
            phi: angle phi

        Returns:
            2D Vector with the given properties
        """
        x = np.cos(phi) * norm
        y = np.sin(phi) * norm
        return cls(x=x, y=y)


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

    def scale_to_length(self, length: float) -> None:
        factor = self._get_scaling_factor_for_length(length=length)
        self.z *= factor
        super().scale_to_length(length)

    @classmethod
    def from_polar(cls, norm: float, phi: float) -> Vector2D:
        """3D vector cannot be created from polar.

        Raises:
            AttributeError: A creation from polar coordinates is not possible
        """
        raise AttributeError("Vector3D cannot be created by polar coordinates.")

    @classmethod
    def from_spherical(cls, norm: float, phi: float, theta: float) -> Vector3D:
        """Create 3D vector based on spherical coordinates

        Args:
            norm: length of the vector
            phi: azimutal angle
            theta: elevation angle

        Raises:
            AttributeError: A creation from polar coordinates is not possible
        """
        x = norm * np.sin(theta) * np.cos(phi)
        y = norm * np.sin(theta) * np.sin(phi)
        z = norm * np.cos(theta)

        return cls(x=x, y=y, z=z)


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
        return np.append(location_array, [self.orientation.norm, self.orientation.phi, self.orientation.theta])
