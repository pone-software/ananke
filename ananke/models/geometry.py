"""This module contains all geometric dataclasses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ananke.models.interfaces import ScientificConvertible


@dataclass
class Vector2D(ScientificConvertible):
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

    def __add__(self, other: Vector2D) -> Vector2D:
        """Adds two 2D vectors together.

        Args:
            other: Vector to add.

        Returns:
            new vector with added values.

        Raises:
            ValueError: Only objects of type Vector2D are allowed

        """
        if not isinstance(other, Vector2D):
            raise ValueError("Can only add Vector2D objects")

        return Vector2D(x=self.x + other.x, y=self.y + other.y)

    def to_pandas(self) -> pd.DataFrame:
        """Gets the dataframe of the Vector2D class.

        Generates a list of the dataframes of the vector.

        Returns:
            Dataframe containing vector information

        """
        return pd.DataFrame({
            'x': self.x,
            'y': self.y,
        }, index=[0])

    def _get_scaling_factor_for_length(self, length: float) -> float:
        """Calculates the factor to scale by to get to length.

        Args:
            length: Length to calculate factor for.

        Returns:
            Scaling factor based on vector norm.

        """
        norm = self.norm
        return length / norm

    def scale_to_length(self, length: float) -> None:
        """Scaling the vector to a given length.

        Args:
            length: length to scale the vector to.
        """
        factor = self._get_scaling_factor_for_length(length)
        self.x *= factor
        self.y *= factor

    @classmethod
    def from_polar(cls, norm: float, phi: float) -> Vector2D:
        """Creates a 2D vector from polar coordinates.

        Args:
            norm: norm of the vector
            phi: angle phi

        Returns:
            2D Vector with the given properties.
        """
        x = np.cos(phi) * norm
        y = np.sin(phi) * norm
        return cls(x=x, y=y)


@dataclass
class Vector3D(Vector2D):
    """A 3D vector with interface to radial and spherical coordinates."""

    #: Z-component
    z: float

    @property
    def theta(self) -> float:
        """Phi coordinate in radial units."""
        return float(np.arccos(self.z / self.norm))

    def __add__(self, other: Vector3D) -> Vector3D:  # type: ignore[override]
        """Adds two 3D vectors together.

        Args:
            other: Vector to add.

        Returns:
            new vector with added values.

        Raises:
            ValueError: Only objects of type Vector3D are allowed

        """
        if not isinstance(other, Vector3D):
            raise ValueError("Can only add Vector3D objects")

        return Vector3D(x=self.x + other.x, y=self.y + other.y, z=self.z + other.z)

    def to_pandas(self) -> pd.DataFrame:
        """Gets the dataframe of the Vector3D class.

        Generates a list of the dataframes of the vector.

        Returns:
            Dataframe containing vector information

        """
        dataframe = super().to_pandas()
        return dataframe.assign(z=self.z)

    def scale_to_length(self, length: float) -> None:
        """Scaling the vector to a given length.

        Args:
            length: length to scale the vector to.
        """
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
        """Create 3D vector based on spherical coordinates.

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
class LocatedObject(ScientificConvertible):
    """Object that has a location."""

    #: Location of the object
    location: Vector3D

    def to_pandas(self) -> pd.DataFrame:
        """Gets the dataframe of the oriented located object class.

        Generates a list of the dataframes of the strings.

        Returns:
            Dataframe containing pmt information

        """
        dataframe = self.location.to_pandas()
        return dataframe.rename(columns={
            'x': 'location_x',
            'y': 'location_y',
            'z': 'location_z',
        })


@dataclass
class OrientedLocatedObject(LocatedObject):
    """Object that has a location and orientation."""

    #: Orientation of the object
    orientation: Vector3D

    def to_pandas(self) -> pd.DataFrame:
        """Gets the dataframe of the oriented located object class.

        Generates a list of the dataframes of the strings.

        Returns:
            Dataframe containing pmt information

        """
        dataframe = super().to_pandas()
        return dataframe.assign(
            orientation_x=self.orientation.x,
            orientation_y=self.orientation.y,
            orientation_z=self.orientation.z,
        )