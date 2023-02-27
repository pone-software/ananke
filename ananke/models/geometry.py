"""This module contains all geometric dataclasses."""
from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.geometry import (
    LocatedObjectSchema,
    OrientedLocatedObjectSchema,
    PolarSchema,
    SphericalSchema,
    Vector2DSchema,
    Vector3DSchema,
)
from pandera import check_types
from pandera.typing import DataFrame

# TODO: Reimplement from_ functions type hints (deactivated due to vms problems)


class Vectors2D(DataFrameFacade):
    """A 2D vector with interface to radial and cartesian coordinates."""

    df: DataFrame[Vector2DSchema] = Vector2DSchema

    @property
    def phi(self) -> pd.DataFrame:
        """Phi coordinate in radial units."""
        d = {"phi": np.arctan((self.df["y"] / self.df["x"]).to_numpy())}
        return pd.DataFrame(d)

    @property
    def norm(self) -> pd.DataFrame:
        """Returns L2-norm of 2D vector."""
        numpy_array = self.df.to_numpy(dtype=np.float)
        d = {"norm": np.linalg.norm(numpy_array, axis=1)}
        return pd.DataFrame(d)

    def _get_scaling_factor_for_length(self, length: float) -> pd.DataFrame:
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
        self.df = self.df.mul(factor)

    @classmethod
    @check_types(with_pydantic=True)
    def from_polar(cls, polar: DataFrame[PolarSchema]):
        """Creates a 2D vector from polar coordinates.

        Args:
            polar: DataFrame with columns (norm, phi)

        Returns:
            2D Vector with the given properties.
        """
        np_polar = polar.to_numpy()
        np_norm = np_polar[:, 0]
        np_phi = np_polar[:, 1]
        d = {
            "x": np.cos(np_phi) * np_norm,
            "y": np.sin(np_phi) * np_norm,
        }
        df = pd.DataFrame(d)
        return cls(df=df)

    @classmethod
    def from_numpy(cls, numpy_array: npt.NDArray[Any]):
        """Creates 2d vector out of numpy array.

        Args:
            numpy_array: numpy array of length two [x, y]

        Returns:
            2d vector by given numpy array

        """
        d = {
            "x": numpy_array[:, 0],
            "y": numpy_array[:, 1],
        }
        df = pd.DataFrame(d)
        return cls(df=df)


class Vectors3D(Vectors2D):
    """A 3D vector with interface to radial and spherical coordinates."""

    df: DataFrame[Vector3DSchema]

    @property
    def theta(self) -> pd.DataFrame:
        """Phi coordinate in radial units."""
        np_ratio = (self.df["z"] / self.norm["norm"]).to_numpy()
        d = {"theta": np.arccos(np_ratio)}
        return pd.DataFrame(d)

    @classmethod
    @check_types(with_pydantic=True)
    def from_polar(cls, polar: DataFrame[PolarSchema]):
        """3D vector cannot be created from polar.

        Raises:
            AttributeError: A creation from polar coordinates is not possible
        """
        raise AttributeError("Vector3D cannot be created by polar coordinates.")

    @classmethod
    @check_types(with_pydantic=True)
    def from_spherical(cls, spherical: DataFrame[SphericalSchema]):
        """Create 3D vector based on spherical coordinates.

        Args:
            spherical: Dataframe with columns (norm, phi, theta)

        Raises:
            AttributeError: A creation from polar coordinates is not possible
        """
        np_spherical = spherical.to_numpy()
        np_norm = np_spherical[:, 0]
        np_phi = np_spherical[:, 1]
        np_theta = np_spherical[:, 2]
        d = {
            "x": np.cos(np_phi) * np.sin(np_theta) * np_norm,
            "y": np.sin(np_phi) * np.sin(np_theta) * np_norm,
            "z": np.cos(np_theta) * np_norm,
        }
        df = pd.DataFrame(d)
        return cls(df=df)

    @classmethod
    def from_numpy(cls, numpy_array: npt.NDArray[Any]):
        """Creates 3d vector out of numpy array.

        Args:
            numpy_array: numpy array of length three [x, y, z]

        Returns:
            3d vector by given numpy array

        """
        d = {"x": numpy_array[:, 0], "y": numpy_array[:, 1], "z": numpy_array[:, 2]}
        df = pd.DataFrame(d)
        return cls(df=df)

    @classmethod
    def from_df(cls, df: pd.DataFrame, prefix: str = ""):
        """Returns a DataFrame of 3d vectors.

        As many of the classes have a prefix in front of the coordinates,
        this method strips the prefix and returns a valid 3dVectors object

        Args:
            df: DataFrame with columns
            prefix: prefix to be stripped

        Returns:
            Valid Vectors3D Object
        """
        mapping = {
            prefix + "x": "x",
            prefix + "y": "y",
            prefix + "z": "z",
        }
        keys = list(mapping.keys())
        renamed_df = df[keys].rename(columns=mapping)
        return Vectors3D(df=renamed_df)

    def get_df_with_prefix(self, prefix: str = "") -> pd.DataFrame:
        """Gets DataFrame from Vectors with prefixed columns for later use.

        Args:
            prefix: prefix to prepend (x,y,z)-columns

        Returns:
            DataFrame with prefixed columns.
        """
        mapping = {
            "x": prefix + "x",
            "y": prefix + "y",
            "z": prefix + "z",
        }
        keys = list(mapping.keys())
        return self.df[keys].rename(columns=mapping)


class LocatedObjects(DataFrameFacade):
    """Object that has a location."""

    df: DataFrame[LocatedObjectSchema]

    @property
    def locations(self) -> Vectors3D:
        """Gets the 3D vectors data frame of the locations."""
        return Vectors3D.from_df(self.df, prefix="location_")


class OrientedLocatedObjects(LocatedObjects):
    """Object that has a location and orientation."""

    df: DataFrame[OrientedLocatedObjectSchema]

    @property
    def orientations(self) -> Vectors3D:
        """Gets the 3D vectors data frame of the orientations."""
        return Vectors3D.from_df(self.df, prefix="orientation_")
