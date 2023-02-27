"""Contains all the classes for representing a detector."""
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

from ananke.configurations.detector import DetectorConfiguration
from ananke.models.geometry import Vectors3D
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.detector import DetectorSchema, ModuleSchema, PMTSchema
from pandera.typing import DataFrame


class PMTs(DataFrameFacade):
    """Python class representing individual PMT."""

    df: DataFrame[PMTSchema]

    @property
    def pmt_locations(self) -> Vectors3D:
        """Gets DataFrame with (x,y,z)-locations of PMTs."""
        return Vectors3D.from_df(self.df, prefix="pmt_location_")

    @property
    def pmt_orientations(self) -> Vectors3D:
        """Gets DataFrame with (x,y,z)-orientations of PMTs."""
        return Vectors3D.from_df(self.df, prefix="pmt_orientation_")

    @property
    def pmt_areas(self) -> pd.Series:
        """Gets Dataframe with PMT area."""
        return self.df["pmt_area"]

    @property
    def pmt_efficiencies(self) -> pd.Series:
        """Gets Dataframe with PMT efficiencies."""
        return self.df["pmt_efficiency"]

    @property
    def pmt_noise_rates(self) -> pd.Series:
        """Gets Dataframe with PMT noise rates."""
        return self.df["pmt_noise_rate"]


class Modules(PMTs):
    """Python class representing individual module."""

    df: DataFrame[ModuleSchema]

    @property
    def module_locations(self) -> Vectors3D:
        """Gets DataFrame with (x,y,z)-location of modules."""
        return Vectors3D.from_df(self.df, prefix="module_location_")

    @property
    def module_radius(self) -> pd.Series:
        """Gets Dataframe with Module radius."""
        return self.df["module_radius"]


class Strings(Modules):
    """Python class representing individual string."""

    df: DataFrame[DetectorSchema]

    @property
    def string_locations(self) -> Vectors3D:
        """Gets DataFrame with (x,y,z)-location of modules."""
        return Vectors3D.from_df(self.df, prefix="string_location_")


class Detector(Strings):
    """Python class representing detector."""

    configuration: Optional[DetectorConfiguration]

    id_columns: List[str] = ["string_id", "module_id", "pmt_id"]

    @property
    def indices(self) -> pd.DataFrame:
        """Returns all indices of the PMTs."""
        return self.df[self.id_columns]

    @property
    def outer_radius(self) -> float:
        """Returns the distance of the farthest out module."""
        return float(np.linalg.norm(self.module_locations.df.to_numpy(), axis=1).max())

    @property
    def outer_cylinder(self) -> Tuple[float, float]:
        """Returns a tuple of the height and radius of the outer cylinder."""
        module_locations = self.module_locations.df.to_numpy()

        return (
            float(np.linalg.norm(module_locations[:, :2], axis=1).max()),
            float(2 * np.abs(module_locations[:, 2].max())),
        )

    def __number_of_ids(
            self,
            slice: slice = slice(0, 3)
    ) -> int:
        """Returns number of elements based on the column id slice.

        Args:
            slice: slice by which to select columns

        Returns:
            number of unique combinations for given id columns
        """
        return len(self.df.groupby(self.id_columns[slice]))

    @property
    def number_of_pmts(self) -> int:
        """Gets total number of pmts."""
        return self.__number_of_ids(slice(0, 3))

    @property
    def number_of_modules(self) -> int:
        """Gets total number of modules."""
        return self.__number_of_ids(slice(0, 2))

    @property
    def number_of_strings(self) -> int:
        """Gets total number of strings."""
        return self.__number_of_ids(slice(0, 1))
