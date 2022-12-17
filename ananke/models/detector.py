"""Contains all the classes for representing a detector."""
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from ananke.models.geometry import Vectors3D
from ananke.models.interfaces import DataFrameFacade
from ananke.configurations.detector import DetectorConfiguration
from ananke.schemas.detector import PMTSchema, ModuleSchema, StringSchema


class PMTs(DataFrameFacade):
    """Python class representing individual PMT."""
    df: DataFrame[PMTSchema]

    @property
    def pmt_locations(self) -> Vectors3D:
        """Gets DataFrame with (x,y,z)-locations of PMTs."""
        return Vectors3D.from_df(self.df, prefix='pmt_location_')

    @property
    def pmt_orientations(self) -> Vectors3D:
        """Gets DataFrame with (x,y,z)-orientations of PMTs."""
        return Vectors3D.from_df(self.df, prefix='pmt_orientation_')

    @property
    def pmt_areas(self) -> pd.DataFrame:
        """Gets Dataframe with PMT area."""
        return self.df[[
            'pmt_area'
        ]]

    @property
    def pmt_efficiencies(self) -> pd.DataFrame:
        """Gets Dataframe with PMT efficiencies."""
        return self.df[[
            'pmt_efficiency'
        ]]


class Modules(PMTs):
    """Python class representing individual module."""
    df: DataFrame[ModuleSchema]

    @property
    def module_locations(self) -> Vectors3D:
        """Gets DataFrame with (x,y,z)-location of modules."""
        return Vectors3D.from_df(self.df, prefix='module_location_')

    @property
    def module_radius(self) -> pd.DataFrame:
        """Gets Dataframe with Module radius."""
        return self.df[[
            'module_radius'
        ]]


class Strings(Modules):
    """Python class representing individual string."""
    df: DataFrame[StringSchema]

    @property
    def string_locations(self) -> Vectors3D:
        """Gets DataFrame with (x,y,z)-location of modules."""
        return Vectors3D.from_df(self.df, prefix='string_location_')


class Detector(Strings):
    """Python class representing detector."""

    configuration: Optional[DetectorConfiguration]

    @property
    def indices(self) -> pd.DataFrame:
        return self.df[[
            'string_id',
            'module_id',
            'pmt_id'
        ]]

    @property
    def outer_radius(self) -> float:
        """Returns the distance of the farthest out module."""
        return np.linalg.norm(self.module_locations.df.to_numpy(), axis=1).max()

    @property
    def outer_cylinder(self) -> Tuple[float, float]:
        """Returns a tuple of the height and radius of the outer cylinder."""
        module_locations = self.module_locations.df.to_numpy()

        return (
            np.linalg.norm(module_locations[:, :2], axis=1).max(),
            2 * np.abs(module_locations[:, 2].max()),
        )

