"""Contains all the classes for representing a detector."""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from ananke.models.geometry import LocatedObject, OrientedLocatedObject, Vector3D
from ananke.models.interfaces import ScientificSequence


@dataclass
class PMT(OrientedLocatedObject):
    """Python class representing individual PMT."""

    #: Index of the current PMT
    ID: int

    #: PMT efficiency
    efficiency: float

    #: Opening area of the PMT
    area: float

    #: PMT noise rate 1/value
    noise_rate: float

    def to_pandas(self) -> pd.DataFrame:
        """Gets the dataframe of the PMT class.

        Generates dataframe of the PMT.

        Returns:
            Dataframe containing pmt information

        """
        dataframe = super().to_pandas()
        dataframe = dataframe.assign(
            pmt_id=self.ID,
            pmt_efficiency=self.efficiency,
            pmt_area=self.area,
            pmt_noise_rate=self.noise_rate,
            pmt_x=lambda value: value.location_x + value.orientation_x,
            pmt_y=lambda value: value.location_y + value.orientation_x,
            pmt_z=lambda value: value.location_z + value.orientation_x,
        )
        dataframe = dataframe.rename(
            columns={
                'orientation_x': 'pmt_orientation_x',
                'orientation_y': 'pmt_orientation_y',
                'orientation_z': 'pmt_orientation_z',
            }
        )
        dataframe = dataframe.drop(
            [
                'location_x',
                'location_y',
                'location_z',
            ], axis=1
        )
        columns = dataframe.columns.tolist()

        return dataframe[columns[3:] + columns[:3]]
    @property
    def surface_location(self) -> Vector3D:
        """Returns the final PMT location (location-vector + orientation-vector).

        Returns:
            Vector with final location
        """
        return self.location + self.orientation


@dataclass
class Module(LocatedObject):
    """Python class representing individual module."""

    #: Index of the current module
    ID: int

    #: radius of the module
    radius: float

    #: Module PMTs
    PMTs: List[PMT]

    def to_pandas(self) -> pd.DataFrame:
        """Gets the dataframe of the module class.

        Generates a list of the dataframes of the module.

        Returns:
            Dataframe containing pmt information

        """

        pmt_dataframes = []

        for pmt in self.PMTs:
            pmt_dataframes.append(pmt.to_pandas())

        dataframe = pd.concat(pmt_dataframes, ignore_index=True)

        return dataframe.assign(
            module_id=self.ID,
            module_radius=self.radius,
            module_x=self.location.x,
            module_y=self.location.y,
            module_z=self.location.z,
        )

    @property
    def pmt_locations(self) -> List[Vector3D]:
        """Aggregates all locations of pmts in one list.

        Returns:
            List of pmt locations

        Raises:
            AttributeError: Only possible when modules have pmts
        """
        pmt_locations = []

        for pmt in self.PMTs:
            pmt_locations.append(pmt.surface_location)

        return pmt_locations


@dataclass
class String(LocatedObject):
    """Python class representing individual string."""

    #: Index of the current string
    ID: int

    #: Modules in string
    modules: List[Module]

    def to_pandas(self) -> pd.DataFrame:
        """Gets the dataframe of the string class.

        Generates a list of the dataframes of the strings.

        Returns:
            Dataframe containing pmt information

        """
        module_frames = []

        for module in self.modules:
            module_frames.append(module.to_pandas())

        dataframe = pd.concat(module_frames, ignore_index=True)

        return dataframe.assign(string_id=self.ID)

    @property
    def module_locations(self) -> List[Vector3D]:
        """Aggregates all locations of modules in one list.

        Returns:
            List of module locations
        """
        module_locations = []

        for module in self.modules:
            module_locations.append(module.location)

        return module_locations

    @property
    def pmt_locations(self) -> List[Vector3D]:
        """Aggregates all locations of pmts in one list.

        Returns:
            List of pmt locations

        Raises:
            AttributeError: Only possible when modules have pmts
        """
        pmt_locations = []

        for module in self.modules:
            pmt_locations += module.pmt_locations

        return pmt_locations


@dataclass
class Detector(ScientificSequence):
    """Python class representing detector."""

    #: list of detector strings
    strings: List[String]

    def to_pandas(self) -> pd.DataFrame:
        """Gets the dataframe of the detector class.

        Generates a list of the dataframes of the strings.

        Returns:
            Dataframe containing pmt information

        """
        string_arrays = []

        for string in self.strings:
            string_arrays.append(string.to_pandas())

        return pd.concat(string_arrays, ignore_index=True)

    @property
    def module_locations(self) -> List[Vector3D]:
        """Aggregates all locations of modules in one list.

        Returns:
            List of module locations
        """
        module_locations = []

        for string in self.strings:
            module_locations += string.module_locations

        return module_locations

    @property
    def pmt_locations(self) -> List[Vector3D]:
        """Aggregates all locations of pmts in one list.

        Returns:
            List of pmt locations

        Raises:
            AttributeError: Only possible when modules have pmts
        """
        pmt_locations = []

        for string in self.strings:
            pmt_locations += string.pmt_locations

        return pmt_locations
