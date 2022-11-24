"""Contains all the classes for representing a detector."""
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Any, Optional

from numpy import typing as npt

from ananke.models.geometry import LocatedObject, OrientedLocatedObject, Vector3D
from ananke.models.interfaces import NumpyRepresentable


@dataclass
class PMT(OrientedLocatedObject):
    """Python class representing individual PMT."""

    #: Index of the current PMT
    ID: int

    #: PMT efficiency
    efficiency: float

    #: PMT noise rate 1/x
    noise_rate: float

    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        """Returns the orientated array and adds the pmt id.

        Args:
            dtype: type of the numpy array

        Returns:
            Array containing the orientation and id.
        """
        oriented_array = super()._get_numpy_array(dtype=dtype)
        return np.insert(oriented_array, 0, self.ID)

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
    PMTs: Optional[List[PMT]] = None

    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        # create numpy array without PMTs
        if self.PMTs is None:
            module_array = np.array(self.location)

            # add ID
            module_array = np.insert(module_array, 0, self.ID)

            # append radius
            module_array = np.append(module_array, self.location.norm)

            return module_array

        module_arrays = []

        for pmt in self.PMTs:
            current_array = np.array(pmt)
            current_array = np.insert(current_array, 0, self.ID)
            module_arrays.append(current_array)

        return np.array(module_arrays, dtype=dtype)

    @property
    def pmt_locations(self) -> List[Vector3D]:
        """Aggregates all locations of pmts in one list.

        Returns:
            List of pmt locations

        Raises:
            AttributeError: Only possible when modules have pmts
        """
        if self.PMTs is None:
            raise AttributeError("Cannot create PMT Coordinates without PMTs")

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

    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        string_arrays = []

        for module in self.modules:
            current_array = np.array(module, dtype=dtype)
            current_array = np.insert(current_array, 0, self.ID, axis=1)
            string_arrays.append(current_array)

        return np.concatenate(string_arrays, dtype=dtype)

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
class Detector(NumpyRepresentable):
    """Python class representing detector."""

    #: list of detector strings
    strings: List[String]

    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        string_arrays = []

        for string in self.strings:
            string_arrays.append(np.array(string))

        return np.concatenate(string_arrays, dtype=dtype)

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
