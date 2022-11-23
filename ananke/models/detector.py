"""Contains all the classes for representing a detector."""
import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Any, Optional

from numpy import typing as npt

from ananke.models.geometry import LocatedObject, OrientedLocatedObject
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
        oriented_array = super()._get_numpy_array(dtype=dtype)
        return np.insert(oriented_array, 0, self.ID)


@dataclass
class Module(LocatedObject):
    """Python class representing individual module."""

    #: Index of the current module
    ID: int

    #: Module PMTs
    PMTs: Optional[List[PMT]] = None

    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        # create numpy array without PMTs
        if self.PMTs is None:
            module_array = super()._get_numpy_array(dtype=dtype)

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
