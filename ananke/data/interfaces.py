"""Place for all interfaces used in the package."""
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import astuple, dataclass
from typing import Any, Union

import numpy as np
import numpy.typing as npt


class INumpyRepresentable(Sequence[Any]):
    """Interface for making a class numpy representable."""

    @abstractmethod
    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        """How should the numpy array representation look like?"""
        raise NotImplementedError("Method __array__ not implemented")


@dataclass
class NumpyRepresentable(Sequence[Any]):
    """Mixin class that provides some standard implementation for the interface."""

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        """Return a numpy array based on class properties.

        Args:
            dtype: Type of the final array.

        Returns:
            Numpy array representing class.
        """
        return np.array(astuple(self, tuple_factory=list), dtype=dtype)

    def __len__(self) -> int:
        """Determines the length of the dataclass.

        Returns:
            Array lenght of the point

        """
        return astuple(self, tuple_factory=list).__len__()

    def __getitem__(self, item: Union[int, slice]) -> Union[Any, Sequence[Any]]:
        """Get a specific set of dataclass tuple.

        Args:
            item: Item number or slice

        Returns:
            item of point

        """
        return astuple(self, tuple_factory=list).__getitem__(item)
