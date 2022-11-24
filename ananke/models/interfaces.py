"""Place for all interfaces used in the package."""
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy.typing as npt


@dataclass
class NumpyRepresentable(Sequence): # type: ignore
    """Interface for making a class numpy representable."""

    #: Numpy array property for easy access and less recalculation
    numpy_array: npt.NDArray[Any] = field(init=False)

    def __post_init__(self) -> None:
        """Update Numpy after Init."""
        self._update_numpy_array()

    def _update_numpy_array(self) -> None:
        """Recalculate the numpy array after change."""
        self.numpy_array = self._get_numpy_array()

    @abstractmethod
    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        """How should the numpy array representation look like?

        Args:
            dtype: Type of the numpy array.

        Returns:
            Generated numpy array.
        """
        raise NotImplementedError("Method __array__ not implemented")

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        """Make class numpy array castable.

        Args:
            dtype: Type of the numpy array.

        Returns:
            Generated numpy array.
        """
        return self.numpy_array

    def __getitem__(self, index: Any) -> Any:
        """Returns slice or item of the created numpy array.

        Args:
            index: index or slice by which to get the value from

        Returns:
            slice or item of the numpy array
        """
        return self.numpy_array[index]

    def __len__(self) -> int:
        """Length of the numpy array.

        Returns:
            Length of the numpy array.
        """
        return len(self.numpy_array)
