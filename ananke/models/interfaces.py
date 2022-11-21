"""Place for all interfaces used in the package."""
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import astuple, dataclass, field
from typing import Any, Union

import numpy as np
import numpy.typing as npt


@dataclass
class NumpyRepresentable(Sequence):
    """Interface for making a class numpy representable."""
    numpy_array: npt.NDArray[Any] = field(init=False)

    def __post_init__(self):
        self.numpy_array = self._get_numpy_array()

    @abstractmethod
    def _get_numpy_array(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        """How should the numpy array representation look like?"""
        raise NotImplementedError("Method __array__ not implemented")

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        return self.numpy_array

    def __getitem__(self, index: Any) -> Any:
        return self.numpy_array[index]

    def __len__(self) -> int:
        return len(self.numpy_array)
