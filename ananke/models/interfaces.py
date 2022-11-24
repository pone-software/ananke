"""Place for all interfaces used in the package."""
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class ScientificConvertible(Sequence): # type: ignore
    """Interface for making a class numpy representable."""

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        """Converts object to pandas dataframe.

        Returns:
            Pandas dataframe of object

        Raises:
            NotImplementedError: method not implemented in child
        """

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        """Make class numpy array castable.

        Args:
            dtype: Type of the numpy array.

        Returns:
            Generated numpy array.
        """
        return np.array(self.to_pandas(), dtype=dtype)

    def __getitem__(self, index: Any) -> Any:
        """Returns slice or item of the created numpy array.

        Args:
            index: index or slice by which to get the value from

        Returns:
            slice or item of the numpy array
        """
        return np.array(self)[index]

    def __len__(self) -> int:
        """Length of the numpy array.

        Returns:
            Length of the numpy array.
        """
        return len(np.array(self))
