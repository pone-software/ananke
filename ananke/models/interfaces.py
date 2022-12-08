"""Place for all interfaces used in the package."""
from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, MutableSequence, TypeVar, Optional, List

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclass
class ScientificSequence(Sequence):
    """Interface for making a class numpy representable."""

    @abstractmethod
    def to_pandas(self) -> pd.DataFrame:
        """Converts object to pandas dataframe.

        Returns:
            Pandas dataframe of object

        Raises:
            NotImplementedError: method not implemented in child
        """

    def to_numpy(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        array = self.to_pandas().to_numpy(dtype=dtype)  # type: npt.NDArray[Any]
        return array.flatten()

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        """Make class numpy array castable.

        Args:
            dtype: Type of the numpy array.

        Returns:
            Generated numpy array.
        """
        return self.to_numpy(dtype=dtype)

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


ScientificSequenceType = TypeVar(
    'ScientificSequenceType',
    bound=ScientificSequence
)


class ScientificCollection(MutableSequence[ScientificSequenceType]):
    """Summarize scientific convertibles."""

    def __init__(
            self,
            sequence: Optional[List[ScientificSequenceType]] = None
    ) -> None:
        """Constructor of the ScientificCollection.

        Args:
            sequence: initial sequence
        """
        if sequence is None:
            sequence = []

        self._sequence = sequence

    def insert(self, index: int, value: ScientificSequenceType) -> None:
        """Insert an element into the sequence

        Args:
            index: Index to insert
            value: Value to insert
        """
        self._sequence.insert(index, value)

    def __add__(self, other: ScientificCollection) -> \
            ScientificCollection:
        return ScientificCollection(self._sequence + other._sequence)

    def __getitem__(self, index: Any) -> Any:
        """Gets element(s) from the sequence

        Args:
            index: integer or slice to get

        Returns:
            MutableSequence or individual of type depending on index.
        """
        return self._sequence[index]

    def __setitem__(self, index: Any, value: ScientificSequenceType) -> None:
        """Sets element(s) to the sequence

        Args:
            index: integer or slice to set
            value: value(s) to set
        """
        self._sequence[index] = value

    def __delitem__(self, index: int) -> None:
        """Deletes element from the sequence

        Args:
            index: Index of element to delete.
        """
        del self._sequence[index]

    def __len__(self) -> int:
        """Returns length of the sequence

        Returns:
            length of the sequence
        """
        return len(self._sequence)

    def to_pandas(self) -> pd.DataFrame:
        """Generate dataframe based on sub dataframes of the sequence

        Returns:
            Combined dataframe of all items
        """
        dataframes = []

        for item in self._sequence:
            dataframes.append(item.to_pandas())

        if not len(dataframes):
            return pd.DataFrame()

        return pd.concat(dataframes, ignore_index=True)
