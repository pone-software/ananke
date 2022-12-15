"""Place for all interfaces used in the package."""
from __future__ import annotations

from pydantic import BaseModel, validator
from typing import Any, List
import numpy.typing as npt
from pandera.typing import DataFrame
import pandas as pd


class DataFrameFacade(BaseModel):
    """Interface for making a class numpy representable."""
    df: DataFrame

    def to_numpy(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        array = self.df.to_numpy(dtype=dtype)  # type: npt.NDArray[Any]
        return array.flatten()

    def __array__(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        """Make class numpy array castable.

        Args:
            dtype: Type of the numpy array.

        Returns:
            Generated numpy array.
        """
        return self.df.to_numpy(dtype=dtype)

    def __len__(self) -> int:
        """Length of the numpy array.

        Returns:
            Length of the numpy array.
        """
        return len(self.df.index)

    @classmethod
    def concat(cls, facades_to_concat: List[DataFrameFacade]) -> DataFrameFacade:
        """Concats multiple facades to one.

        Args:
            facades_to_concat: List of facades to concat

        Returns:
            Concatenated Facade

        """
        dfs = []
        for facade in facades_to_concat:
            dfs.append(facade.df)

        return pd.concat(dfs)
