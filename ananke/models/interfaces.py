"""Place for all interfaces used in the package."""
from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Iterable, Optional, Sequence, TypeVar

import numpy as np
import numpy.typing as npt
import pandas as pd

from pandera.typing import DataFrame
from pydantic import BaseModel


T_ = TypeVar("T_")
DataFrameFacade_ = TypeVar("DataFrameFacade_", bound="DataFrameFacade")


class DataFrameFacadeIterator(Iterator[DataFrameFacade_]):
    """Class serving as iterator for the DataFrameFacade's."""

    def __init__(self, facade: DataFrameFacade_, batch_size: int = 1) -> None:
        """Constructor of the Iterator.

        Args:
            facade: Facade to iterate over
            batch_size: Batch size of each individual iteration
        """
        self.facade: DataFrameFacade_ = facade
        self.__number_of_items = len(self.facade)
        self.__facade_class = facade.__class__
        if batch_size <= 0:
            raise ValueError("Batch size must be positive int")
        self.__batch_size = batch_size
        self.__current = 0

    def __iter__(self) -> DataFrameFacadeIterator:
        """Get iterator object.

        Returns:
            Itself as iterator object
        """
        return self

    def __next__(self) -> DataFrameFacade_:
        """Get next element of iterator.

        Returns:
            Dataframe facade of current batch of data.
        """
        current_index = self.__current
        if current_index >= self.__number_of_items:
            raise StopIteration
        next_index = current_index + self.__batch_size
        self.__current = next_index
        current_facade = self.__facade_class.construct(
            self.__facade_class.__fields_set__,
            df=self.facade.df.iloc[current_index:next_index],
        )
        return current_facade


class DataFrameFacade(BaseModel):
    """Interface for making a class numpy representable."""

    df: DataFrame[T_]

    def to_numpy(self, dtype: npt.DTypeLike = None) -> npt.NDArray[Any]:
        """Converts data frame to numpy array.

        Args:
            dtype: Type of the final dataframe.

        Returns:
            Numpy array based on data frame.
        """
        return self.df.to_numpy(dtype=dtype)

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

    def __eq__(self, other: DataFrameFacade) -> bool:
        """Checks whether it is equal to another facade.

        Args:
            other: Facade to check equality against

        Returns:
            True or false based on dataframe equality.
        """
        return self.df.equals(other.df)

    @classmethod
    def concat(
        cls, facades_to_concat: Sequence[Optional[DataFrameFacade]]
    ) -> Optional[DataFrameFacade]:
        """Concats multiple facades to one.

        Args:
            facades_to_concat: List of facades to concat

        Returns:
            Concatenated Facade

        """
        if len(facades_to_concat) == 0:
            return None
        dfs = []
        for facade in facades_to_concat:
            if facade is not None:
                dfs.append(facade.df)
        if len(dfs) == 0:
            return None

        # TODO: Evaluate whether to drop index or not?
        full_df = pd.concat(dfs)
        return cls.construct(cls.__fields_set__, df=full_df)

    def sample(
        self, n: int = 1, random_state: Optional[np.random.Generator] = None
    ) -> DataFrameFacade:
        """Returns class with a random sample of its dataframe rows.

        Args:
            n: number of samples to draw.
            random_state: random state to choose

        Raises:
            ValueError: When fewer samples than required are available.

        Returns:
            Class with the given number of random samples.
        """
        if len(self) < n:
            raise ValueError("Only {} of needed {} rows available".format(len(self), n))
        return self.__class__(df=self.df.sample(n=n, random_state=random_state))

    def iterbatches(
        self, batch_size: int = 1
    ) -> DataFrameFacadeIterator[DataFrameFacade_]:
        """Get iterator of current facade.

        Args:
            batch_size: How many rows in each iteration

        Returns:
            Iterator for batched dataframe iteration
        """
        return DataFrameFacadeIterator(facade=self, batch_size=batch_size)

    def itertuples(self, index: bool = True) -> Iterable[tuple[Any, ...]]:
        """Pass through itertuples from dataframe with name.

        Args:
            index: Whether index should be part of the Tuple

        Returns:
            Itertuples iterator of dataframe.
        """
        return self.df.itertuples(index=index, name=self.__class__.__name__)
