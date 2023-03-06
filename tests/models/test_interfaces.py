import unittest

import numpy as np
import pandas as pd

import pandera as pa
from pandera.typing import Series, DataFrame

from ananke.models.interfaces import DataFrameFacade, DataFrameFacadeIterator


class MockDataFrameFacadeSchema(pa.SchemaModel):
    a: Series[int]


class MockDataFrameFacade(DataFrameFacade):
    df: DataFrame[MockDataFrameFacadeSchema]


class DataFrameFacadeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.array = np.array([1, 2, 3, 4])
        self.array_extra_axis = self.array[:, np.newaxis]
        self.dataframe = pd.DataFrame(
            {
                'a': self.array
            }
        )
        self.dataframe_facade = MockDataFrameFacade(df=self.dataframe)

    def test_array_conversion(self):
        np.testing.assert_array_equal(
            self.array_extra_axis,
            self.dataframe_facade.to_numpy()
        )
        np.testing.assert_array_equal(
            self.array_extra_axis,
            np.array(self.dataframe_facade)
        )

    def test_length(self):
        assert len(self.dataframe_facade) == 4

    def test_equal(self):
        assert self.dataframe_facade == self.dataframe_facade
        assert self.dataframe_facade != MockDataFrameFacade(df=pd.DataFrame({'a': [2]}))

    def test_itertuples(self):
        for index, tuple in enumerate(self.dataframe_facade.itertuples()):
            assert self.array[index] == getattr(tuple, 'a')

    def test_sample(self):
        rng = np.random.default_rng(3)
        rng2 = np.random.default_rng(3)
        assert self.dataframe_facade.sample(1, rng).df.equals(
            self.dataframe_facade.df.sample(n=1, random_state=rng2)
        )

        with self.assertRaises(ValueError, msg='Throw error when not enough elements'):
            self.dataframe_facade.sample(15)

    def test_concat(self):
        self.assertIsNone(MockDataFrameFacade.concat([]))
        self.assertIsNone(MockDataFrameFacade.concat([None, None]))

        assert MockDataFrameFacade.concat(
            [None, self.dataframe_facade]
        ) == self.dataframe_facade

        assert MockDataFrameFacade.concat(
            [self.dataframe_facade, self.dataframe_facade]
        ) == MockDataFrameFacade(
            df=pd.DataFrame(
                index=[0, 1, 2, 3, 0, 1, 2, 3],
                data={
                    'a': [1, 2, 3, 4, 1, 2, 3, 4]
                }
            )
        )

        self.assertIsInstance(
            MockDataFrameFacade.concat([self.dataframe_facade]),
            MockDataFrameFacade
        )

    def test_data_frame_facade_iterator(self):
        with self.assertRaises(ValueError):
            self.dataframe_facade.iterbatches(-2)

        count = 0
        batch_size = 2
        dataframe_iterator = self.dataframe_facade.iterbatches(batch_size=2)
        assert isinstance(dataframe_iterator, DataFrameFacadeIterator)
        assert dataframe_iterator.facade == self.dataframe_facade

        for index, current_facade in enumerate(dataframe_iterator):
            count += 1
            current_index = index * batch_size
            self.assertTrue(
                self.dataframe_facade.df.iloc[
                current_index:current_index + batch_size].equals(
                    current_facade.df
                )
            )

        assert count == 2
