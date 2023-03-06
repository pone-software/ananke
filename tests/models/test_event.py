import unittest

import pandas as pd

from ananke.models.event import RecordIds, RecordTimes
from ..common.schemas import get_sources


class RecordIdsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.record_ids = RecordIds(
            df=pd.DataFrame(
                {
                    'record_id': [1, 2, 3]
                }
            )
        )

    def test_get_by_record(self) -> None:
        record_ids = self.record_ids

        assert record_ids.get_by_record_ids(record_ids=1) == RecordIds(
            df=pd.DataFrame(
                {
                    'record_id': [1]
                }
            )
        )

        assert record_ids.get_by_record_ids(record_ids=[2, 3]) == RecordIds(
            df=pd.DataFrame(
                {
                    'record_id': [2, 3]
                }
            )
        )

        assert record_ids.get_by_record_ids(record_ids=pd.Series([2, 3])) == RecordIds(
            df=pd.DataFrame(
                {
                    'record_id': [2, 3]
                }
            )
        )

        self.assertIsNone(record_ids.get_by_record_ids(record_ids=5000))

    def test_record_ids_property(self) -> None:
        self.assertIsInstance(self.record_ids.record_ids, pd.Series)
        self.assertTrue(
            self.record_ids.df['record_id'].equals(self.record_ids.record_ids)
        )


class RecordTimesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.record_times = RecordTimes(
            df=pd.DataFrame(
                {
                    'time': [1, 2, 3]
                }
            )
        )

    def test_times_property(self) -> None:
        self.assertIsInstance(self.record_times.times, pd.Series)
        self.assertTrue(self.record_times.df['time'].equals(self.record_times.times))

    def test_add_time(self) -> None:
        self.record_times.add_time(1)
        self.assertTrue(pd.Series([2., 3., 4.]).equals(self.record_times.times))
        self.record_times.add_time([-1., -2., -3.])
        self.assertTrue(pd.Series([1., 1., 1.]).equals(self.record_times.times))


class SourcesTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.sources = get_sources()

    def test_times_property(self) -> None:
        self.assertIsInstance(self.sources.number_of_photons, pd.Series)
        self.assertTrue(
            self.sources.df['number_of_photons'].equals(self.sources.number_of_photons)
        )


if __name__ == '__main__':
    unittest.main()
