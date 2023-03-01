import os
import unittest
from unittest import mock

import pandas as pd

from ananke.configurations.collection import HDF5StorageConfiguration
from ananke.configurations.events import Interval
from ananke.models.event import Hits
from ananke.schemas.event import RecordType
from ananke.services.collection.storage import HDF5CollectionStorage, HDF5StorageKeys


class HDFStorageTestCase(unittest.TestCase):
    tmp_collection_path: str = '_tmp/hdfstoragetest_collection.h5'

    def setUp(self) -> None:
        self.configuration = HDF5StorageConfiguration(data_path=self.tmp_collection_path)
        self.storage = HDF5CollectionStorage(self.configuration)
        self.storage.open()

    def tearDown(self) -> None:
        self.storage.close()
        os.remove(self.tmp_collection_path)


    def test_get_where_record_id(self):
        record_ids = 1
        self.assertEqual(
            '((record_id={}))'.format(record_ids),
            self.storage._HDF5CollectionStorage__get_where(record_ids=record_ids)
        )

        multiple_record_ids = [1, 2]
        multiple_record_ids_expected = '((record_id=1 | record_id=2))'
        self.assertEqual(
            multiple_record_ids_expected,
            self.storage._HDF5CollectionStorage__get_where(record_ids=multiple_record_ids)
        )
        self.assertEqual(
            multiple_record_ids_expected,
            self.storage._HDF5CollectionStorage__get_where(record_ids=pd.Series(multiple_record_ids))
        )

    def test_get_where_type(self):
        record_type = RecordType.CASCADE
        self.assertEqual(
            '((type={}))'.format(record_type.value),
            self.storage._HDF5CollectionStorage__get_where(record_types=record_type)
        )

        multiple_record_types = [RecordType.CASCADE, RecordType.ELECTRICAL]
        multiple_record_types_expected = '((type={} | type={}))'.format(multiple_record_types[0].value,
                                                                        multiple_record_types[1].value)
        self.assertEqual(
            multiple_record_types_expected,
            self.storage._HDF5CollectionStorage__get_where(record_types=multiple_record_types)
        )

    def test_get_where_interval(self):
        interval = Interval(start=0, end=1000)
        self.assertEqual(
            '((time>={} & time<{}))'.format(interval.start, interval.end),
            self.storage._HDF5CollectionStorage__get_where(interval=interval)
        )

    def test_get_where_combined(self):
        self.assertIsNone(self.storage._HDF5CollectionStorage__get_where())

        interval = Interval(start=0, end=1000)
        record_ids = 1
        multiple_record_types = [RecordType.CASCADE, RecordType.ELECTRICAL]
        self.assertEqual(
            '((type={} | type={}) & (record_id=1) & (time>={} & time<{}))'.format(
                multiple_record_types[0].value,
                multiple_record_types[1].value,
                interval.start,
                interval.end
            ),
            self.storage._HDF5CollectionStorage__get_where(
                record_ids=record_ids,
                record_types=multiple_record_types,
                interval=interval
            )
        )

    def test_get_unique_record_ids(self):
        unique_ids = self.storage._HDF5CollectionStorage__get_unique_records_ids(
                key=HDF5StorageKeys.HITS
            )
        self.assertEquals(
            0,
            len(unique_ids)
        )
        self.storage.set_hits(Hits(df=pd.DataFrame({
            'type': RecordType.CASCADE.value,
            'record_id': [45,2, 45],
            'pmt_id': 1,
            'string_id': 1,
            'module_id': [1,2,3],
            'time': [1,100,1000]
        })))
        unique_ids = self.storage._HDF5CollectionStorage__get_unique_records_ids(
                key=HDF5StorageKeys.HITS
            )
        self.assertListEqual(
            [45, 2],
            list(unique_ids.values)
        )

    def test_next_record_ids(self):
        with mock.patch.object(self.storage, '_HDF5CollectionStorage__get_unique_records_ids', return_value=pd.Series([])):
            self.assertEqual(
                [0],
                self.storage.get_next_record_ids(1)
            )

            self.assertEqual(
                [0,1],
                self.storage.get_next_record_ids(2)
            )
        with mock.patch.object(self.storage, '_HDF5CollectionStorage__get_unique_records_ids', return_value=pd.Series([1,45,23])):
            self.assertEqual(
                [46],
                self.storage.get_next_record_ids(1)
            )

            self.assertEqual(
                [46,47],
                self.storage.get_next_record_ids(2)
            )



if __name__ == '__main__':
    unittest.main()
