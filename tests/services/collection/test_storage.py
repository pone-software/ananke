import os
import unittest
from abc import ABC, abstractmethod
from typing import Type, Callable, TypeVar, Optional, TYPE_CHECKING
from unittest import mock
from unittest.mock import patch, MagicMock

import pandas as pd
from ...common.schemas import get_records, get_hits, get_sources, get_detector

from ananke.configurations.collection import (
    HDF5StorageConfiguration,
    StorageConfiguration,
)
from ananke.configurations.events import Interval
from ananke.models.event import Hits
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import RecordType
from ananke.services.collection.storage import (
    HDF5CollectionStorage,
    HDF5StorageKeys,
    AbstractCollectionStorage, StorageFactory,
)

if TYPE_CHECKING:
    _Base = unittest.TestCase
else:
    _Base = object

DataFrameFacade_ = TypeVar('DataFrameFacade_', bound=DataFrameFacade)


class HDFStorageKeysTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.names = [
            'RECORDS',
            'HITS',
            'SOURCES',
            'DETECTOR'
        ]

        self.values = [key.lower() for key in self.names]

    def test_enum(self) -> None:
        self.assertListEqual(
            self.names,
            [e.name for e in HDF5StorageKeys],
            "Enum names not as expected."
        )
        self.assertListEqual(
            self.values,
            [e.value for e in HDF5StorageKeys],
            "Enum values not as expected."
        )

    def test_string_conversion(self) -> None:
        self.assertEqual(
            'records',
            str(HDF5StorageKeys.RECORDS),
            'string conversion broken'
        )


class AbstractStorageCollectionTestCase(unittest.TestCase):

    @patch.multiple(AbstractCollectionStorage, __abstractmethods__=set())
    def test_construction(self) -> None:
        configuration = StorageConfiguration(type="test")
        storage = AbstractCollectionStorage(configuration)

        self.assertEqual(
            configuration,
            storage.configuration,
            "Setting the configuration is broken"
        )


class StorageTests(_Base):
    """Tests to implement by all storage implementations"""

    @classmethod
    def setUpClass(cls):
        """On inherited classes, run our `setUp` and `tearDown` method"""
        # Inspired via http://stackoverflow.com/questions/1323455/python-unit-test-with-base-and-sub-class/17696807#17696807  # noqa
        if cls != StorageTests and cls.setUp != StorageTests.setUp:
            orig_setUp = cls.setUp

            def setUpOverride(self, *args, **kwargs):
                StorageTests.setUp(self)
                return orig_setUp(self, *args, **kwargs)

            cls.setUp = setUpOverride

        if cls != StorageTests and cls.tearDown != StorageTests.tearDown:
            orig_tearDown = cls.tearDown

            def tearDownOverride(self, *args, **kwargs):
                ret_val = orig_tearDown(self, *args, **kwargs)
                StorageTests.tearDown(self)
                return ret_val

            cls.tearDown = tearDownOverride

    def setUp(self):
        """Perform some setup actions"""
        # These still run despite overrides =D
        self.configuration = self._get_configuration()
        self.storage = self.get_storage()
        self.storage.open()

    def tearDown(self):
        """Perform some teardown actions"""
        self.storage.close()
        self.storage.delete()

    def get_storage(self) -> AbstractCollectionStorage:
        return self._get_storage_class()(self._get_configuration())

    @abstractmethod
    def _get_storage_class(self) -> Type[AbstractCollectionStorage]:
        """Get freshly created storage"""
        pass

    @abstractmethod
    def _get_configuration(self) -> StorageConfiguration:
        """Get configuration for current storage"""
        pass

    def test_open(self) -> None:
        self.storage.open()
        self.assertIsNone(
            self.storage.get_detector(),
            "Not working after opening"
        )

    def test_close(self) -> None:
        self.storage.close()
        with self.assertRaises(
                ValueError,
                msg="Open check not implemented"
        ):
            self.storage.get_detector()

    def test_readonly(self) -> None:
        self.storage._read_only = True
        hits = get_hits()
        with self.assertRaises(PermissionError, msg='ReadOnly not implemented'):
            self.storage.set_hits(hits)
        self.storage._read_only = False

        self.storage.set_hits(hits)
        self.assertEqual(
            hits,
            self.storage.get_hits(),
            'Not writing despite writing rights'
        )

    def _get_storage_callables_by_name(self, name: str):
        return (
            getattr(self.storage, 'get_{}'.format(name)),
            getattr(self.storage, 'set_{}'.format(name)),
            getattr(self.storage, 'del_{}'.format(name)),
        )

    def _test_get_set_append_and_delete(
            self,
            name: str,
            data: DataFrameFacade_
    ):
        (getter, setter, deleter) = self._get_storage_callables_by_name(name)
        self.assertIsNone(
            getter(),
            "Getter not none before setting"
        )

        setter(data, False)
        self.assertEqual(
            data,
            getter(),
            "Getter not data after setting"
        )

        setter(data, True)
        self.assertEqual(
            type(data).concat([data, data]),
            getter(),
            "Getter not double data after appending"
        )

        setter(data, False)
        self.assertEqual(
            data,
            getter(),
            "Getter not single data after not appending"
        )

        self.storage.close()
        self.storage.open()
        self.assertEqual(
            data,
            getter(),
            "Getter not data after closing"
        )

        deleter()
        self.assertIsNone(
            getter(),
            "Getter not none after deleting"
        )

    def _test_get_filtered(
            self,
            name: str,
            data: DataFrameFacade_
    ):
        (getter, setter, deleter) = self._get_storage_callables_by_name(name)

        setter(data, False)
        interval = None

        total_mask: Optional[pd.Series] = None

        if 'time' in data.df.columns:
            interval = Interval(start=0.0, end=10.)

            time_condition = (data.df['time'] >= interval.start) & \
                             (data.df['time'] < interval.end)
            self.assertTrue(
                data.df[time_condition].equals(getter(interval=interval).df),
                "Only contains right elements."
            )

            self.assertIsNone(
                getter(interval=Interval(start=10000, end=100000)),
                "Not none empty getter"
            )

            if total_mask is None:
                total_mask = time_condition
            else:
                total_mask = total_mask & time_condition

        record_id = None

        if 'record_id' in data.df.columns:
            record_id = 1

            record_id_condition = (data.df['record_id'] == record_id)
            self.assertTrue(
                data.df[record_id_condition].equals(getter(record_ids=record_id).df),
                "Single record_id contain."
            )

            record_ids = [0, 1]

            record_ids_condition = (data.df['record_id'].isin(record_ids))
            self.assertTrue(
                data.df[record_ids_condition].equals(getter(record_ids=record_ids).df),
                "Multiple record_ids contain."
            )

            self.assertIsNone(
                getter(record_ids=1000),
                "Not none empty getter"
            )

            if total_mask is None:
                total_mask = record_id_condition
            else:
                total_mask = total_mask & record_id_condition

        type_ = None

        if 'type' in data.df.columns:
            type_ = RecordType.CASCADE
            type2_ = RecordType.ELECTRICAL

            type_condition = (data.df['type'] == type_.value)
            self.assertTrue(
                data.df[type_condition].equals(getter(types=type_).df),
                "Single type contain."
            )

            types = [type_, type2_]

            types_condition = (
                data.df['type'].isin([type_loop.value for type_loop in types]))
            self.assertTrue(
                data.df[types_condition].equals(getter(types=types).df),
                "Multiple types contain."
            )

            self.assertIsNone(
                getter(types=RecordType.BIOLUMINESCENCE),
                "Not none empty getter"
            )

            if total_mask is None:
                total_mask = type_condition
            else:
                total_mask = total_mask & type_condition

        if total_mask is not None:
            # Make Sure filters make sense
            self.assertTrue(
                data.df[total_mask].equals(
                    getter(types=type_, record_ids=record_id, interval=interval).df
                ),
                "Multiple filters condition."
            )
            self.assertIsNone(
                getter(types=type_, record_ids=0, interval=interval),
                "No element left"
            )

    def _test_del_filtered(
            self,
            name: str,
            data: DataFrameFacade_
    ):
        (getter, setter, deleter) = self._get_storage_callables_by_name(name)

        interval = None

        total_mask: Optional[pd.Series] = None

        if 'time' in data.df.columns:
            setter(data, False)
            interval = Interval(start=0.0, end=10.)

            time_condition = (data.df['time'] >= interval.start) & \
                             (data.df['time'] < interval.end)
            deleter(interval=interval)
            self.assertTrue(
                data.df[~time_condition].equals(getter().df),
                "Only non deleted left."
            )
            self.assertIsNone(
                getter(interval=interval),
                "None for deleted."
            )

            if total_mask is None:
                total_mask = time_condition
            else:
                total_mask = total_mask & time_condition

        record_id = None

        if 'record_id' in data.df.columns:
            record_id = 1
            setter(data, False)
            deleter(record_ids=record_id)

            record_id_condition = (data.df['record_id'] == record_id)
            self.assertTrue(
                data.df[~record_id_condition].equals(getter().df),
                "Only non deleted left."
            )
            self.assertIsNone(
                getter(record_ids=record_id),
                "None for deleted."
            )

            record_ids = [0, 1]
            setter(data, False)
            deleter(record_ids=record_ids)

            record_ids_condition = (data.df['record_id'].isin(record_ids))
            self.assertTrue(
                data.df[~record_ids_condition].equals(getter().df),
                "Only non deleted left."
            )
            self.assertIsNone(
                getter(record_ids=record_ids),
                "None for deleted."
            )

            if total_mask is None:
                total_mask = record_id_condition
            else:
                total_mask = total_mask & record_id_condition

        type_ = None

        if 'type' in data.df.columns:
            type_ = RecordType.CASCADE
            type2_ = RecordType.ELECTRICAL
            setter(data, False)
            deleter(types=type_)

            type_condition = (data.df['type'] == type_.value)
            self.assertTrue(
                data.df[~type_condition].equals(getter().df),
                "Only non deleted left."
            )
            self.assertIsNone(
                getter(types=type_),
                "None for deleted."
            )

            types = [type_, type2_]
            setter(data, False)
            deleter(types=types)

            types_condition = (
                data.df['type'].isin([type_loop.value for type_loop in types]))
            self.assertTrue(
                data.df[~types_condition].equals(getter().df),
                "Only non deleted left."
            )
            self.assertIsNone(
                getter(types=types),
                "None for deleted."
            )

            if total_mask is None:
                total_mask = type_condition
            else:
                total_mask = total_mask & type_condition

        if total_mask is not None:
            # Make Sure filters make sense
            setter(data, False)
            deleter(types=type_, record_ids=record_id, interval=interval)
            self.assertTrue(
                data.df[~total_mask].equals(getter().df),
                "Only non deleted left."
            )
            self.assertIsNone(
                getter(types=type_, record_ids=record_id, interval=interval),
                "None for deleted."
            )

    def _test_filterable(self, data: DataFrameFacade_, name: str):
        self._test_get_set_append_and_delete(
            data=data,
            name=name
        )
        self._test_get_filtered(
            data=data,
            name=name
        )
        self._test_del_filtered(
            data=data,
            name=name
        )

    def test_detector(self) -> None:
        detector = get_detector()
        self._test_get_set_append_and_delete(
            data=detector,
            name='detector'
        )

    def test_records(self) -> None:
        records = get_records()
        self._test_filterable(
            data=records,
            name='records'
        )

    def test_hits(self) -> None:
        hits = get_hits()
        self._test_filterable(
            data=hits,
            name='hits'
        )

    def test_sources(self) -> None:
        sources = get_sources()
        self._test_filterable(
            data=sources,
            name='sources'
        )

    def test_record_ids_with_sources(self) -> None:
        sources = get_sources()
        records = get_records()
        self.assertTrue(
            self.storage.get_record_ids_with_sources().empty,
            'Empty series if empty.'
        )
        self.storage.set_sources(sources)
        self.storage.set_records(records)
        self.assertTrue(
            pd.Series([0, 1, 2]).equals(
                self.storage.get_record_ids_with_sources().sort_values(ignore_index=True)
            ),
            'All without filters.'
        )
        self.assertTrue(
            pd.Series([1]).equals(
                self.storage.get_record_ids_with_sources(record_ids=1).reset_index(drop=True)
            ),
            'All with filters.'
        )

    def test_record_ids_with_hits(self) -> None:
        hits = get_hits()
        records = get_records()
        self.assertTrue(
            self.storage.get_record_ids_with_hits().empty,
            'Empty series if empty.'
        )
        self.storage.set_hits(hits)
        self.storage.set_records(records)
        self.assertTrue(
            pd.Series([0, 1, 2]).equals(
                self.storage.get_record_ids_with_hits().sort_values(ignore_index=True)
            ),
            'All with filters.'
        )
        self.assertTrue(
            pd.Series([1]).equals(
                self.storage.get_record_ids_with_hits(record_ids=1).reset_index(drop=True)
            ),
            'All with filters.'
        )


class HDFStorageTestCase(StorageTests, unittest.TestCase):
    tmp_collection_path: str = '_tmp_hdfstoragetest_collection.h5'

    def _get_configuration(self) -> HDF5StorageConfiguration:
        return HDF5StorageConfiguration(
            data_path=self.tmp_collection_path
        )

    def _get_storage_class(self) -> Type[HDF5CollectionStorage]:
        return HDF5CollectionStorage

    def test_construction(self) -> None:
        configuration = HDF5StorageConfiguration(data_path=self.tmp_collection_path)
        storage = HDF5CollectionStorage(configuration)

        self.assertEqual(
            self.tmp_collection_path,
            storage.data_path,
            "Seeting the data path is broken"
        )

        self.assertEqual(
            configuration,
            storage.configuration,
            "Setting the configuration is broken"
        )

        self.assertIsNone(
            storage._HDF5CollectionStorage__store,
            "Storage not initialized"
        )

    def test_get_where_record_id(self):
        record_ids = 1
        self.assertEqual(
            '((record_id={}))'.format(record_ids),
            self.storage._HDF5CollectionStorage__get_where(record_ids=record_ids),
            "Where for single record id broken."
        )

        multiple_record_ids = [1, 2]
        multiple_record_ids_expected = '((record_id=1 | record_id=2))'
        self.assertEqual(
            multiple_record_ids_expected,
            self.storage._HDF5CollectionStorage__get_where(
                record_ids=multiple_record_ids
            ),
            "Where for multiple record ids list broken."
        )
        self.assertEqual(
            multiple_record_ids_expected,
            self.storage._HDF5CollectionStorage__get_where(
                record_ids=pd.Series(multiple_record_ids)
            ),
            "Where for multiple record ids pd.Series broken."
        )

    def test_get_where_type(self):
        type = RecordType.CASCADE
        self.assertEqual(
            '((type={}))'.format(type.value),
            self.storage._HDF5CollectionStorage__get_where(types=type),
            "Where for single type broken."
        )

        multiple_types = [RecordType.CASCADE, RecordType.ELECTRICAL]
        multiple_types_expected = '((type={} | type={}))'.format(
            multiple_types[0].value,
            multiple_types[1].value
        )
        self.assertEqual(
            multiple_types_expected,
            self.storage._HDF5CollectionStorage__get_where(
                types=multiple_types
            ),
            "Where for multiple types broken."
        )

    def test_get_where_interval(self):
        interval = Interval(start=0, end=1000)
        self.assertEqual(
            '((time>={} & time<{}))'.format(interval.start, interval.end),
            self.storage._HDF5CollectionStorage__get_where(interval=interval),
            "Where for interval broken."
        )

    def test_get_where_combined(self):
        self.assertIsNone(
            self.storage._HDF5CollectionStorage__get_where(),
            "Empty parameter where is not None."
        )

        interval = Interval(start=0, end=1000)
        record_ids = 1
        multiple_types = [RecordType.CASCADE, RecordType.ELECTRICAL]
        self.assertEqual(
            '((type={} | type={}) & (record_id=1) & (time>={} & time<{}))'.format(
                multiple_types[0].value,
                multiple_types[1].value,
                interval.start,
                interval.end
            ),
            self.storage._HDF5CollectionStorage__get_where(
                record_ids=record_ids,
                types=multiple_types,
                interval=interval
            ),
            "Where for multiple parameters broken."
        )

    def test_get_unique_record_ids(self):
        unique_ids = self.storage._HDF5CollectionStorage__get_unique_records_ids(
            key=HDF5StorageKeys.HITS
        )
        self.assertEqual(
            0,
            len(unique_ids),
            "Unique record id for empty collection broken"
        )
        self.storage.set_hits(
            Hits(
                df=pd.DataFrame(
                    {
                        'type': RecordType.CASCADE.value,
                        'record_id': [45, 2, 45],
                        'pmt_id': 1,
                        'string_id': 1,
                        'module_id': [1, 2, 3],
                        'time': [1, 100, 1000]
                    }
                )
            )
        )
        unique_ids = self.storage._HDF5CollectionStorage__get_unique_records_ids(
            key=HDF5StorageKeys.HITS
        )
        self.assertListEqual(
            [45, 2],
            list(unique_ids.values),
            "Unique record id for not collection broken"
        )

    def test_next_record_ids(self):
        with mock.patch.object(
                self.storage,
                '_HDF5CollectionStorage__get_unique_records_ids',
                return_value=pd.Series([])
        ):
            self.assertEqual(
                [0],
                self.storage.get_next_record_ids(1),
                "Next record ids single id broken"
            )

            self.assertEqual(
                [0, 1],
                self.storage.get_next_record_ids(2),
                "Next record ids multiple broken"
            )
        with mock.patch.object(
                self.storage,
                '_HDF5CollectionStorage__get_unique_records_ids',
                return_value=pd.Series([1, 45, 23])
        ):
            self.assertEqual(
                [46],
                self.storage.get_next_record_ids(1),
                "Next record ids single id broken with offset"
            )

            self.assertEqual(
                [46, 47],
                self.storage.get_next_record_ids(2),
                "Next record ids multiple broken with offset"
            )

class StorageFactoryTestCase(unittest.TestCase):
    def test_no_supported_storage_configuration(self):
        class NoStorageConfiguration(StorageConfiguration):
            pass

        with self.assertRaises(
            ValueError,
            msg='No supported configuration wrong'
        ):
            StorageFactory.create(NoStorageConfiguration())

    def test_hdf5_returned(self):
        configuration = HDF5StorageConfiguration(
            data_path='_tmp_storage_factory_test.hf'
        )
        collection_storage = StorageFactory.create(configuration)
        self.assertIsInstance(collection_storage, HDF5CollectionStorage)


if __name__ == '__main__':
    unittest.main()
