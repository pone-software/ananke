"""Module containing all the storage interfaces for collections."""

import logging
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from subprocess import call
from typing import List, Optional, TypeVar, Union, Generic, Type, Dict

import numpy as np
import pandas as pd
import pandera as pa

from ananke.configurations.collection import (
    StorageConfiguration,
    HDF5StorageConfiguration,
)
from ananke.configurations.events import Interval
from ananke.models.detector import Detector
from ananke.models.event import (
    Hits,
    Records,
    Sources,
    RecordIds,
)
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import FullRecordSchema, TypesTypes_, RecordIdsTypes_

CollectionStorageConfiguration_ = TypeVar(
    "CollectionStorageConfiguration_",
    bound=StorageConfiguration
)
DataFrameFacade_ = TypeVar("DataFrameFacade_", bound=DataFrameFacade)


# TODO: Implement Configuration Saving
# TODO: Implement Deletion


class AbstractCollectionStorage(ABC, Generic[CollectionStorageConfiguration_]):
    """Interface for all Collection Storage interfaces."""

    def __init__(self, configuration: CollectionStorageConfiguration_):
        """Constructor of the Collection Storage.

        Args:
            configuration: Configuration of the Collection Storage Interface
        """
        self.configuration = configuration
        self._read_only = self.configuration.read_only
        self.logger = logging.getLogger(type(self).__name__)

    def __enter__(self):
        """Open storage connection on enter."""
        self.open()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Close storage connection on exit"""
        self.close()

    @abstractmethod
    def open(self) -> None:
        """Opens storage connection."""
        raise NotImplementedError('Opening Connection not possible.')

    @abstractmethod
    def close(self) -> None:
        """Opens storage connection."""
        raise NotImplementedError('Closing Connection not possible.')

    @abstractmethod
    def get_detector(self) -> Optional[Detector]:
        """Gets the detector of the current collection.

        Returns:
            Detector of collection or None if empty.
        """
        raise NotImplementedError('Get detector is not implemented')

    @abstractmethod
    def set_detector(
            self,
            detector: Detector,
            append: bool = False
    ) -> None:
        """Sets the detector for current storage

        Args:
            detector: Detector to set
            append: Append to existing detector
        """
        raise NotImplementedError('Set detector is not implemented')

    @abstractmethod
    def del_detector(self) -> None:
        """Deletes detector of collection storage."""
        raise NotImplementedError('Delete detector is not implemented')

    @abstractmethod
    def get_records(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> Optional[Records]:
        """Gets the records of the current collection.

        Args:
            types: Type of records to get
            record_ids: record_ids to pick
            interval: only records in interval

        Returns:
            Records of collection or None if empty.
        """
        raise NotImplementedError('Get records is not implemented')

    @abstractmethod
    def set_records(
            self,
            records: Records,
            append: bool = True,
    ) -> None:
        """Sets the records for current storage

        Args:
            records: Records to set
            append: Append to existing detector
        """
        raise NotImplementedError('Set records is not implemented')

    @abstractmethod
    def del_records(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> None:
        """Deletes records of collection storage.

        Args:
            types: Type of records to get
            record_ids: record_ids to pick
            interval: only records in interval
        """
        raise NotImplementedError('Delete records is not implemented')

    @abstractmethod
    def get_hits(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> Optional[Hits]:
        """Gets the hits of the current collection.

        Args:
            types: Type of hits to get
            record_ids: record_ids to pick
            interval: only hits in interval

        Returns:
            Hits of collection or None if empty.
        """
        raise NotImplementedError('Get hits is not implemented')

    @abstractmethod
    def set_hits(
            self,
            hits: Hits,
            append: bool = True,
    ) -> None:
        """Sets the hits for current storage

        Args:
            hits: Hits to set
            append: Append to existing detector
        """
        raise NotImplementedError('Set hits is not implemented')

    @abstractmethod
    def del_hits(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> None:
        """Deletes hits of collection storage.

        Args:
            types: Type of hits to get
            record_ids: record_ids to pick
            interval: only hits in interval
        """
        raise NotImplementedError('Delete hits is not implemented')

    @abstractmethod
    def get_sources(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> Optional[Sources]:
        """Gets the sources of the current collection.

        Args:
            types: Type of sources to get
            record_ids: record_ids to pick
            interval: only sources in interval

        Returns:
            Sources of collection or None if empty.
        """
        raise NotImplementedError('Get sources is not implemented')

    @abstractmethod
    def set_sources(
            self,
            sources: Sources,
            append: bool = True,
    ) -> None:
        """Sets the sources for current storage

        Args:
            sources: Sources to set
            append: Append to existing detector
        """
        raise NotImplementedError('Set sources is not implemented')

    @abstractmethod
    def del_sources(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> None:
        """Deletes sources of collection storage.

        Args:
            types: Type of sources to get
            record_ids: record_ids to pick
            interval: only sources in interval
        """
        raise NotImplementedError('Delete sources is not implemented')

    @abstractmethod
    def get_next_record_ids(self, n: int = 1) -> List[int]:
        """Gets n next record ids for generation.

        Args:
            n: Number of record ids to generate

        Returns:
            List of integers
        """
        raise NotImplementedError('Get next record ids not implemented')

    @abstractmethod
    def delete(self) -> None:
        """Deletes current collection."""
        raise NotImplementedError('Delete collection is not implemented')

    def get_record_ids_with_hits(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> pd.Series:
        """Gets all record ids with hits.

        Args:
            types: Type of sources to get
            record_ids: record_ids to pick
            interval: only sources in interval

        Returns:
            Pandas Series containing all record ids
        """
        raise NotImplementedError('Get record ids with hits is not implemented')

    def get_record_ids_with_sources(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> pd.Series:
        """Gets all record ids with hits.

        Args:
            types: Type of sources to get
            record_ids: record_ids to pick
            interval: only sources in interval

        Returns:
            Pandas Series containing all record ids
        """
        raise NotImplementedError('Get record ids with sources is not implemented')

    def optimize(self):
        """Optimizes storage, if optimizable."""
        pass


class HDF5StorageKeys(Enum):
    """Enum containing keys for the collection file."""

    RECORDS = "records"
    HITS = "hits"
    SOURCES = "sources"
    DETECTOR = "detector"

    def __str__(self):
        return str(self.value)


class HDF5CollectionStorage(AbstractCollectionStorage[HDF5StorageConfiguration]):
    """Implementation for collection storage using HDF5 file format."""

    def __init__(self, configuration: HDF5StorageConfiguration):
        """Constructor of HDF5 collection storage."""
        super(HDF5CollectionStorage, self).__init__(configuration=configuration)
        self.__store = None
        self.data_path = self.configuration.data_path

    @property
    def store(self) -> pd.HDFStore:
        if self.__store is None or not self.__store.is_open:
            raise ValueError('You forgot to open the storage')

        return self.__store

    def open(self) -> None:
        if self.__store is None:
            self.__store = self.__get_store()
        elif not self.__store.is_open:
            self.__store.open()

    def close(self) -> None:
        self.__store.close()

    def delete(self) -> None:
        self.close()
        if os.path.isfile(self.data_path):
            os.remove(self.data_path)

    def get_detector(self) -> Optional[Detector]:
        return self.__get_data(
            key=HDF5StorageKeys.DETECTOR,
            facade_class=Detector
        )

    def set_detector(
            self,
            detector: Detector,
            append: bool = True,
    ) -> None:
        return self.__set_data(
            key=HDF5StorageKeys.DETECTOR,
            data=detector,
            append=append
        )

    def del_detector(self) -> None:
        return self.__del_data(
            key=HDF5StorageKeys.DETECTOR,
        )

    def get_records(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> Optional[Records]:
        return self.__get_data(
            key=HDF5StorageKeys.RECORDS,
            facade_class=Records,
            types=types,
            record_ids=record_ids,
            interval=interval
        )

    def set_records(
            self,
            records: Records,
            append: bool = True
    ) -> None:
        return self.__set_data(
            key=HDF5StorageKeys.RECORDS,
            data=records,
            append=append,
            schema=FullRecordSchema
        )

    def del_records(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> None:
        return self.__del_data(
            key=HDF5StorageKeys.RECORDS,
            types=types,
            record_ids=record_ids,
            interval=interval
        )

    def get_hits(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> Optional[Hits]:
        return self.__get_data(
            key=HDF5StorageKeys.HITS,
            facade_class=Hits,
            types=types,
            record_ids=record_ids,
            interval=interval
        )

    def set_hits(
            self,
            hits: Hits,
            append: bool = True
    ) -> None:
        return self.__set_data(
            key=HDF5StorageKeys.HITS,
            data=hits,
            append=append
        )

    def del_hits(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> None:
        return self.__del_data(
            key=HDF5StorageKeys.HITS,
            types=types,
            record_ids=record_ids,
            interval=interval
        )

    def get_sources(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> Optional[Sources]:
        return self.__get_data(
            key=HDF5StorageKeys.SOURCES,
            facade_class=Sources,
            types=types,
            record_ids=record_ids,
            interval=interval
        )

    def set_sources(
            self,
            sources: Sources,
            append: bool = True
    ) -> None:
        return self.__set_data(
            key=HDF5StorageKeys.SOURCES,
            data=sources,
            append=append,
        )

    def del_sources(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> None:
        return self.__del_data(
            key=HDF5StorageKeys.SOURCES,
            types=types,
            record_ids=record_ids,
            interval=interval
        )

    def __get_store(self) -> pd.HDFStore:
        """Gets the stored hdf object.

        Returns:
            stored hdf object.
        """
        mode = 'a'
        if self._read_only:
            # TODO Make bulletproof on update of read only
            mode = 'r'

        data_path = self.data_path

        if not os.path.isfile(data_path):
            dir_path = os.path.dirname(data_path)
            if len(dir_path) > 0:
                os.makedirs(os.path.dirname(data_path), exist_ok=True)

        return pd.HDFStore(
            self.data_path,
            mode=mode,
            complevel=self.configuration.complevel,
            complib=self.configuration.complib
        )

    def __get_data(
            self,
            key: HDF5StorageKeys,
            facade_class: Optional[Type[DataFrameFacade_]] = None,
            types: TypesTypes_ = None,
            record_ids: Optional[
                Union[int, List[int], pd.Series]
            ] = None,
            interval: Optional[Interval] = None,
            columns: Optional[List[str]] = None,
    ) -> Optional[DataFrameFacade_]:
        """Gets data frame facade from file.

        Args:
            key: Key to get data by
            facade_class: Data frame facade class to instantiate
            types: Record types to filter by
            record_ids: Record ids to filter by
            interval: timeframe to filter by

        Returns:
            Data frame facade containing data or None
        """
        logging.debug(
            'Get {} with key {} at \'{}\''.format(facade_class, key, self.data_path)
        )

        str_key = str(key)

        where = self.__get_where(
            types=types,
            record_ids=record_ids,
            interval=interval
        )

        try:
            store = self.store
            df = store.select(key=str_key, where=where, columns=columns)
            if df.empty:
                return None
            if facade_class is not None:
                facade = facade_class(df=df)
                return facade

            return df
        except KeyError:
            return None

    def __del_data(
            self,
            key: HDF5StorageKeys,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> None:
        """Delete rows that fit the criteria. If no criteria is passed, dataframe is deleted.

        Args:
            key: Key to delete data by
            types: Record types to filter by
            record_ids: Record ids to filter by
            interval: Interval to filter by
        """
        self.__raise_writable()

        str_key = str(key)

        where = self.__get_where(
            types=types,
            record_ids=record_ids,
            interval=interval
        )

        self.store.remove(key=str_key, where=where)

    def __raise_writable(self) -> None:
        """Raises exception if not writable.

        Raises:
              ValueError: When class not writable
        """
        if self._read_only:
            raise PermissionError('Class cannot write as its opened in read only mode.')

    def __set_data(
            self,
            key: HDF5StorageKeys,
            data: DataFrameFacade,
            append: bool = True,
            schema: Optional[Type[pa.SchemaModel]] = None
    ) -> None:
        """Sets data frame facade to file by a specific key.

        Args:
            key: Key to store data at
            data: Data frame facade to store
            append: Append data to existing
            schema: ensures all columns exist for later appending
        """
        self.__raise_writable()

        str_key = str(key.value)

        if schema is not None:
            columns = schema.to_schema().columns
            for column in columns:
                if column not in data.df.columns:
                    data.df[column] = np.nan
                    data.df[column] = data.df[column].astype(float)

        logging.debug(
            'Set {} with key {} at \'{}\''.format(type(data), key, self.data_path)
        )

        # TODO: Test override

        self.store.put(
            key=str_key,
            value=data.df,
            format="t",
            append=append,
            complevel=self.configuration.complevel,
            complib=self.configuration.complib,
            data_columns=True,
            index=False,
        )

        # TODO: Make smart and encourage once per append
        if issubclass(type(data), RecordIds):
            self.store.create_table_index(
                key=str_key,
                columns=['record_id'],
                optlevel=self.configuration.optlevel
            )

    def __get_where(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> str:
        """Gets the where clause to select records by.

        Args:
            types: Types to select
            record_ids: record ids to select
            interval: Interval to filter by

        Returns:
            string to be passed to hdf
        """
        wheres = []

        if types is not None:
            if not issubclass(type(types), list):
                types = [types]
            types_wheres = ['type={}'.format(current_type.value) for current_type
                            in types]
            wheres.append('({})'.format(' | '.join(types_wheres)))

        if record_ids is not None:
            if isinstance(record_ids, int):
                record_ids = [record_ids]
            if isinstance(record_ids, pd.Series):
                record_ids = record_ids.drop_duplicates().values

            record_ids_wheres = ['record_id={}'.format(record_id) for record_id in
                                 record_ids]
            wheres.append('({})'.format(' | '.join(record_ids_wheres)))

        if interval is not None:
            wheres.append('(time>={} & time<{})'.format(interval.start, interval.end))

        if len(wheres) == 0:
            where = None
        else:
            where = '({})'.format(' & '.join(wheres))

        return where

    def __get_unique_records_ids(
            self,
            key: HDF5StorageKeys,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> pd.Series:
        """Gets all unique record ids or empty series if empty.

        Args:
            key: Key to get data by
            types: Record types to filter by
            record_ids: Record ids to filter by
            interval: timeframe to filter by

        Returns:
            Unique record ids or empty series
        """
        record_ids = self.__get_data(
            key=key,
            types=types,
            record_ids=record_ids,
            interval=interval,
            columns=['record_id']
        )
        if record_ids is None:
            record_ids_series = pd.Series([], dtype='int64')
        else:
            record_ids_series = record_ids['record_id'].drop_duplicates()
        return record_ids_series

    def get_next_record_ids(self, n: int = 1) -> List[int]:
        """Generates an auto increment based of the highest current record id.

        Args:
            n: Number of record ids to be generated

        Returns:
            List of integer ids
        """
        record_ids = self.__get_unique_records_ids(HDF5StorageKeys.RECORDS)
        if len(record_ids) == 0:
            start = 0
        else:
            start = record_ids.max() + 1
        return list(range(start, start + n))

    def get_record_ids_with_hits(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> pd.Series:
        return self.__get_unique_records_ids(
            HDF5StorageKeys.HITS,
            types=types,
            record_ids=record_ids,
            interval=interval
        )

    def get_record_ids_with_sources(
            self,
            types: TypesTypes_ = None,
            record_ids: RecordIdsTypes_ = None,
            interval: Optional[Interval] = None
    ) -> pd.Series:
        return self.__get_unique_records_ids(
            HDF5StorageKeys.SOURCES,
            types=types,
            record_ids=record_ids,
            interval=interval
        )

    def optimize(self):
        """Optimize HDF5 Storage by recompressing and recreating indices."""
        dir = os.path.dirname(self.data_path)
        self.logger.info("Starting to optimize HDF5.")
        tmp_file = os.path.join(dir, '{}.h5'.format(str(uuid.uuid4())))
        indices_to_create_index = [
            HDF5StorageKeys.HITS,
            HDF5StorageKeys.SOURCES,
            HDF5StorageKeys.RECORDS
        ]
        self.logger.debug("Starting to recreate indices.")
        for index in indices_to_create_index:
            self.store.create_table_index(
                key=str(index),
                columns=['record_id'],
                optlevel=self.configuration.optlevel
            )
        self.logger.debug("Finished recreating indices.")
        self.close()
        try:
            self.logger.debug("Starting to ptrepack files.")
            command = [
                "ptrepack",
                "-o",
                "--chunkshape=auto",
                "--propindexes",
                "--complevel={}".format(self.configuration.complevel),
                "--complib={}".format(self.configuration.complib),
                self.data_path,
                tmp_file
            ]
            call(command)
            os.remove(self.data_path)
            shutil.move(tmp_file, self.data_path)
            self.logger.debug("Finished to ptrepack files.")
        except:
            logging.warning('PTRepack not working. Skipping compression')
        self.store.open()
        self.logger.info("Finished to optimize HDF5.")


class StorageFactory:
    """Class as factory for collection storages"""

    #: Mapping from configuration to actual storage class
    configuration_mapping: Dict[
        Type[StorageConfiguration], Type[AbstractCollectionStorage]] = {
        HDF5StorageConfiguration: HDF5CollectionStorage
    }

    @classmethod
    def create(cls, configuration: StorageConfiguration) -> AbstractCollectionStorage:
        """Takes a configuration for a storage and returns according storage class.

        Args:
            configuration: Configuration for current storage

        Returns:
            Collection storage to get and retrieve data from
        """
        for configuration_type in cls.configuration_mapping.keys():
            if isinstance(configuration, configuration_type):
                return cls.configuration_mapping[configuration_type](configuration)

        raise ValueError('Configuration type \'{}\' is not supported')
