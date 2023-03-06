"""Module containing all logic to import collections."""
from __future__ import annotations

import logging

from typing import TYPE_CHECKING, List, Type, TypeVar

import pandas as pd

from ananke.configurations.events import Interval
from ananke.models.detector import Detector
from ananke.models.event import Hits, Records, Sources
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import RecordType, SourceType, Types
from tqdm import tqdm


if TYPE_CHECKING:
    from ananke.models.collection import Collection

import os

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Union


DataFrameFacade_ = TypeVar("DataFrameFacade_", bound=DataFrameFacade)


class AbstractCollectionImporter(ABC):
    """Abstract parent class for collection importers."""

    def __init__(self, collection: Collection):
        """Constructor of the Abstract Collection Importer.

        Args:
            collection: Path to collection or collection
        """
        self.collection = collection
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def import_data(self, **kwargs) -> None:
        """Abstract stub for the import of a collection.

        Args:
            kwargs: Additional importer args

        Returns:
            Imported collection
        """
        pass


# class JuliaArrowCollectionImporter(AbstractCollectionImporter):
#     """Concrete implementation for Julia Arrow imports."""
#
#     def __read_file(self, filename: Union[str, bytes, os.PathLike]):
#
#         with pa.ipc.open_file(filename) as reader:
#             df = reader.read_pandas()
#             print(df.dtypes)
#
#     def import_data(
#             self,
#             import_path: Union[str, bytes, os.PathLike],
#             **kwargs: object
#     ) -> None:
#         """Import of a julia arrow collection.
#
#         Args:
#             import_path: File path to import
#             **kwargs: Additional importer args
#         """
#         directory = os.fsencode(import_path)
#         for file in os.listdir(directory):
#             filename = os.path.join(import_path, os.fsdecode(file))
#             if filename.endswith(".arrow"):
#                 self.__read_file(filename)


class LegacyCollectionKeys(str, Enum):
    """Enum containing keys for the collection file."""

    RECORDS = "records"
    HITS = "hits"
    SOURCES = "sources"
    DETECTOR = "detector"


class LegacyCollection:
    """Class combining all data frames to a complete record collection."""

    def __init__(
        self,
        data_path: str,
        complevel: int = 3,
        complib: str = "lzo",
        read_only: bool = False,
    ):
        """Constructor for the collection.

        Args:
            data_path: Path to store/read collection from
            records: Optional records to store directly
            detector: Optional detector to store directly
            complevel: Compression level for data file
            complib: Compression Algorithm to use
            read_only: Ensure that no data is written
        """
        logging.debug("Instantiated collection with path {}".format(data_path))
        file_extensions = (".hdf", ".h5")
        if not str(data_path).lower().endswith(file_extensions):
            raise ValueError(
                "Only {} and {} are supported file extensions".format(
                    file_extensions[0], file_extensions[1]
                )
            )

        self.data_path = data_path
        self.complevel = complevel
        self.complib = complib
        self.read_only = read_only

        self.store: pd.HDFStore = self.__get_store()

    def __get_store(self) -> pd.HDFStore:
        """Gets the stored hdf object.

        Returns:
            stored hdf object.
        """
        mode = "a"
        if self.read_only:
            mode = "r"
        return pd.HDFStore(self.data_path, mode=mode)

    def __del__(self):
        """Close store on delete."""
        self.store.close()

    def __get_hdf_path(
        self, collection_key: LegacyCollectionKeys, record_id: int | str | pd.Series
    ) -> str:
        """Gets a proper hdf path for all subgrouped datasets.

        Args:
            collection_key: collection key to start with
            record_id: record id of the record to save

        Returns:
            string combining the two elements
        """
        if not (
            collection_key == LegacyCollectionKeys.HITS
            or collection_key == LegacyCollectionKeys.SOURCES
        ):
            raise ValueError("Paths only possible for hits and sources.")
        if type(record_id) == pd.Series:
            record_id = record_id.astype(str)
        elif type(record_id) != str:
            record_id = str(record_id)
        return "/{key}/".format(key=collection_key.value) + record_id

    def __get_data(
        self,
        key: str,
        facade_class: Type[DataFrameFacade_],
        where: Optional[list] = None,
    ) -> Optional[DataFrameFacade_]:
        """Gets data frame facade from file.

        Args:
            key: Key to get data by
            facade_class: Data frame facade class to instantiate

        Returns:
            Data frame facade containing data or None
        """
        logging.debug(
            "Get {} with key {} at '{}'".format(facade_class, key, self.data_path)
        )

        try:
            store = self.store
            df = pd.DataFrame(store.select(key=key, where=where))
            if "type" in df.columns:
                new_dict = {}
                for type in RecordType:
                    new_dict[type.name.lower()] = type.value
                for type in SourceType:
                    new_dict[type.name.lower()] = type.value

                df["type"] = df["type"].map(new_dict)
            facade = facade_class(df=df)
            return facade
        except KeyError:
            return None

    def get_detector(self) -> Optional[Detector]:
        """Gets detector of the collection.

        Returns:
            Detector of the collection.
        """
        return self.__get_data(key=LegacyCollectionKeys.DETECTOR, facade_class=Detector)

    def get_records(
        self, record_type: Optional[Union[List[Types], Types]] = None
    ) -> Optional[Records]:
        """Gets records of the collection.

        Args:
            record_type: record type to include

        Returns:
            Records of the collection.
        """
        where = None
        if record_type is not None:
            if type(record_type) is not list:
                record_type = [record_type]

            wheres = ["type={}".format(current_type) for current_type in record_type]
            where = "({})".format(" & ".join(wheres))
        return self.__get_data(
            key=LegacyCollectionKeys.RECORDS, facade_class=Records, where=where
        )

    def __get_subgroup_dataset(
        self,
        base_key: LegacyCollectionKeys,
        facade_class: Type[DataFrameFacade_],
        record_id: int | str,
        interval: Optional[Interval] = None,
    ) -> Optional[DataFrameFacade_]:
        """Gets data of group based on record id.

        Args:
            base_key: key of the subgroup
            facade_class: Data frame facade class to instantiate
            record_id: Record id to get hits from

        Returns:
            Data frame facade with the data from disc.
        """
        key = self.__get_hdf_path(base_key, record_id)
        # TODO: Check right place
        where = None
        if interval is not None:
            where = "(time < {end_time} & time >= {start_time})".format(
                end_time=interval.end, start_time=interval.start
            )
        return self.__get_data(key=key, facade_class=facade_class, where=where)

    # TODO: Allow multiple or all
    def get_sources(
        self, record_id: int | str, interval: Optional[Interval] = None
    ) -> Optional[Sources]:
        """Gets sources by a specific record id.

        Args:
            record_id: Record id to get hits from
            interval: Interval to get hits in

        Returns:
            None if not available or sources for record id
        """
        return self.__get_subgroup_dataset(
            base_key=LegacyCollectionKeys.SOURCES,
            facade_class=Sources,
            record_id=record_id,
            interval=interval,
        )

    # TODO: Allow multiple or all
    def get_hits(
        self, record_id: int | str, interval: Optional[Interval] = None
    ) -> Optional[Hits]:
        """Gets hits by a specific record id.

        Args:
            record_id: Record id to get hits from
            interval: Interval to get hits in

        Returns:
            None if not available or hits for record id
        """
        return self.__get_subgroup_dataset(
            base_key=LegacyCollectionKeys.HITS,
            facade_class=Hits,
            record_id=record_id,
            interval=interval,
        )


class LegacyCollectionImporter(AbstractCollectionImporter):
    """Class to import legacy collection to current one."""

    def import_data(
        self, import_path: Union[str, bytes, os.PathLike], **kwargs
    ) -> None:
        """Imports data from legacy collection.

        Args:
            import_path: Path to import data from
            **kwargs: additional args for data import.
        """
        self.logger.info(
            "Starting Legacy import from path '{}'".format(str(import_path))
        )
        legacy_collection = LegacyCollection(import_path)
        legacy_detector = legacy_collection.get_detector()
        legacy_records = legacy_collection.get_records()
        if legacy_records is None:
            raise ValueError("No records to import")
        number_of_records = len(legacy_records)
        with self.collection:
            if legacy_detector is not None:
                self.collection.storage.set_detector(legacy_detector)
            new_ids = self.collection.storage.get_next_record_ids(number_of_records)
            legacy_records_ids = legacy_records.record_ids
            legacy_records.df["record_id"] = new_ids
            if legacy_records is not None:
                self.collection.storage.set_records(legacy_records)

                with tqdm(total=len(legacy_records), mininterval=0.5) as pbar:
                    for index, legacy_record_id in enumerate(legacy_records_ids):
                        new_id = new_ids[index]
                        legacy_hits = legacy_collection.get_hits(
                            record_id=legacy_record_id
                        )
                        legacy_sources = legacy_collection.get_sources(
                            record_id=legacy_record_id
                        )
                        if legacy_hits is not None:
                            legacy_hits.df["record_id"] = new_id
                            self.collection.storage.set_hits(legacy_hits)
                        if legacy_sources is not None:
                            legacy_sources.df["record_id"] = new_id
                            self.collection.storage.set_sources(legacy_sources)

                        pbar.update()
