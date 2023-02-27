"""Module containing a collection."""
from __future__ import annotations

import logging
import os
import shutil
import uuid
import warnings
from enum import Enum
from subprocess import call
from typing import List, Optional, Type, TypeVar, Union

import h5py
import numpy as np
import pandas as pd
from tables import NaturalNameWarning, PerformanceWarning
from tqdm import tqdm

from ananke.configurations.collection import MergeConfiguration
from ananke.configurations.events import (
    EventRedistributionMode,
    RedistributionConfiguration, Interval,
)
from ananke.models.detector import Detector
from ananke.models.event import Hits, RecordIds, Records, Sources, RecordStatistics
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import EventTypes, EventType
from ananke.services.collection.exporters import (
    CollectionExporters,
    CollectionExporterFactory,
)
from ananke.services.collection.importers import (
    CollectionImporterFactory,
    CollectionImporters,
)
from ananke.utils import get_64_bit_signed_uuid_int, save_configuration

warnings.filterwarnings("ignore", category=NaturalNameWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)


class CollectionKeys(str, Enum):
    """Enum containing keys for the collection file."""

    RECORDS = "records"
    HITS = "hits"
    SOURCES = "sources"
    DETECTOR = "detector"


DataFrameFacade_ = TypeVar("DataFrameFacade_", bound=DataFrameFacade)


# TODO: Implement smooth opening and closing of file.
class Collection:
    """Class combining all data frames to a complete record collection."""

    def __init__(
            self,
            data_path: str,
            records: Optional[Records] = None,
            detector: Optional[Detector] = None,
            complevel: int = 3,
            complib: str = 'lzo',
            override: bool = False,
            read_only: bool = False
    ):
        """Constructor for the collection.

        Args:
            data_path: Path to store/read collection from
            records: Optional records to store directly
            detector: Optional detector to store directly
            complevel: Compression level for data file
            complib: Compression Algorithm to use
            override: Override existing data
            read_only: Ensure that no data is written
        """
        logging.debug('Instantiated collection with path {}'.format(data_path))
        file_extensions = (".hdf", ".h5")
        if not str(data_path).lower().endswith(file_extensions):
            raise ValueError(
                "Only {} and {} are supported file extensions".format(
                    file_extensions[0], file_extensions[1]
                )
            )

        file_exists = os.path.isfile(data_path)

        if not file_exists:
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            hf = h5py.File(data_path, "w")
            hf.create_group(CollectionKeys.HITS.value)
            hf.create_group(CollectionKeys.SOURCES.value)
            hf.close()

        self.data_path = data_path
        self.complevel = complevel
        self.complib = complib
        self.read_only = read_only

        self.store: pd.HDFStore = self.__get_store()

        if detector is not None:
            self.set_detector(detector, override)

        if records is not None:
            self.set_records(records, override)

    def __get_store(self) -> pd.HDFStore:
        """Gets the stored hdf object.

        Returns:
            stored hdf object.
        """
        mode = 'a'
        if self.read_only:
            mode = 'r'
        return pd.HDFStore(self.data_path, mode=mode)

    def __del__(self):
        """Close store on delete."""
        self.store.close()

    def __get_hdf_path(
            self, collection_key: CollectionKeys, record_id: int | str | pd.Series
    ) -> str:
        """Gets a proper hdf path for all subgrouped datasets.

        Args:
            collection_key: collection key to start with
            record_id: record id of the record to save

        Returns:
            string combining the two elements
        """
        if not (
                collection_key == CollectionKeys.HITS
                or collection_key == CollectionKeys.SOURCES
        ):
            raise ValueError("Paths only possible for hits and sources.")
        if type(record_id) == pd.Series:
            record_id = record_id.astype(str)
        elif type(record_id) != str:
            record_id = str(record_id)
        return "/{key}/".format(
            key=collection_key.value
        ) + record_id

    def __get_data(
            self,
            key: str,
            facade_class: Type[DataFrameFacade_],
            where: Optional[list] = None
    ) -> Optional[DataFrameFacade_]:
        """Gets data frame facade from file.

        Args:
            key: Key to get data by
            facade_class: Data frame facade class to instantiate

        Returns:
            Data frame facade containing data or None
        """
        logging.debug(
            'Get {} with key {} at \'{}\''.format(facade_class, key, self.data_path)
        )

        try:
            store = self.store
            df = store.select(key=key, where=where)
            facade = facade_class(df=df)
            return facade
        except KeyError:
            return None

    def __set_data(
            self,
            key: str,
            data: DataFrameFacade,
            append: bool = False,
            override: bool = False,
    ) -> None:
        """Sets data frame facade to file by a specific key.

        Args:
            key: Key to store data at
            data: Data frame facade to store
            append: Append to existing data if necessary
            override: Override existing data yes or no.
        """
        if self.read_only:
            raise ValueError('Class cannot write as its opened in read only mode.')

        get_data = self.__get_data(
            key=key,
            facade_class=type(data)
        )
        if (
                not override
                and get_data is not None
                and not append
        ):
            raise ValueError("Cannot override data with key {}".format(key))

        # TODO: Evaluate different options
        store = self.store
        if append and get_data is not None:
            # TODO: Work out smart Appending again
            # combined_columns = pd.concat([get_data.df.dtypes, data.df.dtypes])
            # combined_columns = combined_columns[~combined_columns.index.duplicated()]
            # get_data_changed = False
            # for (current_column, current_column_type) in combined_columns.items():
            #     if current_column not in data.df.columns:
            #         data.df[current_column] = -1
            #         data.df[current_column] = data.df[current_column].astype(
            #             current_column_type
            #         )
            #
            #     if current_column not in get_data.df.columns:
            #         get_data.df[current_column] = -1
            #         get_data.df[current_column] = get_data.df[current_column].astype(
            #             current_column_type
            #         )
            #         get_data_changed = True
            # if get_data_changed:
            #     self.__set_data(key, get_data, append=False, override=True)
            data = type(data).concat([get_data, data])

        min_itemsize = None
        if 'type' in data.df.dtypes.index:
            min_itemsize = {'type': 20}
        logging.debug(
            'Set {} with key {} at \'{}\''.format(type(data), key, self.data_path)
        )

        store.put(
            key=key,
            value=data.df,
            format="t",
            min_itemsize=min_itemsize,  # TODO: Make clever
            complevel=self.complevel,
            complib=self.complib,
            data_columns=True,
            index=False,
        )

    def get_detector(self) -> Optional[Detector]:
        """Gets detector of the collection.

        Returns:
            Detector of the collection.
        """
        return self.__get_data(
            key=CollectionKeys.DETECTOR,
            facade_class=Detector
        )

    def get_records(
            self,
            record_type: Optional[
                Union[
                    List[EventTypes],
                    EventTypes
                ]
            ] = None
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

            wheres = ['type={}'.format(current_type) for current_type in record_type]
            where = '({})'.format(' & '.join(wheres))
        return self.__get_data(
            key=CollectionKeys.RECORDS,
            facade_class=Records,
            where=where
        )

    def __drop_records_without_hits_in_interval(
            self,
            records: Records,
            interval: Interval
    ) -> Records:
        """Drops all records without hits in intverval.

        Args:
            records: Records to drop without hits in interval from
            interval: Interval for interval in question

        Returns:

        """
        record_ids_without_hits_in_interval = []
        current_record_ids = records.record_ids
        for index, record_id in current_record_ids.items():
            if self.get_hits(record_id=record_id, interval=interval) is None:
                record_ids_without_hits_in_interval.append(record_id)

        records.df = records.df[
            ~current_record_ids.isin(record_ids_without_hits_in_interval)
        ]

        return records

    def __get_subgroup_dataset(
            self,
            base_key: CollectionKeys,
            facade_class: Type[DataFrameFacade_],
            record_id: int | str,
            interval: Optional[Interval] = None
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
            where = '(time < {end_time} & time >= {start_time})'.format(
                end_time=interval.end,
                start_time=interval.start
            )
        return self.__get_data(
            key=key,
            facade_class=facade_class,
            where=where
        )

    # TODO: Allow multiple or all
    def get_sources(
            self,
            record_id: int | str,
            interval: Optional[Interval] = None
    ) -> Optional[Sources]:
        """Gets sources by a specific record id.

        Args:
            record_id: Record id to get hits from
            interval: Interval to get hits in

        Returns:
            None if not available or sources for record id
        """
        return self.__get_subgroup_dataset(
            base_key=CollectionKeys.SOURCES,
            facade_class=Sources,
            record_id=record_id,
            interval=interval
        )

    # TODO: Allow multiple or all
    def get_hits(
            self,
            record_id: int | str,
            interval: Optional[Interval] = None
    ) -> Optional[Hits]:
        """Gets hits by a specific record id.

        Args:
            record_id: Record id to get hits from
            interval: Interval to get hits in

        Returns:
            None if not available or hits for record id
        """
        return self.__get_subgroup_dataset(
            base_key=CollectionKeys.HITS,
            facade_class=Hits,
            record_id=record_id,
            interval=interval
        )

    def get_new_by_record_ids(
            self,
            data_path: Union[str, bytes, os.PathLike],
            record_ids: pd.Series[int],
            record_type: Optional[
                Union[
                    List[EventTypes],
                    EventTypes
                ]
            ] = None
    ) -> Collection:
        """Gets a new collection based on a subset of the current one.

        Args:
            data_path: path for the new collection
            record_ids: record ids of the records to keep for the collection
            record_type: optional type of the records to account for

        Returns:
            Collection with only the records of the included ids.
        """
        records = self.get_records(record_type=record_type)
        new_collection = self.__class__(data_path)
        detector = self.get_detector()
        if detector is not None:
            new_collection.set_detector(detector)

        records.df = records.df[records.record_ids.isin(record_ids)]
        new_collection.set_records(records)

        with tqdm(total=record_ids.size, mininterval=0.5) as pbar:

            for index, record_id in record_ids.items():
                record_hits = self.get_hits(record_id=record_id)
                if record_hits is not None:
                    new_collection.set_hits(record_hits)
                record_sources = self.get_sources(record_id=record_id)
                if record_sources is not None:
                    new_collection.set_sources(record_sources)

                pbar.update()

        return new_collection

    def set_detector(
            self, detector: Detector, override: bool = False
    ) -> None:
        """Sets, overrides or appends detector to the existing data file.

        Args:
            detector: Detector to set
            override: Override already saved data
        """
        self.__set_data(
            CollectionKeys.DETECTOR.value, detector, override=override
        )

    def set_records(
            self,
            records: Records,
            override: bool = False,
            append: bool = False,
    ) -> None:
        """Sets, overrides or appends records to the existing data file.

        Args:
            records: Records to set
            override: Override already saved data
            append: Append data instead of override
        """
        self.__set_data(
            CollectionKeys.RECORDS.value,
            records,
            override=override,
            append=append,
        )

    def __set_subgroup_dataset(
            self,
            base_key: CollectionKeys,
            data: RecordIds,
            override: bool = False,
            append: bool = False,
    ) -> None:
        """Sets grouped datasets.

        Args:
            base_key: Key of the group
            data: Data frame with record ids
            override: Override already saved hits
            append: Append data instead of override
        """
        logging.debug('Set {} at \'{}\''.format(base_key, self.data_path))
        record_ids = data.record_ids.drop_duplicates()
        for index, record_id in record_ids.items():
            entry_by_record = data.get_by_record(record_id)
            hdf_key = self.__get_hdf_path(base_key, record_id)
            self.__set_data(
                hdf_key,
                entry_by_record,
                override=override,
                append=append
            )

    def set_hits(
            self,
            hits: Hits,
            override: bool = False,
            append: bool = False
    ) -> None:
        """Sets hits to the collection.

        Args:
            hits: Hits to set
            override: Override already saved hits
            append: Append data instead of override
        """
        self.__set_subgroup_dataset(
            base_key=CollectionKeys.HITS,
            data=hits,
            override=override,
            append=append
        )

    def set_sources(
            self,
            sources: Sources,
            override: bool = False,
            append: bool = False
    ) -> None:
        """Sets sources to the collection.

        Args:
            sources: Sources to set
            override: Override already saved sources
            append: Append data instead of override
        """
        self.__set_subgroup_dataset(
            base_key=CollectionKeys.SOURCES,
            data=sources,
            override=override,
            append=append
        )

    def append(
            self,
            collection_to_append: Collection,
            with_hits_only: bool = True,
            interval: Optional[Interval] = None
    ) -> None:
        """Concatenate multiple connections.

        Args:
            collection_to_append: Collection to append to current one.
            with_hits_only: Only take records with hits.
            interval: interval to consider for appending

        Returns:
            A single collection combining the previous ones.
        """
        logging.info(
            'Starting to append collection {} to {}.'.format(
                collection_to_append.data_path,
                self.data_path
            )
        )
        append_detector = collection_to_append.get_detector()
        own_detector = self.get_detector()

        if own_detector is not None and not append_detector.df.equals(own_detector.df):
            raise ValueError("Cannot merge two collections with different detectors.")

        if own_detector is None:
            self.set_detector(append_detector)

        keys = collection_to_append.store.keys()

        records_with_hits = collection_to_append.get_records_with_hits(keys=keys)
        if records_with_hits is not None:
            records_with_hits_record_ids = records_with_hits.record_ids.values
        else:
            records_with_hits_record_ids = []

        if with_hits_only:
            append_records = records_with_hits
        else:
            append_records = collection_to_append.get_records()

        records_with_sources = collection_to_append.get_records_with_sources(keys=keys)
        if records_with_sources is not None:
            records_with_sources_record_ids = records_with_sources.record_ids.values
        else:
            records_with_sources_record_ids = []

        if append_records is not None:
            self.set_records(append_records, append=True)

        with tqdm(total=len(append_records), mininterval=.5) as pbar:

            for (index, record_id) in append_records.record_ids.items():
                if record_id in records_with_hits_record_ids:
                    current_hits = collection_to_append.get_hits(
                        record_id=record_id,
                        interval=interval
                    )
                    self.set_hits(current_hits)

                if record_id in records_with_sources_record_ids:
                    current_sources = collection_to_append.get_sources(
                        record_id=record_id,
                        interval=interval
                    )
                    self.set_sources(current_sources)
                pbar.update()
        logging.info(
            'Finished to append collection {} to {}.'.format(
                collection_to_append.data_path,
                self.data_path
            )
        )

    def export(
            self,
            export_path: Union[str, bytes, os.PathLike],
            exporter: CollectionExporters,
            **kwargs
    ) -> None:
        """Export the current collection by a given exporter

        Args:
            export_path: path to export to
            exporter: exporter to choose
            **kwargs: additional arguments for exporter
        """
        exporter = CollectionExporterFactory.create_exporter(
            file_path=export_path,
            exporter=exporter
        )
        exporter.export(collection=self, **kwargs)

    @classmethod
    def import_data(
            cls,
            collection_path: Union[str, bytes, os.PathLike],
            import_path: Union[str, bytes, os.PathLike],
            importer: CollectionImporters,
            **kwargs
    ) -> Optional[Collection]:
        """Export the current collection by a given exporter

        Args:
            collection: Collection or Path of the final collection
            import_path: path to export to
            importer: importer to choose
            **kwargs: additional arguments for exporter
        """
        collection = cls(data_path=collection_path)
        importer = CollectionImporterFactory.create_importer(
            collection=collection,
            importer=importer
        )
        return importer.import_data(import_path=import_path, **kwargs)

    def redistribute(
            self,
            redistribution_configuration: RedistributionConfiguration,
            record_type: Optional[
                Union[
                    List[EventTypes],
                    EventTypes
                ]
            ] = None
    ) -> None:
        """Redistributes the events records according to the configuration.

        Args:
            redistribution_configuration: Configuration to redistribute by
            record_type: Record type to be redistributed.
        """
        rng = np.random.default_rng(redistribution_configuration.seed)

        if record_type is None:
            record_type = [e for e in EventType]

        if type(record_type) is not list:
            record_type = [record_type]

        record_type = [e.value for e in record_type]
        records = self.get_records()

        mode = redistribution_configuration.mode
        interval = redistribution_configuration.interval

        new_differences = []
        logging.info('Starting to redistribute with mode: {}'.format(mode))
        logging.debug(
            'Redistribution interval: [{},{})'.format(interval.start, interval.end)
        )

        if records is None:
            raise ValueError("No records to redistribute")

        with tqdm(total=len(records), mininterval=.5) as pbar:
            for record in records.df.itertuples():
                current_record_id = getattr(record, 'record_id')
                current_record_type = getattr(record, 'type')
                current_sources = self.get_sources(record_id=current_record_id)
                current_hits = self.get_hits(record_id=current_record_id)
                current_time = getattr(record, 'time')
                current_start = interval.start
                current_end = interval.end
                skip_record = False

                if current_record_type not in record_type:
                    skip_record = True

                if current_hits is None:
                    skip_record = True
                    logging.info(
                        'No hits for event {}. Skipping!'.format(current_record_id)
                    )

                if skip_record:
                    new_differences.append(0)
                    continue

                if mode != EventRedistributionMode.START_TIME:
                    percentile = None
                    if mode == EventRedistributionMode.CONTAINS_PERCENTAGE:
                        percentile = redistribution_configuration.percentile

                    hits_statistics = current_hits.get_statistics(percentile=percentile)

                    current_min = hits_statistics.min
                    current_max = hits_statistics.max

                    if mode == EventRedistributionMode.CONTAINS_HIT:
                        current_start = interval.start - current_max + current_time
                        current_end = interval.end - current_min + current_time

                    if (
                            mode == EventRedistributionMode.CONTAINS_EVENT
                            or mode == EventRedistributionMode.CONTAINS_PERCENTAGE
                    ):
                        current_length = current_max - current_min
                        current_offset = current_min - current_time
                        current_start = interval.start - current_offset
                        current_end = interval.end - current_offset - current_length

                # What happens when the event is having an error
                if current_end < current_start:
                    current_end = current_start + 1

                # What happens when the event is having an error
                new_time = rng.uniform(current_start, current_end)
                new_difference = new_time - current_time
                current_hits.add_time(new_difference)
                self.set_hits(current_hits, override=True)
                if current_sources is not None:
                    current_sources.add_time(new_difference)
                    self.set_sources(current_sources, override=True)

                new_differences.append(new_difference)
            pbar.update()

        records.add_time(new_differences)
        self.set_records(records=records, override=True)
        logging.info('Finished to redistribute with mode: {}'.format(mode))

    @classmethod
    def from_merge(cls, merge_configuration: MergeConfiguration) -> Collection:
        logging.info('Starting to merge collections with config.')
        collection_paths = merge_configuration.collection_paths
        if len(collection_paths) == 0:
            raise ValueError('No collection paths passed')
        dirname = os.path.dirname(merge_configuration.out_path)
        tmp_file = os.path.join(
            dirname,
            '_tmp_' + str(uuid.uuid4()) + 'data.h5'
        )
        os.makedirs(dirname, exist_ok=True)
        save_configuration(
            os.path.join(dirname, 'configuration.json'),
            merge_configuration
        )
        rng = np.random.default_rng(merge_configuration.seed)
        if len(collection_paths) > 1:
            logging.info('Starting to create joined temporary collection.')
            shutil.copy(collection_paths[0], tmp_file)
            tmp_collection = cls(data_path=tmp_file)

            for sub_collection_path in collection_paths[1:]:
                sub_collection = Collection(
                    data_path=sub_collection_path,
                    read_only=True
                )
                tmp_collection.append(
                    collection_to_append=sub_collection
                )
                tmp_collection.read_only = True
            logging.info('Finished creating joined temporary collection.')
        else:
            tmp_collection = Collection(collection_paths[0], read_only=True)

        if merge_configuration.redistribution is not None:
            tmp_collection.redistribute(
                redistribution_configuration=merge_configuration.redistribution
            )

        if merge_configuration.content is None:
            shutil.copy(tmp_file, merge_configuration.out_path)
            new_collection = cls(data_path=merge_configuration.out_path)
        else:
            new_collection = cls(data_path=merge_configuration.out_path)
            new_collection.set_detector(tmp_collection.get_detector())
            for content in merge_configuration.content:
                logging.info(
                    'Starting to create {} {} records'.format(
                        content.number_of_records,
                        content.primary_type
                    )
                )
                if content.secondary_types is not None:
                    logging.info(
                        'Secondary types: {}'.format(', '.join(content.secondary_types))
                    )
                # Collect for duplicate ID
                new_collection_records = new_collection.get_records()
                if new_collection_records is not None:
                    new_collection_record_ids = new_collection_records.record_ids
                else:
                    new_collection_record_ids = []
                # First load all primary records
                if content.number_of_records is not None:
                    number_of_records = content.number_of_records
                else:
                    number_of_records = len(new_collection_records)
                interval = content.interval
                primary_type = content.primary_type
                primary_records = tmp_collection.get_records(
                    record_type=primary_type
                )
                if primary_records is None or len(primary_records) < number_of_records:
                    raise ValueError(
                        'Not enough primary records of type {} given'.format(
                            primary_type
                        )
                    )

                # Now all secondary records
                secondary_records_list = []
                if content.secondary_types is not None:
                    for secondary_type in content.secondary_types:
                        secondary_records_list.append(
                            tmp_collection.get_records(
                                record_type=secondary_type
                            )
                        )

                added_record_ids = []
                misses = 0
                misses_break_number = 50

                with tqdm(total=number_of_records, mininterval=.5) as pbar:

                    # TODO: Check Performance
                    while len(
                            added_record_ids
                    ) < number_of_records and misses < misses_break_number:
                        current_primary_record = primary_records.sample(
                            n=1,
                            random_state=rng
                        )
                        current_primary_record_id = current_primary_record \
                            .record_ids.iloc[0]
                        # First set the primary hits and sources
                        primary_hits = tmp_collection.get_hits(
                            current_primary_record_id,
                            interval=interval
                        )

                        current_sources_list: List[Sources] = []
                        current_hits_list: List[Hits] = []

                        # skip primary records without hits
                        if primary_hits is None and content.filter_no_hits:
                            misses += 1
                            continue
                        else:
                            current_hits_list.append(primary_hits)
                            primary_sources = tmp_collection.get_sources(
                                current_primary_record_id,
                                interval=interval
                            )
                            if primary_sources is not None:
                                current_sources_list.append(primary_sources)

                        for secondary_records in secondary_records_list:
                            if secondary_records is None:
                                continue
                            current_secondary_record_id = secondary_records.sample(
                                n=1,
                                random_state=rng
                            ).record_ids.iloc[0]

                            current_sources = tmp_collection.get_sources(
                                record_id=current_secondary_record_id,
                                interval=interval
                            )
                            current_hits = tmp_collection.get_hits(
                                record_id=current_secondary_record_id,
                                interval=interval
                            )
                            if current_hits is not None:
                                current_hits_list.append(current_hits)
                            if current_sources is not None:
                                current_sources_list.append(current_sources)

                        combined_current_sources = Sources.concat(current_sources_list)
                        combined_current_hits = Hits.concat(current_hits_list)

                        if current_primary_record_id in added_record_ids or \
                                current_primary_record_id in new_collection_record_ids:
                            # TODO: Discuss what happens if record_id already added?
                            new_record_id = get_64_bit_signed_uuid_int()
                            logging.debug(
                                'Record id {} already added: Renaming to {}'.format(
                                    current_primary_record_id,
                                    new_record_id
                                )
                            )
                        else:
                            new_record_id = current_primary_record_id

                        new_record_id = np.int(new_record_id)

                        # Set all ids
                        current_primary_record.df['record_id'] = new_record_id
                        combined_current_hits.df['record_id'] = new_record_id

                        new_collection.set_records(current_primary_record, append=True)
                        new_collection.set_hits(combined_current_hits)

                        if combined_current_sources is not None:
                            combined_current_sources.df['record_id'] = new_record_id
                            new_collection.set_sources(combined_current_sources)

                        added_record_ids.append(new_record_id)
                        misses = 0
                        if new_collection.get_hits(
                                new_collection.get_records().record_ids.iloc[0]
                        ) is None:
                            print(current_primary_record_id)
                            print(new_record_id)

                        pbar.update()

                if misses == misses_break_number:
                    raise ValueError(
                        'Not enough primary records of type {} with hits'.format(
                            content.primary_type
                        )
                    )

                logging.info(
                    'Finished to create {} {} records'.format(
                        content.number_of_records,
                        content.primary_type
                    )
                )

        if os.path.isfile(tmp_file):
            os.remove(tmp_file)
        logging.info('Finished to merge collections with config.')
        return new_collection

    def recompress(self):
        dir = os.path.dirname(self.data_path)
        tmp_file = os.path.join(dir, '{}.h5'.format(str(uuid.uuid4())))
        self.store.close()
        command = [
            "ptrepack",
            "-o",
            "--chunkshape=auto",
            "--propindexes",
            "--complevel={}".format(self.complevel),
            self.data_path,
            tmp_file
        ]
        call(command)
        os.remove(self.data_path)
        shutil.move(tmp_file, self.data_path)
        self.store: pd.HDFStore = self.__get_store()

    def __get_records_where_path_exists(
            self,
            collection_key: CollectionKeys,
            record_type: Optional[
                Union[
                    List[EventTypes],
                    EventTypes
                ]
            ] = None,
            invert: bool = False,
            interval: Optional[Interval] = None,
            keys: Optional[List[str]] = None
    ) -> Optional[Records]:
        """Returns records where the path exists in store.

        Args:
            collection_key: Collection key to return records for
            record_type: Type of Record to get
            invert: Get records where path does not exist
            interval: Interval to get have hits in.
            keys: Cached store keys instead of regenerating (Optimization)

        Returns:
            Records that have a path within the collection
        """
        records = self.get_records(record_type=record_type)
        if records is None:
            return None
        column_name = '_tmp_data_path'
        records.df[column_name] = self.__get_hdf_path(
            collection_key,
            records.record_ids
        )
        if keys is None:
            keys = self.store.keys()
        record_in_keys = records.df[column_name].isin(keys)
        if not invert:
            records.df = records.df[record_in_keys]
        else:
            records.df = records.df[~record_in_keys]
        records.df.drop(columns=[column_name], inplace=True)

        if interval is not None:
            records = self.__drop_records_without_hits_in_interval(
                records=records,
                interval=interval
            )

        if len(records) == 0:
            return None

        return records

    def get_records_with_hits(
            self,
            record_type: Optional[
                Union[
                    List[EventTypes],
                    EventTypes
                ]
            ] = None,
            invert: bool = False,
            interval: Optional[Interval] = None,
            keys: Optional[List[str]] = None
    ) -> Optional[Records]:
        """Gets all records that have hits.

        Args:
            invert: Get records without hits instead.
            record_type: Type of Record to get
            interval: Interval to get have hits in.
            keys: Cached store keys instead of regenerating (Optimization)

        Returns:
            Records that have a hit dataset in store
        """
        return self.__get_records_where_path_exists(
            collection_key=CollectionKeys.HITS,
            record_type=record_type,
            invert=invert,
            interval=interval,
            keys=keys
        )

    def get_records_with_sources(
            self,
            record_type: Optional[
                Union[
                    List[EventTypes],
                    EventTypes
                ]
            ] = None,
            invert: bool = False,
            interval: Optional[Interval] = None,
            keys: Optional[List[str]] = None
    ) -> Optional[Records]:
        """Gets all records that have sources.

        Args:
            record_type: Type of Record to get
            invert: Get records without sources instead.
            interval: Interval to get have hits in.
            keys: Cached store keys instead of regenerating (Optimization)

        Returns:
            Records that have a source dataset in store
        """
        return self.__get_records_where_path_exists(
            collection_key=CollectionKeys.SOURCES,
            record_type=record_type,
            invert=invert,
            interval=interval,
            keys=keys
        )

    def drop_no_hit_records(self) -> None:
        """Deletes all the records without hits."""
        logging.info('Dropping no hit records')
        keys = self.store.keys()
        records_without_hits = self.get_records_with_hits(invert=True, keys=keys)
        if records_without_hits is None:
            return
        records_with_sources = self.get_records_with_sources(keys=keys)
        if records_with_sources is not None:
            records_with_sources.df = records_with_sources.df[
                records_with_sources.record_ids.isin(records_without_hits.record_ids)
            ]
            for index, record_id in records_with_sources.record_ids.items():
                sources_path = self.__get_hdf_path(
                    collection_key=CollectionKeys.SOURCES,
                    record_id=record_id
                )
                del self.store[sources_path]
        logging.info('Dropped {} records'.format(len(records_without_hits)))
        all_records = self.get_records()
        all_records.df = all_records.df[
            ~all_records.record_ids.isin(records_without_hits.record_ids)
        ]
        if len(all_records) == 0:
            self.store.remove(CollectionKeys.RECORDS.value)
        else:
            self.set_records(all_records, override=True)

    def get_record_statistics(self) -> Optional[RecordStatistics]:
        """Gets all the statistics added to the current records.

        Returns:
            RecordStatistics enriched by counts and first and last sources and hits.
        """
        records = self.get_records()
        if records is None:
            return None
        hits_counts = []
        sources_counts = []
        first_sources = []
        last_sources = []
        first_hits = []
        last_hits = []
        number_records = len(records)
        empty_hits = 0
        empty_sources = 0
        with tqdm(total=number_records, mininterval=0.5) as pbar:
            for index, record_id in records.record_ids.items():
                record_hits = self.get_hits(record_id=record_id)
                if record_hits is None:
                    hits_counts.append(np.nan)
                    first_hits.append(np.nan)
                    last_hits.append(np.nan)
                    empty_hits += 1
                else:
                    hits_statistics = record_hits.get_statistics()
                    hits_counts.append(hits_statistics.count)
                    first_hits.append(hits_statistics.min)
                    last_hits.append(hits_statistics.max)
                record_sources = self.get_sources(record_id=record_id)
                if record_sources is None:
                    sources_counts.append(np.nan)
                    first_sources.append(np.nan)
                    last_sources.append(np.nan)
                    empty_sources += 1
                else:
                    sources_statistics = record_sources.get_statistics()
                    sources_counts.append(sources_statistics.count)
                    first_sources.append(sources_statistics.min)
                    last_sources.append(sources_statistics.max)
                pbar.update()

        df = records.df
        if empty_hits != number_records:
            df['hit_count'] = hits_counts
            df['first_hit'] = first_hits
            df['last_hit'] = last_hits
        if empty_sources != number_records:
            df['source_count'] = sources_counts
            df['first_source'] = first_sources
            df['last_source'] = last_sources

        return RecordStatistics(df=df)
