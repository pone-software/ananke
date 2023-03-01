"""Module containing a collection."""
from __future__ import annotations

import logging
import os
import warnings
from enum import Enum
from typing import List, Optional, Type, TypeVar, Union

import numpy as np
import pandas as pd
from tables import NaturalNameWarning, PerformanceWarning
from tqdm import tqdm

from ananke.configurations.collection import StorageConfiguration, ExportConfiguration, MergeConfiguration
from ananke.configurations.events import (
    Interval, RedistributionConfiguration, EventRedistributionMode,
)
from ananke.models.detector import Detector
from ananke.models.event import Hits, Records, Sources, RecordStatistics
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import EventTypes, EventType
from ananke.services.collection.exporters import (
    AbstractCollectionExporter,
)
from ananke.services.collection.importers import (
    AbstractCollectionImporter,
)
from ananke.services.collection.storage import StorageFactory
from ananke.utils import get_64_bit_signed_uuid_int

warnings.filterwarnings("ignore", category=NaturalNameWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)


class CollectionKeys(str, Enum):
    """Enum containing keys for the collection file."""

    RECORDS = "records"
    HITS = "hits"
    SOURCES = "sources"
    DETECTOR = "detector"


DataFrameFacade_ = TypeVar("DataFrameFacade_", bound=DataFrameFacade)


class Collection:
    def __init__(self, configuration: StorageConfiguration):
        self.storage = StorageFactory.create(configuration)
        self.logger = logging.getLogger(self.__class__.__name__)

    def __enter__(self):
        self.storage.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.storage.__exit__(exc_type, exc_val, exc_tb)

    def open(self):
        self.storage.open()

    def close(self):
        self.storage.close()

    def copy(
            self,
            configuration: StorageConfiguration,
            record_types: Optional[
                Union[
                    List[EventTypes],
                    EventTypes
                ]
            ] = None,
            record_ids: Optional[
                Union[int, List[int], pd.Series]
            ] = None,
            interval: Optional[Interval] = None,
            batch_size: int = 100
    ) -> Collection:
        """Create a copy of current collection with new storage configuration

        Args:
            configuration: Storage configuration of new collection
            record_types: Types to select
            record_ids: record ids to select
            interval: Interval to filter by
            batch_size: Number of records to do in one batch

        Returns:

        """
        copy = self.__class__(configuration)
        copy.open()
        copy.storage.set_detector(self.storage.get_detector())
        copy_records = self.storage.get_records(record_types=record_types, record_ids=record_ids, interval=interval)

        copy.storage.set_records(copy_records)

        copy_records.record_ids.items()

        with tqdm(total=len(copy_records)) as pbar:
            for records in copy_records.iterbatches(batch_size=batch_size):
                record_ids = records.record_ids
                current_sources = self.storage.get_sources(record_ids=record_ids)
                current_hits = self.storage.get_hits(record_ids=record_ids)
                if current_sources is not None:
                    copy.storage.set_sources(current_sources)
                if current_hits is not None:
                    copy.storage.set_hits(current_hits)
                pbar.update(batch_size)
        copy.close()
        return copy

    def import_data(
            self,
            importer: Type[AbstractCollectionImporter],
            **kwargs
    ) -> Optional[Collection]:
        """Export the current collection by a given exporter

        Args:
            importer: importer to choose
            **kwargs: additional arguments for exporter
        """
        importer_instance = importer(collection=self)
        return importer_instance.import_data(**kwargs)

    def export(
            self,
            configuration: ExportConfiguration,
            exporter: Type[AbstractCollectionExporter],
            **kwargs
    ) -> None:
        """Export the current collection by a given exporter

        Args:
            configuration: configuration of the exporter.
            exporter: exporter to choose
            **kwargs: additional arguments for exporter
        """
        exporter_instance = exporter(configuration=configuration)
        exporter_instance.export(collection=self, **kwargs)

    def redistribute(
            self,
            redistribution_configuration: RedistributionConfiguration,
            record_types: Optional[
                Union[
                    List[EventTypes],
                    EventTypes
                ]
            ] = None
    ) -> None:
        """Redistributes the events records according to the configuration.

        Args:
            redistribution_configuration: Configuration to redistribute by
            record_types: Record type to be redistributed.
        """
        rng = np.random.default_rng(redistribution_configuration.seed)

        if record_types is None:
            record_types = [e for e in EventType]
        records = self.storage.get_records()

        mode = redistribution_configuration.mode
        interval = redistribution_configuration.interval

        new_differences = []
        self.logger.info('Starting to redistribute with mode: {}'.format(mode))
        self.logger.debug(
            'Redistribution interval: [{},{})'.format(interval.start, interval.end)
        )

        if records is None:
            raise ValueError("No records to redistribute")

        with tqdm(total=len(records), mininterval=.5) as pbar:
            for record in records.df.itertuples():
                current_record_id = getattr(record, 'record_id')
                current_record_type = getattr(record, 'type')
                current_sources = self.storage.get_sources(record_ids=current_record_id)
                current_hits = self.storage.get_hits(record_ids=current_record_id)
                current_time = getattr(record, 'time')
                current_start = interval.start
                current_end = interval.end
                skip_record = False

                if current_record_type not in record_types:
                    skip_record = True

                if current_hits is None:
                    skip_record = True
                    self.logger.info(
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
                self.storage.del_hits(record_ids=current_record_id)
                self.storage.set_hits(current_hits)
                if current_sources is not None:
                    current_sources.add_time(new_difference)
                    self.storage.del_sources(record_ids=current_record_id)
                    self.storage.set_sources(current_sources)

                new_differences.append(new_difference)

                pbar.update()

        records.add_time(new_differences)
        self.storage.set_records(records=records, append=False)
        self.logger.info('Finished to redistribute with mode: {}'.format(mode))

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
        self.logger.info('Starting to append collection.')

        collection_to_append.open()
        append_detector = collection_to_append.storage.get_detector()
        own_detector = self.storage.get_detector()

        if own_detector is not None and not append_detector.df.equals(own_detector.df):
            raise ValueError("Cannot merge two collections with different detectors.")

        if own_detector is None:
            self.storage.set_detector(append_detector)

        records_ids_with_hits = collection_to_append.storage.get_record_ids_with_hits()
        if records_ids_with_hits is not None:
            records_with_hits_record_ids = records_ids_with_hits.values
        else:
            records_with_hits_record_ids = []

        if with_hits_only:
            append_records = collection_to_append.storage.get_records(record_ids=records_ids_with_hits)
        else:
            append_records = collection_to_append.storage.get_records()

        records_ids_with_sources = collection_to_append.storage.get_record_ids_with_sources()
        if records_ids_with_sources is not None:
            records_with_sources_record_ids = records_ids_with_sources.values
        else:
            records_with_sources_record_ids = []

        if append_records is not None:
            self.storage.set_records(append_records, append=True)

        with tqdm(total=len(append_records), mininterval=.5) as pbar:

            for (index, record_id) in append_records.record_ids.items():
                if record_id in records_with_hits_record_ids:
                    current_hits = collection_to_append.storage.get_hits(
                        record_ids=record_id,
                        interval=interval
                    )
                    self.storage.set_hits(current_hits)

                if record_id in records_with_sources_record_ids:
                    current_sources = collection_to_append.storage.get_sources(
                        record_ids=record_id,
                        interval=interval
                    )
                    self.storage.set_sources(current_sources)
                pbar.update()
        self.logger.info('Finished to append collection.')

    def drop_no_hit_records(self) -> None:
        """Deletes all the records without hits."""
        logging.info('Dropping no hit records')
        record_ids_with_hits = self.storage.get_record_ids_with_hits()
        if record_ids_with_hits is None:
            record_ids_with_hits = []

        records = self.storage.get_records()
        records_ids = records.record_ids
        records_ids_without_hits = records_ids[~records_ids.isin(record_ids_with_hits)]
        self.storage.del_sources(record_ids=records_ids_without_hits)
        self.storage.del_records(record_ids=records_ids_without_hits)
        logging.info('Dropped {} records'.format(len(records_ids_without_hits)))

    def get_record_statistics(self) -> Optional[RecordStatistics]:
        """Gets all the statistics added to the current records.

        Returns:
            RecordStatistics enriched by counts and first and last sources and hits.
        """
        records = self.storage.get_records()
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
                record_hits = self.storage.get_hits(record_ids=record_id)
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
                record_sources = self.storage.get_sources(record_ids=record_id)
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

        df: pd.DataFrame = records.df
        if empty_hits != number_records:
            df['hit_count'] = hits_counts
            df['first_hit'] = first_hits
            df['last_hit'] = last_hits
        if empty_sources != number_records:
            df['source_count'] = sources_counts
            df['first_source'] = first_sources
            df['last_source'] = last_sources

        return RecordStatistics(df=df)

    @classmethod
    def from_merge(cls, merge_configuration: MergeConfiguration) -> Collection:
        logging.info('Starting to merge collections with config.')
        collections = merge_configuration.in_collections
        if len(collections) == 0:
            raise ValueError('No collections passed')
        rng = np.random.default_rng(merge_configuration.seed)
        if len(collections) > 1:
            first_collection = cls(collections[0])
            logging.info('Starting to create joined temporary collection.')
            tmp_collection = first_collection.copy(merge_configuration.tmp_collection)
            tmp_collection.open()

            for sub_collection in collections[1:]:
                sub_collection = cls(configuration=sub_collection)
                tmp_collection.append(
                    collection_to_append=sub_collection
                )
                tmp_collection.read_only = True
            tmp_collection.close()
            logging.info('Finished creating joined temporary collection.')
        else:
            tmp_collection = cls(collections[0])

        if merge_configuration.redistribution is not None:
            tmp_collection.redistribute(
                redistribution_configuration=merge_configuration.redistribution
            )

        if merge_configuration.content is None:
            new_collection = tmp_collection.copy(merge_configuration.out_collection)
        else:
            new_collection = cls(merge_configuration.out_collection)
            new_collection.open()
            tmp_collection.open()
            new_collection.storage.set_detector(tmp_collection.storage.get_detector())
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
                new_collection_records = new_collection.storage.get_records()
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
                primary_records = tmp_collection.storage.get_records(
                    record_types=primary_type
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
                            tmp_collection.storage.get_records(
                                record_types=secondary_type
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
                        primary_hits = tmp_collection.storage.get_hits(
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
                            primary_sources = tmp_collection.storage.get_sources(
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

                            current_sources = tmp_collection.storage.get_sources(
                                record_ids=current_secondary_record_id,
                                interval=interval
                            )
                            current_hits = tmp_collection.storage.get_hits(
                                record_ids=current_secondary_record_id,
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

                        new_collection.storage.set_records(current_primary_record, append=True)
                        new_collection.storage.set_hits(combined_current_hits)

                        if combined_current_sources is not None:
                            combined_current_sources.df['record_id'] = new_record_id
                            new_collection.storage.set_sources(combined_current_sources)

                        added_record_ids.append(new_record_id)
                        misses = 0
                        if new_collection.storage.get_hits(
                                new_collection.storage.get_records().record_ids.iloc[0]
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

        # TODO: Delete temporary collection
        logging.info('Finished to merge collections with config.')
        return new_collection
