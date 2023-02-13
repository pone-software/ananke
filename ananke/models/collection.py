"""Module containing a collection."""
from __future__ import annotations

import logging
import os
import shutil
import uuid
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, List, Optional, Type, TypeVar, Union

import h5py
import numpy as np
import pandas as pd
import awkward as ak
from tables import NaturalNameWarning

from ananke.configurations.collection import MergeConfiguration
from ananke.configurations.events import (
    EventRedistributionMode,
    RedistributionConfiguration, Interval,
)
from ananke.models.detector import Detector
from ananke.models.event import Hits, RecordIds, Records, Sources
from ananke.models.geometry import Vectors3D
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import RecordType
from ananke.utils import percentile

warnings.filterwarnings("ignore", category=NaturalNameWarning)


class CollectionKeys(str, Enum):
    """Enum containing keys for the collection file."""

    RECORDS = "records"
    HITS = "hits"
    SOURCES = "sources"
    DETECTOR = "detector"


DataFrameFacade_ = TypeVar("DataFrameFacade_", bound=DataFrameFacade)


# TODO: Check Closing
class Collection:
    """Class combining all data frames to a complete record collection."""

    def __init__(
            self,
            data_path: str,
            records: Optional[Records] = None,
            detector: Optional[Detector] = None,
            complevel: int = 3,
            override: bool = False,
    ):
        """Constructor for the collection.

        Args:
            data_path: Path to store/read collection from
            records: Optional records to store directly
            detector: Optional detector to store directly
            complevel: Compression level for data file
            override: Override existing data
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
        return pd.HDFStore(self.data_path)

    def __del__(self):
        """Close store on delete."""
        self.store.close()

    def __get_hdf_path(
            self, collection_key: CollectionKeys, record_id: int | str
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
        return "/{key}/{record_id}".format(
            key=collection_key.value, record_id=record_id
        )

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
            data_columns=True,
            # append=append,
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
            record_type: Optional[RecordType] = None
    ) -> Optional[Records]:

        """Gets records of the collection.

        Args:
            record_type: record type to include

        Returns:
            Records of the collection.
        """
        # TODO: Fix Where Caching
        where = None
        if record_type is not None:
            where = '(type={})'.format(record_type)
        return self.__get_data(
            key=CollectionKeys.RECORDS,
            facade_class=Records,
            where=where
        )

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
            interval: Optional[Interval] = None
    ) -> None:
        """Concatenate multiple connections.

        Args:
            collection_to_append: Collection to append to current one.
            interval: interval to consider for appending

        Returns:
            A single collection combining the previous ones.
        """
        append_detector = collection_to_append.get_detector()
        own_detector = self.get_detector()

        if own_detector is not None and not append_detector.df.equals(own_detector.df):
            raise ValueError("Cannot merge two collections with different detectors.")

        if own_detector is None:
            self.set_detector(append_detector)

        append_records = collection_to_append.get_records()

        if append_records is not None:
            self.set_records(append_records, append=True)

        for (index, record_id) in append_records.record_ids.items():
            current_hits = collection_to_append.get_hits(
                record_id=record_id,
                interval=interval
            )
            if current_hits is not None:
                self.set_hits(current_hits)

            current_sources = collection_to_append.get_sources(
                record_id=record_id,
                interval=interval
            )
            if current_sources is not None:
                self.set_sources(current_sources)

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

    # TODO: Adapt to new
    def redistribute(
            self,
            redistribution_configuration: RedistributionConfiguration
    ) -> None:
        """Redistributes the events records according to the configuration.

        Args:
            redistribution_configuration: Configuration to redistribute by
        """
        rng = np.random.default_rng(redistribution_configuration.seed)

        records = self.get_records()

        if records is None:
            raise ValueError("No records to redistribute")

        modification_df = records.df[["record_id", "time"]]

        mode = redistribution_configuration.mode
        interval = redistribution_configuration.interval

        if mode == EventRedistributionMode.START_TIME.value:
            modification_df["start"] = interval.start
            modification_df["end"] = interval.end
        else:
            if mode == EventRedistributionMode.CONTAINS_PERCENTAGE:
                beginning_percentile = 0.5 - redistribution_configuration.percentile / 2.0
                ending_percentile = 0.5 + redistribution_configuration.percentile / 2.0
                aggregations: List[Union[str, Callable[[Any], Any]]] = [
                    percentile(beginning_percentile, "min"),
                    percentile(ending_percentile, "max"),
                ]
            else:
                aggregations = ["min", "max"]

            grouped_hits = self.hits.df.groupby("record_id").agg({"time": aggregations})
            modification_df = modification_df.merge(
                grouped_hits, on="record_id", how="left"
            )

        if mode == EventRedistributionMode.CONTAINS_HIT.value:
            modification_df["start"] = (
                    interval.start - modification_df["max"] + modification_df["time"]
            )
            modification_df["end"] = (
                    interval.end - modification_df["min"] + modification_df["time"]
            )

        if (
                mode == EventRedistributionMode.CONTAINS_EVENT.value
                or mode == EventRedistributionMode.CONTAINS_PERCENTAGE.value
        ):
            modification_df["length"] = modification_df["max"] - modification_df["min"]
            modification_df["offset"] = modification_df["min"] - modification_df["time"]
            modification_df["start"] = interval.start - modification_df["offset"]
            modification_df["end"] = (
                    interval.end - modification_df["offset"] - modification_df["length"]
            )

        # What happens when the event is having an error
        modification_df[modification_df["end"] < modification_df["start"]] = (
                modification_df["start"] + 1
        )

        modification_df["new_time"] = rng.uniform(
            modification_df["start"], modification_df["end"]
        )

        modification_df["difference"] = (
                modification_df["time"] - modification_df["new_time"]
        )

    @classmethod
    def from_merge(cls, merge_configuration: MergeConfiguration) -> Collection:
        collection_paths = merge_configuration.collection_paths
        if len(collection_paths) == 0:
            raise ValueError('No collection paths passed')
        tmp_path = os.path.join(os.path.dirname(merge_configuration.out_path), '_tmp')
        tmp_file = os.path.join(tmp_path, 'data.h5')
        os.makedirs(tmp_path, exist_ok=False)
        rng = np.random.default_rng(merge_configuration.seed)
        shutil.copy(collection_paths[0], tmp_file)
        tmp_collection = cls(data_path=tmp_file)

        for sub_collection_path in collection_paths[1:]:
            sub_collection = Collection(data_path=sub_collection_path)
            tmp_collection.append(
                collection_to_append=sub_collection
            )

        if merge_configuration.redistribution is not None:
            tmp_collection.redistribute(
                redistribution_configuration=merge_configuration.redistribution
            )

        if merge_configuration.content is None:
            shutil.move(tmp_file, merge_configuration.out_path)
            new_collection = cls(data_path=merge_configuration.out_path)
        else:
            new_collection = cls(data_path=merge_configuration.out_path)
            new_collection.set_detector(tmp_collection.get_detector())
            for content in merge_configuration.content:
                # Collect for duplicate ID
                new_collection_records = new_collection.get_records()
                if new_collection_records is not None:
                    new_collection_record_ids = new_collection_records.record_ids
                else:
                    new_collection_record_ids = []
                # First load all primary records
                number_of_records = content.number_of_records
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

                # TODO: Check Perfornance
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
                        new_record_id = uuid.uuid1().int >> 64
                        logging.warning(
                            'Record id {} already added: Renaming to {}'.format(
                                current_primary_record_id,
                                new_record_id
                            )
                        )
                    else:
                        new_record_id = current_primary_record_id


                    if new_record_id == 8173088588305338368:
                        print('cool')

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

                if misses == misses_break_number:
                    raise ValueError(
                        'Not enough primary records of type {} with hits'.format(
                            content.primary_type
                        )
                    )

        shutil.rmtree(tmp_path)
        return new_collection


class CollectionExporters(Enum):
    """Enum for possible exporters"""
    GRAPH_NET = 'graph_net'


class AbstractCollectionExporter(ABC):
    """Abstract parent class for collection exporters."""

    def __init__(self, file_path: Union[str, bytes, os.PathLike]):
        """Constructor of the Abstract Collection Exporter.

        Args:
            file_path: Filepath to store data at
        """
        self.file_path = file_path

    @abstractmethod
    def export(self, collection: Collection, **kwargs) -> None:
        """Abstract stub for the export of a collection.

        Args:
            collection: Collection to be exported
            kwargs: Additional exporter args
        """
        pass


class GraphNetCollectionExporter(AbstractCollectionExporter):
    """Concrete implementation for Graph Net exports."""

    def __get_file_path(self, batch_number: int) -> str:
        """Generates a batches file path.

        Args:
            batch_number: number of the batch of file.

        Returns:
            complete path of current file.
        """
        return os.path.join(self.file_path, 'batch_{}.parquet'.format(batch_number))

    @staticmethod
    def __get_mapped_hits_df(hits: Hits) -> pd.DataFrame:
        """Return hits mapped to graph nets columns and format.

        Args:
            hits: Hits to map

        Returns:
            Data frame with mapped hits
        """
        new_hits_df = hits.df[[
            'record_id',
            'pmt_id',
            'string_id',
            'module_id',
            'time'
        ]].rename(
            columns={
                'record_id': 'event_id',
                'pmt_id': 'pmt_idx',
                'module_id': 'dom_idx',
                'string_id': 'string_idx'
            }
        )

        return new_hits_df

    def export(self, collection: Collection, batch_size=20, **kwargs) -> None:
        """Graph net export of a collection.

        Args:
            collection: Collection to be exported
            batch_size: Events per file
            kwargs: Additional exporter args
        """
        records = collection.get_records()

        # TODO: Properly implement interaction type
        new_records = pd.DataFrame(
            {
                'event_id': records.df['record_id'],
                'interaction_type': 0
            },
            dtype='int'
        )

        if 'orientation_x' in records.df:
            orientations = Vectors3D.from_df(records.df, prefix='orientation_')
            new_records['azimuth'] = orientations.phi
            new_records['zenith'] = orientations.theta

        if 'location_x' in records.df:
            new_records['interaction_x'] = records.df['location_x']
            new_records['interaction_y'] = records.df['location_y']
            new_records['interaction_z'] = records.df['location_z']

        if 'energy' in records.df:
            new_records['energy'] = records.df['energy']

        if 'particle_id' in records.df:
            new_records['pid'] = records.df['particle_id']

        mandatory_columns = [
            'azimuth', 'zenith', 'interaction_x', 'interaction_y',
            'interaction_z', 'energy'
        ]

        for mandatory_column in mandatory_columns:
            if mandatory_column not in new_records:
                new_records[mandatory_column] = -1

        new_records.fillna(-1, inplace=True)
        new_records_event_ids = new_records['event_id']

        os.makedirs(self.file_path, exist_ok=True)

        number_of_records = len(records)
        mc_truths = []
        detector_responses = []
        batch = 0

        detector = collection.get_detector()

        indices = detector.indices.rename(
            columns={
                'string_id': 'string_idx',
                'module_id': 'dom_idx',
                'pmt_id': 'pmt_idx',
            }
        )
        orientations = detector.pmt_orientations
        locations = detector.pmt_locations.get_df_with_prefix('pmt_')

        merge_detector_df = pd.concat([indices, locations], axis=1)
        merge_detector_df['pmt_azimuth'] = orientations.phi
        merge_detector_df['pmt_zenith'] = orientations.theta

        for index, row in enumerate(new_records.itertuples(index=False)):
            current_record_id = getattr(row, 'event_id')

            current_hits = collection.get_hits(record_id=current_record_id)

            if current_hits is None:
                continue

            mapped_hits = self.__get_mapped_hits_df(current_hits)
            mapped_hits = pd.merge(
                mapped_hits,
                merge_detector_df,
                how='inner',
                on=['string_idx', 'dom_idx', 'pmt_idx']
            )

            mc_truths.append(row._asdict())
            detector_responses.append(mapped_hits.to_dict(orient='records'))

            if (index + 1) % batch_size == 0 or index + 1 == number_of_records:
                array = ak.Array(
                    {
                        'mc_truth': mc_truths,
                        'detector_response': detector_responses
                    }
                )
                ak.to_parquet(array, self.__get_file_path(batch), compression='GZIP')
                mc_truths = []
                detector_responses = []
                batch += 1


class CollectionExporterFactory:
    @staticmethod
    def create_exporter(
            file_path,
            exporter: CollectionExporters
    ) -> AbstractCollectionExporter:
        if exporter == CollectionExporters.GRAPH_NET:
            return GraphNetCollectionExporter(file_path)
        else:
            raise ValueError(f'Unsupported exporter {exporter.value}')
