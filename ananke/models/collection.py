"""Module containing a collection."""
from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod

from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union

import h5py
import numpy as np
import pandas as pd
import awkward as ak

from ananke.configurations.events import (
    EventRedistributionMode,
    RedistributionConfiguration,
)
from ananke.models.detector import Detector
from ananke.models.event import Hits, RecordIds, Records, Sources
from ananke.models.geometry import Vectors3D
from ananke.models.interfaces import DataFrameFacade
from ananke.utils import percentile
from tables import NaturalNameWarning

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
        file_extensions = (".hdf", ".h5")
        if not data_path.lower().endswith(file_extensions):
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
        self.__cache: Dict[str, DataFrameFacade] = {}

        if detector is not None:
            self.set_detector(detector, override)

        if records is not None:
            self.set_records(records, override)

    def get_store(self) -> pd.HDFStore:
        """Gets the stored hdf object.

        Returns:
            stored hdf object.
        """
        return pd.HDFStore(self.data_path)

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

    def __get_cache(self, key: str) -> Optional[DataFrameFacade]:
        """Get a data frame facade from cache or return None.

        Args:
            key: Key to get

        Returns:
            Data frame facade with key or none.
        """
        if key in self.__cache:
            return self.__cache[key]

        return None

    def __set_cache(self, key: str, data: DataFrameFacade) -> None:
        """Sets a data frame facade to the cache.

        Args:
            key: Key to store data at
            data: Data to store
        """
        self.__cache[key] = data

    # TODO: Beautify Cache Flags
    def __get_data(
            self,
            key: str,
            facade_class: Type[DataFrameFacade_],
            update: bool = True,
            cache: bool = True,
    ) -> Optional[DataFrameFacade_]:
        """Gets data frame facade from file or cache.

        Args:
            key: Key to get data by
            facade_class: Data frame facade class to instantiate
            update: Get from file and not memory
            cache: Cache gotten result or not

        Returns:
            Data frame facade containing data or None
        """
        if not update:
            data_in_cache = self.__get_cache(key)
            if data_in_cache is not None:
                return data_in_cache

        try:
            store = self.get_store()
            df = store.get(key)
            facade = facade_class(df=df)
            if cache:
                self.__set_cache(key=key, data=facade)
            store.close()
            return facade
        except KeyError:
            return None

    def __set_data(
            self,
            key: str,
            data: DataFrameFacade,
            append: bool = False,
            cache: bool = True,
            override: bool = False,
    ) -> None:
        """Sets data frame facade to file by a specific key.

        Args:
            key: Key to store data at
            data: Data frame facade to store
            append: Append to existing data if necessary
            cache: Cache new data at the end
            override: Override existing data yes or no.
        """
        if (
                not override
                and self.__get_data(key, type(data), cache=False) is not None
                and not append
        ):
            raise ValueError("Cannot override data with key {}".format(key))

        # TODO: Evaluate different options
        store = self.get_store()
        store.put(
            key=key,
            value=data.df,
            format="t",
            complevel=self.complevel,
            append=append,
            index=False,
        )
        store.close()
        if cache:
            if append:
                self.__set_cache(
                    key,
                    data=self.__get_data(key=key, facade_class=type(data), update=True),
                )
            else:
                self.__set_cache(key=key, data=data)

    def get_detector(self, update: bool = False) -> Optional[Detector]:
        """Gets detector of the collection.

        Args:
            update: get from file and not cache

        Returns:
            Detector of the collection.
        """
        return self.__get_data(CollectionKeys.DETECTOR, Detector, update)

    def get_records(self, update: bool = False) -> Optional[Records]:
        """Gets records of the collection.

        Args:
            update: get from file and not cache

        Returns:
            Records of the collection.
        """
        return self.__get_data(CollectionKeys.RECORDS, Records, update)

    def __get_subgroup_dataset(
            self,
            base_key: CollectionKeys,
            facade_class: Type[DataFrameFacade_],
            record_id: int | str,
            update: bool = False,
    ) -> Optional[DataFrameFacade_]:
        """Gets data of group based on record id.

        Args:
            base_key: key of the subgroup
            facade_class: Data frame facade class to instantiate
            record_id: Record id to get hits from
            update: get from file and not cache

        Returns:
            Data frame facade with the data from cache or disc.
        """
        key = self.__get_hdf_path(base_key, record_id)
        return self.__get_data(key, facade_class, update)

    # TODO: Allow multiple or all
    def get_sources(
            self, record_id: int | str, update: bool = False
    ) -> Optional[Sources]:
        """Gets sources by a specific record id.

        Args:
            record_id: Record id to get hits from
            update: get from file and not cache

        Returns:
            None if not available or sources for record id
        """
        return self.__get_subgroup_dataset(
            base_key=CollectionKeys.SOURCES,
            facade_class=Sources,
            record_id=record_id,
            update=update,
        )

    # TODO: Allow multiple or all
    def get_hits(self, record_id: int | str, update: bool = False) -> Optional[Hits]:
        """Gets hits by a specific record id.

        Args:
            record_id: Record id to get hits from
            update: get from file and not cache

        Returns:
            None if not available or hits for record id
        """
        return self.__get_subgroup_dataset(
            base_key=CollectionKeys.HITS,
            facade_class=Hits,
            record_id=record_id,
            update=update,
        )

    def set_detector(
            self, detector: Detector, cache: bool = True, override: bool = False
    ) -> None:
        """Sets, overrides or appends detector to the existing data file.

        Args:
            detector: Detector to set
            cache: Cache set data
            override: Override already saved data
        """
        self.__set_data(
            CollectionKeys.DETECTOR.value, detector, cache=cache, override=override
        )

    def set_records(
            self,
            records: Records,
            cache: bool = True,
            override: bool = False,
            append: bool = False,
    ) -> None:
        """Sets, overrides or appends records to the existing data file.

        Args:
            records: Records to set
            cache: Cache set data
            override: Override already saved data
            append: Append data instead of override
        """
        self.__set_data(
            CollectionKeys.RECORDS.value,
            records,
            cache=cache,
            override=override,
            append=append,
        )

    def __set_subgroup_dataset(
            self,
            base_key: CollectionKeys,
            data: RecordIds,
            cache: bool = False,
            override: bool = False,
    ) -> None:
        """Sets grouped datasets.

        Args:
            base_key: Key of the group
            data: Data frame with record ids
            cache: Cache set hits
            override: Override already saved hits
        """
        record_ids = data.record_ids.drop_duplicates()
        for index, record_id in record_ids.items():
            entry_by_record = data.get_by_record(record_id)
            hdf_key = self.__get_hdf_path(base_key, record_id)
            self.__set_data(hdf_key, entry_by_record, cache=cache, override=override)

    def set_hits(self, hits: Hits, cache: bool = False, override: bool = False) -> None:
        """Sets hits to the collection.

        Args:
            hits: Hits to set
            cache: Cache set hits
            override: Override already saved hits
        """
        self.__set_subgroup_dataset(
            base_key=CollectionKeys.HITS, data=hits, cache=cache, override=override
        )

    def set_sources(
            self, sources: Sources, cache: bool = False, override: bool = False
    ) -> None:
        """Sets sources to the collection.

        Args:
            sources: Sources to set
            cache: Cache set sources
            override: Override already saved sources
        """
        self.__set_subgroup_dataset(
            base_key=CollectionKeys.SOURCES,
            data=sources,
            cache=cache,
            override=override,
        )

    def append(self, collection_to_append: Collection) -> None:
        """Concatenate multiple connections.

        Args:
            collection_to_append: Collection to append to current one.

        Returns:
            A single collection combining the previous ones.
        """
        append_detector = collection_to_append.get_detector()
        own_detector = self.get_detector()

        if append_detector is None or own_detector is None:
            raise ValueError("One detector is not set")

        if not append_detector.df.equals(own_detector.df):
            raise ValueError("Cannot merge two collections with different detectors.")

        append_records = collection_to_append.get_records()

        if append_records is not None:
            self.set_records(append_records, append=True)

        collection_to_append_store = collection_to_append.get_store()

        for path, group, leaves in collection_to_append_store.walk("/hits"):
            for leave in leaves:
                current_hits = collection_to_append.get_hits(leave)
                if current_hits is not None:
                    self.set_hits(current_hits)

        for path, group, leaves in collection_to_append_store.walk("/sources"):
            for leave in leaves:
                current_sources = collection_to_append.get_sources(leave)
                if current_sources is not None:
                    self.set_sources(current_sources)

        collection_to_append_store.close()

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
    def redistribute(self, redistribution_config: RedistributionConfiguration) -> None:
        """Redistributes the events records according to the configuration.

        Args:
            redistribution_config: Configuration to redistribute by
        """
        rng = np.random.default_rng(redistribution_config.seed)

        records = self.get_records()

        if records is None:
            raise ValueError("No records to redistribute")

        modification_df = records.df[["record_id", "time"]]

        mode = redistribution_config.mode
        interval = redistribution_config.interval

        if mode == EventRedistributionMode.START_TIME.value:
            modification_df["start"] = interval.start
            modification_df["end"] = interval.end
        else:
            if mode == EventRedistributionMode.CONTAINS_PERCENTAGE:
                beginning_percentile = 0.5 - redistribution_config.percentile / 2.0
                ending_percentile = 0.5 + redistribution_config.percentile / 2.0
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
            'pmt_location_x',
            'pmt_location_y',
            'pmt_location_z'
        ]].rename(
            columns={
                'record_id': 'event_id',
                'pmt_id': 'pmt_idx',
                'module_id': 'module_idx',
                'string_id': 'string_idx',
                'pmt_location_x': 'pmt_x',
                'pmt_location_y': 'pmt_y',
                'pmt_location_z': 'pmt_z',
            }
        )

        orientations = hits.orientations
        new_hits_df['azimuth'] = orientations.phi
        new_hits_df['zenith'] = orientations.theta

        return new_hits_df

    def export(self, collection: Collection, batch_size=100, **kwargs) -> None:
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
            }
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

        os.makedirs(self.file_path, exist_ok=True)

        number_of_records = len(records)
        mc_truths = []
        detector_responses = []
        batch = 0

        # detector_df =

        for index in range(number_of_records):
            current_record = new_records.iloc[index]
            current_record_id = current_record['event_id']

            current_hits = collection.get_hits(record_id=current_record_id)

            mapped_hits = self.__get_mapped_hits_df(current_hits)

            mc_truths.append(current_record)
            detector_responses.append(mapped_hits)

            if (index + 1) % batch_size == 0 or index + 1 == number_of_records:
                array = ak.Array(
                    {
                        'mc_truth': mc_truths,
                        'detector_response': ak.concatenate(detector_responses)
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
            raise ValueError(f'Unsupported file format: {file_path}')
