"""This module contains all event and photon source related structures."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd

from ananke.configurations.events import (
    EventRedistributionMode,
    RedistributionConfiguration,
)
from ananke.models.detector import Detector
from ananke.models.geometry import OrientedLocatedObjects
from ananke.models.interfaces import DataFrameFacade
from ananke.schemas.event import (
    EventRecordSchema,
    HitSchema,
    NoiseRecordSchema,
    OrientedRecordSchema,
    RecordIdSchema,
    RecordSchema,
    SourceRecordSchema,
    TimedSchema,
)
from ananke.utils import percentile
from pandera.typing import DataFrame


class RecordIds(DataFrameFacade):
    """General description of a record ids."""

    df: DataFrame[RecordIdSchema] = RecordIdSchema.example(size=0)

    def get_by_record(self, record_id: int) -> RecordIds:
        """Gets all sources by a record id.

        Args:
            record_id: ID of the record to get

        Returns:
            Sources of the record
        """
        return self.__class__(df=self.df[self.df["record_id"] == record_id])

    @property
    def record_ids(self) -> pd.Series[int]:
        """Gets all the record ids of the current df."""
        return self.df["record_id"]


class RecordTimes(DataFrameFacade):
    """General description of intervals."""

    df: DataFrame[TimedSchema] = TimedSchema.example(size=0)

    @property
    def times(self) -> pd.Series[float]:
        """Gets DataFrame with all times."""
        return self.df["time"]


class Records(RecordIds, RecordTimes):
    """General description of a record for events or sources."""

    df: DataFrame[RecordSchema] = RecordSchema.example(size=0)


class OrientedRecords(OrientedLocatedObjects, Records):
    """General description of a record for events or sources."""

    df: DataFrame[OrientedRecordSchema] = OrientedRecordSchema.example(size=0)


class Sources(OrientedRecords):
    """Record for a photon source."""

    df: DataFrame[SourceRecordSchema] = SourceRecordSchema.example(size=0)

    # TODO: Fix THis
    # angle_distribution: Optional[npt.ArrayLike] = None

    @property
    def number_of_photons(self) -> pd.Series[int]:
        """Gets DataFrame with all numbers of photons."""
        return self.df["number_of_photons"]


class EventRecords(OrientedRecords):
    """Record of an event that happened."""

    df: DataFrame[EventRecordSchema] = EventRecordSchema.example(size=0)


class NoiseRecords(Records):
    """Record of an event that happened."""

    df: DataFrame[NoiseRecordSchema] = NoiseRecordSchema.example(size=0)


class Hits(Records):
    """Record of an event that happened."""

    df: DataFrame[HitSchema] = HitSchema.example(size=0)


@dataclass
class Collection:
    """Class combining all data frames to a complete record collection."""

    detector: Detector
    records: Records
    hits: Hits
    sources: Optional[Sources] = None

    @classmethod
    def concat(cls, collections_to_concat: List[Collection]) -> Collection:
        """Concatenate multiple connections.

        Args:
            collections_to_concat: List of collections to concat.

        Returns:
            A single collection combining the previous ones.
        """
        if len(collections_to_concat) == 0:
            raise ValueError("You have to pass at least one Collection object in list")
        sources_list: Optional[list] = []
        records_list = []
        hits_list = []
        for collection_to_concat in collections_to_concat:
            if collection_to_concat.sources is not None:
                sources_list.append(collection_to_concat.sources)
            records_list.append(collection_to_concat.records)
            hits_list.append(collection_to_concat.hits)

        if len(sources_list) == 0:
            sources_list = None
        collection = Collection(
            detector=collections_to_concat[0].detector,
            sources=Sources.concat(sources_list),
            records=Records.concat(records_list),
            hits=Hits.concat(hits_list),
        )
        return collection

    def redistribute(self, redistribution_config: RedistributionConfiguration) -> None:
        """Redistributes the events records according to the configuration.

        Args:
            redistribution_config: Configuration to redistribute by
        """
        rng = np.random.default_rng(redistribution_config.seed)

        modification_df = self.records.df[["record_id", "time"]]

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
