"""Module containing all collection exporters."""
from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

from ananke.configurations.collection import (
    ExportConfiguration,
    GraphNetExportConfiguration,
)
from ananke.models.exporters import GraphNetDetector, GraphNetTruth
from ananke.schemas.exporters import GraphNetTruthSchema
from tqdm import tqdm


if TYPE_CHECKING:
    from ananke.models.collection import Collection

import os

from abc import ABC, abstractmethod

import awkward as ak
import pandas as pd

from ananke.models.event import Hits
from ananke.models.geometry import Vectors3D


ExportConfiguration_ = TypeVar("ExportConfiguration_", bound=ExportConfiguration)


class AbstractCollectionExporter(ABC, Generic[ExportConfiguration_]):
    """Abstract parent class for collection exporters."""

    def __init__(self, configuration: ExportConfiguration_):
        """Constructor of the Abstract Collection Exporter.

        Args:
            configuration: Configuration about storage
        """
        self.configuration = configuration

    @abstractmethod
    def export(self, collection: Collection, **kwargs) -> None:
        """Abstract stub for the export of a collection.

        Args:
            collection: Collection to be exported
            kwargs: Additional exporter args
        """
        pass


class GraphNetCollectionExporter(
    AbstractCollectionExporter[GraphNetExportConfiguration]
):
    """Concrete implementation for Graph Net exports."""

    # TODO: Implement sim_type, pid, interaction_type by graph net definitions

    def __get_file_path(self, batch_number: int) -> str:
        """Generates a batches file path.

        Args:
            batch_number: number of the batch of file.

        Returns:
            complete path of current file.
        """
        return os.path.join(
            self.configuration.data_path, "batch_{}.parquet".format(batch_number)
        )

    @staticmethod
    def __get_mapped_hits_df(hits: Hits) -> pd.DataFrame:
        """Return hits mapped to graph nets columns and format.

        Args:
            hits: Hits to map

        Returns:
            Data frame with mapped hits
        """
        new_hits_df = hits.df[["record_id", "pmt_id", "string_id", "module_id", "time"]]

        return new_hits_df

    def export(self, collection: Collection, **kwargs) -> None:
        """Graph net export of a collection.

        Args:
            collection: Collection to be exported
            batch_size: Events per file
            kwargs: Additional exporter args
        """
        records = collection.storage.get_records()

        if records is None:
            raise ValueError("Can't export empty collection")

        # TODO: Properly implement interaction type
        new_records_df = pd.DataFrame(
            {
                "record_id": records.record_ids,
                "time": records.times,
                "type": records.df["type"],
            },
            dtype="int",
        )

        if "orientation_x" in records.df:
            orientations = Vectors3D.from_df(records.df, prefix="orientation_")
            new_records_df["interaction_azimuth"] = orientations.phi
            new_records_df["interaction_zenith"] = orientations.theta

        if "location_x" in records.df:
            new_records_df["interaction_x"] = records.df["location_x"]
            new_records_df["interaction_y"] = records.df["location_y"]
            new_records_df["interaction_z"] = records.df["location_z"]

        if "energy" in records.df:
            new_records_df["energy"] = records.df["energy"]

        if "particle_id" in records.df:
            new_records_df["particle_id"] = records.df["particle_id"]

        mandatory_columns = GraphNetTruthSchema.to_schema().columns

        for mandatory_column in mandatory_columns:
            if mandatory_column not in new_records_df:
                new_records_df[mandatory_column] = -1

        new_records_df.fillna(-1, inplace=True)

        graph_net_truth = GraphNetTruth(df=new_records_df)

        os.makedirs(self.configuration.data_path, exist_ok=True)

        number_of_records = len(graph_net_truth)

        detector = collection.storage.get_detector()

        indices = detector.indices
        orientations = detector.pmt_orientations
        locations = detector.pmt_locations.get_df_with_prefix("pmt_")

        merge_detector_df = pd.concat([indices, locations], axis=1)
        merge_detector_df["pmt_azimuth"] = orientations.phi
        merge_detector_df["pmt_zenith"] = orientations.theta

        batch_size = self.configuration.batch_size

        with tqdm(total=number_of_records, mininterval=0.5) as pbar:
            for index, batch in enumerate(
                graph_net_truth.iterbatches(batch_size=batch_size)
            ):
                current_hits = collection.storage.get_hits(record_ids=batch.record_ids)

                if current_hits is None:
                    continue

                mapped_hits_df = self.__get_mapped_hits_df(current_hits)
                mapped_hits_df = pd.merge(
                    mapped_hits_df,
                    merge_detector_df,
                    how="inner",
                    on=detector.id_columns,
                )

                graph_net_detector = GraphNetDetector(df=mapped_hits_df)
                detector_response = []

                # TODO: Increase performance with GroupBy

                for record_id in batch.record_ids.drop_duplicates():
                    current_hits = graph_net_detector.get_by_record_ids(
                        record_ids=record_id
                    )
                    if current_hits is None:
                        detector_response.append([])
                    else:
                        detector_response.append(
                            current_hits.df.to_dict(orient="records")
                        )
                array = ak.Array(
                    {
                        "mc_truth": batch.df.to_dict(orient="records"),
                        "detector_response": detector_response,
                    }
                )
                ak.to_parquet(array, self.__get_file_path(index), compression="GZIP")

                pbar.update(batch_size)
