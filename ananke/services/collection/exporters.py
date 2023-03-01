"""Module containing all collection exporters."""
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, Generic

from tqdm import tqdm

from ananke.configurations.collection import ExportConfiguration, GraphNetExportConfiguration

if TYPE_CHECKING:
    from ananke.models.collection import Collection
    from ananke.models.event import Hits

import os
from abc import ABC, abstractmethod
import awkward as ak
import pandas as pd

from ananke.models.geometry import Vectors3D

ExportConfiguration_ = TypeVar('ExportConfiguration_', bound=ExportConfiguration)


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


class GraphNetCollectionExporter(AbstractCollectionExporter[GraphNetExportConfiguration]):
    """Concrete implementation for Graph Net exports."""

    def __get_file_path(self, batch_number: int) -> str:
        """Generates a batches file path.

        Args:
            batch_number: number of the batch of file.

        Returns:
            complete path of current file.
        """
        return os.path.join(self.configuration.data_path, 'batch_{}.parquet'.format(batch_number))

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

        if records is None:
            raise ValueError('Can\'t export empty collection')

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

        os.makedirs(self.configuration.data_path, exist_ok=True)

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

        with tqdm(total=len(new_records), mininterval=0.5) as pbar:

            for index, row in enumerate(new_records.itertuples(index=False)):
                current_record_id = getattr(row, 'event_id')

                current_hits = collection.get_hits(record_ids=current_record_id)

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

                pbar.update()
