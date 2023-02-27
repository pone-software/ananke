"""Module containing all logic to import collections."""
from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ananke.models.collection import Collection


import os
import pyarrow as pa
from abc import ABC, abstractmethod
from enum import Enum
from typing import Union, Optional


class CollectionImporters(Enum):
    """Enum for possible importers."""
    JULIA_ARROW = 'julia_arrow'


class AbstractCollectionImporter(ABC):
    """Abstract parent class for collection importers."""

    def __init__(
            self,
            collection: Collection
    ):
        """Constructor of the Abstract Collection Importer.

        Args:
            collection: Path to collection or collection
        """
        self.collection = collection

    @abstractmethod
    def import_data(
            self,
            import_path: Union[str, bytes, os.PathLike],
            **kwargs
    ) -> Optional[Collection]:
        """Abstract stub for the import of a collection.

        Args:
            import_path: file path to import
            kwargs: Additional importer args

        Returns:
            Imported collection
        """
        pass


class JuliaArrowCollectionImporter(AbstractCollectionImporter):
    """Concrete implementation for Julia Arrow imports."""

    def __read_file(self, filename: Union[str, bytes, os.PathLike]):

        with pa.ipc.open_file(filename) as reader:
            df = reader.read_pandas()
            print(df.dtypes)


    def import_data(
            self,
            import_path: Union[str, bytes, os.PathLike],
            **kwargs: object
    ) -> Optional[Collection]:
        """Import of a julia arrow collection.

        Args:
            import_path: File path to import
            **kwargs: Additional importer args

        Returns:
            Imported collection
        """
        directory = os.fsencode(import_path)
        for file in os.listdir(directory):
            filename = os.path.join(import_path, os.fsdecode(file))
            if filename.endswith(".arrow"):
                self.__read_file(filename)


class CollectionImporterFactory:
    @staticmethod
    def create_importer(
            collection: Collection,
            importer: CollectionImporters
    ) -> AbstractCollectionImporter:
        """

        Args:
            collection: collection to store imported data in
            importer: importer to choose

        Returns:
            Importer that has been selected
        """
        if importer == CollectionImporters.JULIA_ARROW:
            return JuliaArrowCollectionImporter(collection)
        else:
            raise ValueError(f'Unsupported importer {importer.value}')
