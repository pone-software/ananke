from ananke.configurations.collection import (
    HDF5StorageConfiguration,
    GraphNetExportConfiguration,
)
from ananke.models.collection import Collection
from ananke.services.collection.exporters import GraphNetCollectionExporter

collection = Collection(
    HDF5StorageConfiguration(
        data_path='../../data/new_collection/cascades_10.h5'
    )
)

collection.open()
collection.export(
    GraphNetExportConfiguration(
        data_path='../../data/graph_net/cascades_10',
        batch_size=5
    ),
    exporter=GraphNetCollectionExporter,
)
collection.close()
