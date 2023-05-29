from ananke.models.collection import Collection
from ananke.configurations.collection import HDF5StorageConfiguration
from ananke.services.collection.importers import LegacyCollectionImporter

collection_config = HDF5StorageConfiguration(
    data_path='../../data/new_collection/cascades_30000.h5',
    read_only=False
)

collection = Collection(configuration=collection_config)
collection.open()
collection.import_data(
    importer=LegacyCollectionImporter,
    import_path='../../data/cascades_30000/data.h5'
    )
