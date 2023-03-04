from ananke.configurations.events import Interval
from ananke.models.collection import Collection
from ananke.configurations.collection import HDF5StorageConfiguration
from ananke.services.collection.importers import LegacyCollectionImporter


collection_config = HDF5StorageConfiguration(
    data_path='../../data/new_collection/combined_10_20_redistributed.h5',
)

collection = Collection(configuration=collection_config)
collection.open()
#collection.import_data(importer=LegacyCollectionImporter, import_path='../../data/combined_10_20_redistributed/data.h5')

collection.g
records = collection.get_records()

print(records)

collection.