from ananke.configurations.events import Interval
from ananke.models.collection import Collection
from ananke.configurations.collection import HDF5StorageConfiguration
from ananke.services.collection.importers import LegacyCollectionImporter


collection_config = HDF5StorageConfiguration(
    data_path='../../data/new_collection/cascades_100.h5',
)

collection = Collection(configuration=collection_config)

with collection:
    print(collection.storage.get_hits(interval=Interval(start=0, end=500)))
#collection.import_data(importer=LegacyCollectionImporter, import_path='../../data/cascades_100/data.h5')