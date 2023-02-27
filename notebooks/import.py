"""How to import from a different format."""

from ananke.models.collection import Collection
from ananke.services.collection.importers import CollectionImporters

collection = Collection.import_data(
    import_path='../../data/biolumi_sims',
    collection_path='../../data/import/test.h5',
    importer=CollectionImporters.JULIA_ARROW
)
