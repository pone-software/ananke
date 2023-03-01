from ananke.models.collection import Collection, CollectionExporters

collection = Collection(data_path='../../data/combined_10_20_redistributed/data.h5')

collection.export(export_path='../../data/graph_net/combined_10_20_redistributed', exporter=CollectionExporters.GRAPH_NET)
