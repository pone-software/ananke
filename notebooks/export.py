from ananke.models.collection import Collection, CollectionExporters

collection = Collection(data_path='../../data/merge/noise_only.h5')

collection.export(export_path='../../data/graph_net/noise_random', exporter=CollectionExporters.GRAPH_NET)
