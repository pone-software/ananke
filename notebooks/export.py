from ananke.models.collection import Collection, CollectionExporters

collection = Collection(data_path='../../data/merge/merge503.h5')

collection.export(export_path='../../data/graph_net/merge503_graph_net', exporter=CollectionExporters.GRAPH_NET)
