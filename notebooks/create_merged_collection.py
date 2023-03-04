from ananke.models.collection import Collection
from ananke.configurations.collection import MergeConfiguration
from ananke.schemas.event import RecordType

# configuration = MergeConfiguration.parse_obj(
#     {
#         'collection_paths': [
#             '../../data/merge/combined_noise.h5',
#             '../../data/feasable_cascades_data_all_noise/data.h5'
#         ],
#         'out_path': '../../data/allcombined/data.h5',
#         'content': [
#             # {
#             #     'primary_type': RecordType.ELECTRICAL.value,
#             #     'secondary_types': [RecordType.BIOLUMINESCENCE.value],
#             #     'number_of_records': 10000,
#             #     'interval': {
#             #         'start': 0,
#             #         'end': 1000
#             #     }
#             # },
#             # {
#             #     'primary_type': RecordType.ELECTRICAL.value,
#             #     'number_of_records': 500,
#             #     'interval': {
#             #         'start': 0,
#             #         'end': 1000
#             #     }
#             #
#             # }
#         ]
#     }
# )


configuration = MergeConfiguration.parse_obj(
    {
        'in_collections': [
            {
                'type': 'hdf5',
                'data_path': '../../data/new_collection/combined_10_20_redistributed.h5'
            }
        ],
        'out_collection':
            {
                'type': 'hdf5',
                'data_path': '../../data/combined_10_20/data.h5'
            },
        'content': [
            {
                'primary_type': RecordType.CASCADE.value,
                'secondary_types': [
                    RecordType.ELECTRICAL.value
                ],
                'number_of_records': 500,
                'interval': {
                    'start': 0,
                    'end': 1000
                }
            },
            {
                'primary_type': RecordType.ELECTRICAL.value,
                'number_of_records': 500,
                'interval': {
                    'start': 0,
                    'end': 1000
                }

            }
        ]
    }
)

collection = Collection.from_merge(configuration)

print(len(collection.get_records()))
