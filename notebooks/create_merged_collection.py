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
                'data_path': '../../data/new_collection/combined_noise_10_test.h5'
            },
        ],
        'out_collection': {
            'type': 'hdf5',
            'data_path': '../../data/new_collection/combined_noise_10_test_2.h5',
            'read_only': False
        },
        'redistribution': {
            'interval': {
                'start': 0,
                'end': 1000
            },
            'mode': 'contains_hit'
        },
        # 'content': None
        # 'content': [
        #     {
        #         'primary_type': RecordType.CASCADE.value,
        #         'secondary_types': [RecordType.ELECTRICAL.value],
        #         'number_of_records': 5,
        #         'interval': {
        #             'start': 0,
        #             'end': 1000
        #         }
        #     },
        #     # {
        #     #     'primary_type': RecordType.ELECTRICAL.value,
        #     #     'number_of_records': 500,
        #     #     'interval': {
        #     #         'start': 0,
        #     #         'end': 1000
        #     #     }
        #     #
        #     # }
        # ]
    }
)

# configuration = MergeConfiguration.parse_obj(
#     {
#         'in_collections': [
#             {
#                 'type': 'hdf5',
#                 'data_path': '../../data/new_collection/combined_10_20_redistributed.h5'
#             }
#         ],
#         'out_collection':
#             {
#                 'type': 'hdf5',
#                 'data_path': '../../data/new_collection/combined_10_20_redistributed_self.h5'
#             },
#         'content': [
#             {
#                 'primary_type': RecordType.CASCADE.value,
#                 'secondary_types': [
#                     RecordType.ELECTRICAL.value
#                 ],
#                 'number_of_records': 500,
#                 'interval': {
#                     'start': 0,
#                     'end': 1000
#                 }
#             },
#             {
#                 'primary_type': RecordType.ELECTRICAL.value,
#                 'number_of_records': 500,
#                 'interval': {
#                     'start': 0,
#                     'end': 1000
#                 }
#
#             }
#         ]
#     }
# )

collection = Collection.from_merge(configuration)

collection.open()
print(len(collection.storage.get_records()))
