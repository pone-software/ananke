from ananke.models.collection import Collection
from ananke.configurations.collection import MergeConfiguration
from ananke.schemas.event import RecordType

configuration = MergeConfiguration.parse_obj(
    {
        'collection_paths': [
            '../../data/merge/data.h5',
        ],
        'out_path': '../../data/merge/merge503.h5',
        'content': [
            {
                'primary_type': RecordType.CASCADE.value,
                'secondary_types': [RecordType.ELECTRICAL.value],
                'number_of_records': 50,
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