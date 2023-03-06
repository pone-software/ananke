import numpy as np
import pandas as pd

from ananke.models.detector import Detector
from ananke.models.event import Records, Hits, Sources
from ananke.schemas.event import RecordType, SourceType


def get_records() -> Records:
    return Records(
        df=pd.DataFrame(
            {
                'type': [
                    0,
                    1,
                    0,
                    1
                ],
                'record_id': [0, 1, 2, 3],
                'time': [0., 5., 10., 0.],
                'duration': [1000, np.nan, 500, 1000]
            }
        )
    )
def get_records_with_particle_ids() -> Records:
    records = get_records()
    records.df['particle_id'] = 4
    return records


def get_hits() -> Hits:
    return Hits(
        df=pd.DataFrame(
            {
                'type': [
                    0,
                    0,
                    1,
                    1,
                ],
                'record_id': [0, 0, 2, 1],
                'time': [0., 5., 10., 0],
                'string_id': [0, 0, 1, 1],
                'module_id': [0, 0, 1, 1],
                'pmt_id': [0, 0, 1, 1],
            }
        )
    )


def get_sources() -> Sources:
    return Sources(
        df=pd.DataFrame(
            {
                'type': [
                    0,
                    0,
                    1,
                    1,
                ],
                'record_id': [0, 0, 2, 1],
                'location_x': [0, 0, 1, 1],
                'location_y': [1, 1, 2, 2],
                'location_z': [1, 2, 3, 4],
                'time': [0., 5., 10., 0],
                'orientation_x': [0, 0, 1, 1],
                'orientation_y': [1, 1, 2, 2],
                'orientation_z': [1, 2, 3, 4],
                'number_of_photons': [100, 200, 300, 400],
            }
        )
    )


def get_detector() -> Detector:
    return Detector(
        df=pd.DataFrame(
            {
                'string_id': [0, 1, 0, 1],
                'string_location_x': [0., 5., 0., 5.],
                'string_location_y': [0., 5., 0., 5.],
                'string_location_z': [0., 5., 0., 5.],
                'module_id': [0, 1, 2, 1],
                'module_location_x': [10., 15.,10., 15.,],
                'module_location_y': [10., 15.,10., 15.,],
                'module_location_z': [10., 15.,10., 15.,],
                'module_radius': 0.5,
                'pmt_id': [0, 0,0,3],
                'pmt_location_x': [20., 25.,20., 25.,],
                'pmt_location_y': [20., 25.,20., 25.,],
                'pmt_location_z': [20., 25.,20., 25.,],
                'pmt_orientation_x': [30., 35.,30., 35.,],
                'pmt_orientation_y': [30., 35.,30., 35.,],
                'pmt_orientation_z': [30., 35.,30., 35.,],
                'pmt_noise_rate': 0.5,
                'pmt_area': 0.5,
                'pmt_efficiency': 0.5,

            }
        )
    )
