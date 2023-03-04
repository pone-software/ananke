import unittest

import numpy as np
import pandas as pd

from ananke.models.geometry import Vectors3D
from ..common.schemas import get_detector


class DetectorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = get_detector()

    def test_pmt_properties(self):
        self.assertIsInstance(self.detector.pmt_locations, Vectors3D)
        self.assertIsInstance(self.detector.pmt_orientations, Vectors3D)
        self.assertTrue(self.detector.df['pmt_area'].equals(self.detector.pmt_areas))
        self.assertTrue(
            self.detector.df['pmt_efficiency'].equals(self.detector.pmt_efficiencies)
        )
        self.assertTrue(
            self.detector.df['pmt_noise_rate'].equals(self.detector.pmt_noise_rates)
        )
        np.testing.assert_array_equal(
            self.detector.df[[
                'pmt_location_x',
                'pmt_location_y',
                'pmt_location_z',
            ]].to_numpy(),
            self.detector.pmt_locations.to_numpy()
        )
        np.testing.assert_array_equal(
            self.detector.df[[
                'pmt_orientation_x',
                'pmt_orientation_y',
                'pmt_orientation_z',
            ]].to_numpy(),
            self.detector.pmt_orientations.to_numpy()
        )

    def test_modules_properties(self):
        self.assertIsInstance(self.detector.module_locations, Vectors3D)
        self.assertTrue(
            self.detector.df['module_radius'].equals(self.detector.module_radius)
        )
        np.testing.assert_array_equal(
            self.detector.df[[
                'module_location_x',
                'module_location_y',
                'module_location_z',
            ]].to_numpy(),
            self.detector.module_locations.to_numpy()
        )

    def test_string_properties(self):
        self.assertIsInstance(self.detector.string_locations, Vectors3D)
        np.testing.assert_array_equal(
            self.detector.df[[
                'string_location_x',
                'string_location_y',
                'string_location_z',
            ]].to_numpy(),
            self.detector.string_locations.to_numpy()
        )

    def test_detector_numbers(self):
        assert self.detector.number_of_strings == 2
        assert self.detector.number_of_modules == 3
        assert self.detector.number_of_pmts == 4

    def test_detector_indices(self):
        self.assertTrue(pd.DataFrame({
            'string_id': [0,1,0,1],
            'module_id': [0,1,2,1],
            'pmt_id': [0,0,0,3]
        }).equals(self.detector.indices))

    def test_detector_dimensions(self):
        self.assertEqual(self.detector.outer_radius, np.sqrt(3 * 15**2))
        self.assertEqual(
            self.detector.outer_cylinder,
            (np.sqrt(2 * 15**2), 2*15)
        )


if __name__ == '__main__':
    unittest.main()
