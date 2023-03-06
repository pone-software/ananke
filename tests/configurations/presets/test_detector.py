import unittest

import ananke.defaults as defaults

from ananke.configurations.detector import (
    DetectorConfiguration,
    SingleGeometryConfiguration,
)
from ananke.configurations.presets.detector import (
    modules_per_line,
    distance_between_modules,
    dark_noise_rate,
    module_radius,
    pmt_area_radius,
    pmt_efficiency,
    single_line_configuration,
)


class DetectorPresetTestCase(unittest.TestCase):
    def test_default_values(self):
        assert modules_per_line == 20
        assert distance_between_modules == 50
        assert dark_noise_rate == 16 * 1e-5
        assert module_radius == 0.21
        assert pmt_efficiency == 0.42
        assert pmt_area_radius == 75e-3 / 2.0

    def test_single_line_configuration(self):
        self.assertIsInstance(single_line_configuration, DetectorConfiguration)
        self.assertIsInstance(
            single_line_configuration.geometry,
            SingleGeometryConfiguration
        )
        assert single_line_configuration.seed == defaults.seed


if __name__ == '__main__':
    unittest.main()
