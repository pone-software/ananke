import unittest

from ananke import defaults
from ananke.configurations.events import (
    EventRedistributionMode,
    Interval,
    HistogramConfiguration, RedistributionConfiguration,
)


class EventRedistributionModeTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.names = [
            'START_TIME',
            'CONTAINS_HIT',
            'CONTAINS_EVENT',
            'CONTAINS_PERCENTAGE'
        ]

        self.values = [key.lower() for key in self.names]

    def test_enum(self) -> None:
        self.assertListEqual(
            self.names,
            [e.name for e in EventRedistributionMode],
            "Enum names not as expected."
        )
        self.assertListEqual(
            self.values,
            [e.value for e in EventRedistributionMode],
            "Enum values not as expected."
        )
        self.assertListEqual(
            self.values,
            [e for e in EventRedistributionMode],
            "Enum is string enum not as expected."
        )


class IntervalTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.interval = Interval(start=0, end=1000)

    def test_default(self):
        default_interval = Interval()
        self.assertEqual(0, default_interval.start)
        self.assertEqual(1000, default_interval.end)

    def test_length(self):
        self.assertEqual(1000, self.interval.length)
        self.interval.start = 500
        self.assertEqual(500, self.interval.length)

    def test_range(self):
        self.assertEqual(self.interval.start, self.interval.range[0])
        self.assertEqual(self.interval.end, self.interval.range[1])
        self.assertIsInstance(self.interval.range, tuple)

    def test_is_between(self):
        self.assertTrue(self.interval.is_between(0))
        self.assertTrue(self.interval.is_between(500))
        self.assertFalse(self.interval.is_between(1000))


class HistogramTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.histogram = HistogramConfiguration(start=0, end=1000, bin_size=10)

    def test_default(self):
        default_histogram = HistogramConfiguration()
        self.assertEqual(10, default_histogram.bin_size)

    def test_number_of_bins(self):
        self.assertEqual(100, self.histogram.number_of_bins)


class RedistributionConfigurationTestCase(unittest.TestCase):

    def test_default(self):
        default_redistribution_configuration = RedistributionConfiguration(
            interval=Interval(),
            mode=EventRedistributionMode.CONTAINS_EVENT
        )
        self.assertEqual(defaults.seed, default_redistribution_configuration.seed)
        self.assertEqual(.5, default_redistribution_configuration.percentile)


if __name__ == '__main__':
    unittest.main()
