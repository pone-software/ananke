from enum import Enum
from typing import Tuple
import math

from pydantic import BaseModel

from ..defaults import seed


class EventRedistributionMode(str, Enum):
    """Enum for all the different redistribution modes."""
    # I want the start time be that it is in the interval
    # ..............start....first_hit....last_hit........
    # ...........[..............................]..............
    START_TIME = 'start_time'
    # I want the new start time be that the at least one hit overlaps with the time
    # ....start....first_hit....last_hit........
    # ...........[.................].................
    CONTAINS_HIT = 'containts_hit'
    # I want the new start time be that all hits overlap with the time
    # ....start....first_hit....last_hit........
    # ...........[..............................]....
    CONTAINS_EVENT = 'containts_event'
    # I want the new start time to be that x percent of the event are included
    # I want the new start time be that all hits overlap with the time
    # ....start....first_hit....last_hit........
    # ....................[..............]...........
    CONTAINS_PERCENTAGE = 'containts_percentage'


class Interval(BaseModel):
    """Class defining a basic interval [start, end)."""

    #: Start time of the interval (>=).
    start: float = 0

    #: End time of the interval (<).
    end: float = 1000

    @property
    def range(self) -> Tuple[float, float]:
        """Tuple containing the interval range."""
        return self.start, self.end

    @property
    def length(self) -> float:
        """Represents length of the interval."""
        return self.end - self.start

    def is_between(self, value: float) -> bool:
        """Tells you whether your value is between or outside.

        Args:
            value: Value to check

        Returns: Boolean containing whether value is between start and end.
        """
        left = self.start is not None and value >= self.start
        right = self.end is not None and value < self.end
        return left and right

class HistogramConfiguration(Interval):
    """Subclass of Interval adding tht bin size to configure a histogram."""

    bin_size: int = 10

    @property
    def number_of_bins(self) -> int:
        """Calculate how many bins are between start and end.

        Returns:
            number of bins

        """
        return int(math.ceil(self.length / self.bin_size))


class RedistributionConfiguration(BaseModel):
    """Config to make redistribution standardized."""

    #: Interval in which to redistribute
    interval: Interval

    #: Mode by which to redistribute
    mode: str

    #: Seed for redistribution
    seed: int = seed

    #: Percentile of hits in interval
    percentile: float = .5