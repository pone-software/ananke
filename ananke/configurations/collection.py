"""Module for all the configuration models of the collection."""
from typing import List, Optional

from pydantic import BaseModel, PositiveInt

import ananke.defaults as defaults
from ananke.configurations.events import Interval, RedistributionConfiguration
from ananke.schemas.event import RecordType


class MergeContentConfiguration(BaseModel):
    """Configuration saying what should be in one.

    It first tries to collect the number of records of the primary type. Then it
    mixes in the secondary types. Each type must at least have the number of types
    available.
    """

    #: Types to keep and search for
    primary_type: RecordType

    #: Number of records
    number_of_records: Optional[PositiveInt] = None

    #: Types to mix in
    secondary_types: Optional[List[RecordType]] = None

    #: Interval of hits
    interval: Optional[Interval] = None

    #: Should empty hit records be filtered?
    filter_no_hits: bool = True


class MergeConfiguration(BaseModel):
    """Configuration to merge multiple collections into one"""
    collection_paths: List[str]
    out_path: str
    content: Optional[List[MergeContentConfiguration]] = None
    redistribution: Optional[RedistributionConfiguration] = None
    seed: int = defaults.seed
