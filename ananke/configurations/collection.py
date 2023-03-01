"""Module for all the configuration models of the collection."""
import os
import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, PositiveInt, conint, constr, Field

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


class StorageTypes(str, Enum):
    HDF5 = 'hdf5'


class StorageConfiguration(BaseModel):
    """Base configuration for all collection storages."""

    read_only: bool = False


class HDF5StorageConfiguration(StorageConfiguration):
    """Configuration for hdf5 collection storage"""

    data_path: str

    #: Compression level; See documentation of `pd.to_hdf`
    complevel: conint(ge=0, lt=10) = 3

    #: Compression library; See documentation of `pd.to_hdf`
    complib: constr(regex=r'^(zlib|lzo|bzip2|blosc)') = 'lzo'

    #: Index optimization level
    optlevel: conint(ge=0, lt=10) = 6


class ExportConfiguration(BaseModel):
    """Base configuration for all exports."""
    pass


class GraphNetExportConfiguration:
    """Configuration for GraphNetExports"""

    data_path: str


class MergeConfiguration(BaseModel):
    """Configuration to merge multiple collections into one"""
    in_collections: List[StorageConfiguration]
    tmp_collection: StorageConfiguration = Field(
        default_factory=lambda: HDF5StorageConfiguration(
            data_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '../../_tmp/', '_tmp_' + str(uuid.uuid4()) + 'data.h5')
        )
    )
    out_collection: StorageConfiguration
    content: Optional[List[MergeContentConfiguration]] = None
    redistribution: Optional[RedistributionConfiguration] = None
    seed: int = defaults.seed
