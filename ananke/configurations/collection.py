"""Module for all the configuration models of the collection."""
import os
import uuid

from enum import Enum
from typing import Annotated, List, Literal, Optional, Union

import ananke.defaults as defaults

from ananke.configurations.events import Interval, RedistributionConfiguration
from ananke.schemas.event import RecordType
from pydantic import BaseModel, Field, PositiveInt, conint, constr


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
    """Types of storage configurations."""

    HDF5 = "hdf5"
    MEMORY = "memory"


class StorageConfiguration(BaseModel):
    """Base configuration for all collection storages."""

    type: str

    read_only: bool = True

    batch_size: PositiveInt = 100


class MemoryStorageConfiguration(StorageConfiguration):
    """Configuration for collection storage in memory."""

    type: Literal[StorageTypes.MEMORY] = StorageTypes.MEMORY


ComplibConstraintType_ = constr(regex=r"^(zlib|lzo|bzip2|blosc)")


class HDF5StorageConfiguration(StorageConfiguration):
    """Configuration for hdf5 collection storage."""

    type: Literal[StorageTypes.HDF5] = StorageTypes.HDF5

    # TODO: Implement is file check
    #: Path of the HDF5 file
    data_path: str

    #: Compression level; See documentation of `pd.to_hdf`
    complevel: conint(ge=0, lt=10) = 3

    #: Compression library; See documentation of `pd.to_hdf`
    complib: ComplibConstraintType_ = "lzo"

    #: Index optimization level
    optlevel: conint(ge=0, lt=10) = 6


class ExportConfiguration(BaseModel):
    """Base configuration for all exports."""

    pass


class GraphNetExportConfiguration(ExportConfiguration):
    """Configuration for GraphNetExports."""

    data_path: str

    # Currently limited by 32 record ids per batch due to numpy
    # https://github.com/numpy/numpy/issues/4398
    batch_size: PositiveInt = 100


# TODO: Implement Memory Storage
AnnotatedStorageConfiguration = Annotated[
    Union[HDF5StorageConfiguration, MemoryStorageConfiguration],
    Field(discriminator="type"),
]


class MergeConfiguration(BaseModel):
    """Configuration to merge multiple collections into one."""

    in_collections: List[AnnotatedStorageConfiguration]
    tmp_collection: Union[HDF5StorageConfiguration, MemoryStorageConfiguration] = Field(
        discriminator="type",
        default_factory=lambda: HDF5StorageConfiguration(
            data_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../",
                "_tmp_" + str(uuid.uuid4()) + "data.h5",
            ),
            read_only=False,
        ),
    )
    out_collection: AnnotatedStorageConfiguration
    content: Optional[List[MergeContentConfiguration]] = None
    redistribution: Optional[RedistributionConfiguration] = None
    seed: int = defaults.seed
