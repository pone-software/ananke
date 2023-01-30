"""Module containing all the configuration classes."""

import pandera as pa

from pandera.typing import Series


class PMTSchema(pa.SchemaModel):
    """Schema for the PMT data frame."""

    pmt_id: Series[int] = pa.Field(coerce=True)
    pmt_efficiency: Series[float] = pa.Field(coerce=True)
    pmt_area: Series[float] = pa.Field(coerce=True)
    pmt_noise_rate: Series[float] = pa.Field(coerce=True)
    pmt_location_x: Series[float] = pa.Field(coerce=True)
    pmt_location_y: Series[float] = pa.Field(coerce=True)
    pmt_location_z: Series[float] = pa.Field(coerce=True)
    pmt_orientation_x: Series[float] = pa.Field(coerce=True)
    pmt_orientation_y: Series[float] = pa.Field(coerce=True)
    pmt_orientation_z: Series[float] = pa.Field(coerce=True)


class ModuleSchema(PMTSchema):
    """Schema for the Module data frame."""

    module_id: Series[int] = pa.Field(coerce=True)
    module_radius: Series[float] = pa.Field(coerce=True)
    module_location_x: Series[float] = pa.Field(coerce=True)
    module_location_y: Series[float] = pa.Field(coerce=True)
    module_location_z: Series[float] = pa.Field(coerce=True)


class DetectorSchema(ModuleSchema):
    """Schema for the Detector data frame."""

    string_id: Series[int] = pa.Field(coerce=True)
    string_location_x: Series[float] = pa.Field(coerce=True)
    string_location_y: Series[float] = pa.Field(coerce=True)
    string_location_z: Series[float] = pa.Field(coerce=True)
