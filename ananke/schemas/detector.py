"""Module containing all the configuration classes."""

from pandera.typing import Series
import pandera as pa


class PMTSchema(pa.SchemaModel):
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
    module_id: Series[int] = pa.Field(coerce=True)
    module_radius: Series[float] = pa.Field(coerce=True)
    module_location_x: Series[float] = pa.Field(coerce=True)
    module_location_y: Series[float] = pa.Field(coerce=True)
    module_location_z: Series[float] = pa.Field(coerce=True)


class StringSchema(ModuleSchema):
    string_id: Series[int] = pa.Field(coerce=True)
    string_location_x: Series[float] = pa.Field(coerce=True)
    string_location_y: Series[float] = pa.Field(coerce=True)
    string_location_z: Series[float] = pa.Field(coerce=True)
