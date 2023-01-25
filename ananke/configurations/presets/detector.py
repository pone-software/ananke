"""Module containing configuration presets for the detector."""
from ananke import defaults
from ananke.configurations.detector import DetectorConfiguration

# Before changing, be aware that a lot of already generated examples use this.

modules_per_line = 20
distance_between_modules = 50.0 # m
dark_noise_rate = 16 * 1e-5 # 1/ns
module_radius = 0.21 # m
pmt_efficiency = 0.42 # by Christian S.
pmt_area_radius = 75e-3 / 2.  # m

single_line_configuration = DetectorConfiguration.parse_obj(
    {
        "string": {
            "module_number": modules_per_line,
            "module_distance": distance_between_modules
        },
        "pmt": {
            "efficiency": pmt_efficiency,
            "noise_rate": dark_noise_rate,
            "area": pmt_area_radius
        },
        "module": {
            "radius": module_radius
        },
        "geometry": {
            "type": "single",
        },
        "seed": defaults.seed
    }
)
