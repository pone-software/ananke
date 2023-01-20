"""Module containing all the Services for a detector."""
import itertools

from abc import ABC, abstractmethod
from typing import Mapping, Type, Optional, TypeVar, Dict, List

import numpy as np
import pandas as pd
from pandera import check_output
from pandera.typing import DataFrame

from ananke.models.detector import Detector
from ananke.models.geometry import Vectors3D
from ananke.utils import get_repeated_df
from ananke.schemas.detector import DetectorSchema
from ananke.configurations.detector import (
    DetectorConfiguration,
    DetectorGeometries,
    LengthGeometryConfiguration,
    SidedGeometryConfiguration,
)


class AbstractDetectorBuilder(ABC):
    """Detector builder interface.

    Todo:
        - Implement Calibration module
    """

    def __init__(
            self,
            configuration: DetectorConfiguration,
            detector_subclass: Optional[Type[Detector]] = None
    ) -> None:
        """Constructor of the Detector builder.

        Args:
            configuration: Configuration to build the detector from.
            detector_subclass: Subclass by which detector should be generated.
        """
        self.configuration = configuration
        self.rng = np.random.default_rng(configuration.seed)

        if detector_subclass is None:
            self.detector_class = Detector
        else:
            self.detector_class = detector_subclass

    @abstractmethod
    def _get_string_locations(self) -> Vectors3D:
        """Abstract method supposed to return an array of string locations.

        Returns:
            List containing the string locations for the detector to build

        Raises:
            NotImplementedError: get string locations not implemented

        """
        raise NotImplementedError("get string locations not implemented")

    def __get_noise_rate_for_pmt(self, number_of_pmts: int) -> float:
        """Generates a noise rate based on input or gamma distribution.

        Args:
            number_of_pmts: Number of noise rates that should be drawn

        Returns:
            Noise rate for a PMT
        """
        noise_rate = self.configuration.pmt.noise_rate

        # Randomize noise level with given parameters
        # TODO: Adapt Noise rate variation
        if (
                noise_rate > 0
                and self.configuration.pmt.gamma_scale > 0
        ):
            noise_rate = (
                    self.rng.gamma(
                        1,
                        self.configuration.pmt.gamma_scale,
                        number_of_pmts
                    )
                    * noise_rate
            )
        return noise_rate

    @staticmethod
    def _get_pmt_orientations(module_as_pmt: bool = False) -> Vectors3D:
        """Gets pmt orientations within a module.

        Args:
            module_as_pmt: Whether each module consists of one pmt only

        Returns:
            Vectors3D of PMT orientations
        """

        orientations = []  # type: List[Dict[str, float]]

        if module_as_pmt:
            orientations.append(
                {
                    'norm': 1.0,
                    'phi': 0.0,
                    'theta': 0.0
                }
            )
        else:
            for i in range(8):
                phi = 2 * np.pi / 8 * i
                if i % 2:
                    theta = (1 - 57.5 / 90) * np.pi / 2
                else:
                    theta = (1 - 25 / 90) * np.pi / 2

                theta_start = np.pi / 2

                orientations.append(
                    {
                        'norm': 1.0,
                        'phi': phi,
                        'theta': theta_start + theta
                    }
                )

                orientations.append(
                    {
                        'norm': 1.0,
                        'phi': phi,
                        'theta': theta_start - theta
                    }
                )
        orientations_df = pd.DataFrame(orientations)

        orientations_vectors = Vectors3D.from_spherical(orientations_df)

        return orientations_vectors

    def _extend_df_by_pmts(
            self, modules_df: pd.DataFrame, module_as_pmt=False
    ) -> pd.DataFrame:
        """Build the PMTs for a given module.

        The method is as follows. At the moment, we have two layers at each half of
        the module. An inner one and an outer one. Starting from the vertical
        separation ring, the inner PMTs have an angle of 25° and the outer ones one
        of 57.5°. There is four each spread out evenly. The outer and inner PMTs are
        shifted by 45°. The inner modules start at an azimuthal angle of 45°. In total
        16 PMTs are generated as we have two halves

        Args:
            modules_df: DataFrame containing module information.
            module_as_pmt: When there should only be one PMT at module position

        Returns:
            DataFrame extended by the PMTs information.
        """
        # get orientations of the pmts
        pmt_vectors = self._get_pmt_orientations(module_as_pmt=module_as_pmt)
        pmt_df = pmt_vectors.get_df_with_prefix('pmt_orientation_')

        # get relevant numbers for replicating rows
        number_of_pmts = len(pmt_df.index)
        number_of_modules = len(modules_df.index)
        total_number_of_rows = number_of_modules * number_of_pmts
        # Assign constant PMT properties
        pmt_df['pmt_id'] = range(number_of_pmts)
        pmt_df['pmt_efficiency'] = self.configuration.pmt.efficiency
        pmt_df['pmt_area'] = self.configuration.pmt.area
        # Scale PMT DataFrame to number of modules
        extended_pmt_df = pd.concat([pmt_df] * number_of_modules)
        extended_pmt_df['pmt_noise_rate'] = self.__get_noise_rate_for_pmt(
            total_number_of_rows
        )

        # Scale module Locations to number of PMTs
        extended_module_locations_df = get_repeated_df(modules_df, number_of_pmts)

        extended_module_locations_df.reset_index(inplace=True, drop=True)
        extended_pmt_df.reset_index(inplace=True, drop=True)

        # Combine both dataframes
        complete_df = pd.concat(
            [extended_module_locations_df, extended_pmt_df],
            axis=1
        )

        # Add PMT Locations

        module_radius = self.configuration.module.radius
        complete_df = complete_df.assign(
            pmt_location_x=lambda
                x: x.module_location_x + module_radius * x.pmt_orientation_x
        )
        complete_df = complete_df.assign(
            pmt_location_y=lambda
                x: x.module_location_y + module_radius * x.pmt_orientation_y
        )
        complete_df = complete_df.assign(
            pmt_location_z=lambda
                x: x.module_location_z + module_radius * x.pmt_orientation_z
        )

        return complete_df

    def _extend_df_by_modules(
            self, strings_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build the modules for a given string.

        Args:
            strings_df: DataFrame containing strings properties

        Returns:
            List containing all Modules for a given string

        """
        string_configuration = self.configuration.string
        number_of_modules = string_configuration.module_number
        extended_strings_df = get_repeated_df(strings_df, number_of_modules)
        total_number_of_modules = len(extended_strings_df.index)
        module_ids = np.array(
            [x % number_of_modules for x in range(total_number_of_modules)],
            dtype=np.int
        )
        extended_strings_df['module_id'] = module_ids
        module_z_locations = module_ids * string_configuration.module_distance + string_configuration.z_offset
        extended_strings_df['module_location_x'] \
            = extended_strings_df.loc[:, 'string_location_x']
        extended_strings_df['module_location_y'] \
            = extended_strings_df.loc[:, 'string_location_y']
        extended_strings_df['module_location_z'] \
            = module_z_locations
        extended_strings_df['module_radius'] = self.configuration.module.radius
        return extended_strings_df

    @check_output(DetectorSchema.to_schema())
    def _get_strings_df(self) -> DataFrame[DetectorSchema]:
        """Build the strings dataframe for a detector.

        Returns:
            DataFrame containing all strings

        """
        location_vectors = self._get_string_locations()
        strings_df = location_vectors.get_df_with_prefix('string_location_')
        strings_df['string_id'] = range(len(location_vectors))

        strings_df = self._extend_df_by_modules(strings_df)
        strings_df = self._extend_df_by_pmts(strings_df)

        return strings_df

    def get(self) -> Detector:
        """Builds a detector based on a given configuration.

        Returns:
            Detector containing all strings and modules

        """
        return self.detector_class(
            df=self._get_strings_df(),
            configuration=self.configuration
        )


class SingleStringDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for one string detectors."""

    def _get_string_locations(self) -> Vectors3D:
        """Get string locations for string detector."""
        positions = pd.DataFrame(
            [{
                'x': 0,
                'y': 0,
                'z': 0
            }]
        )

        return Vectors3D(df=positions)


class TriangularDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for triangular detectors."""

    def _get_string_locations(self) -> Vectors3D:
        """Get string locations for triangular detector.

        Returns:
            Locations of the triangular strings.

        Raises:
            ValueError: Rhombus Geometry needs LengthGeometryConfiguration
        """
        if not isinstance(self.configuration.geometry, LengthGeometryConfiguration):
            raise ValueError("Triangular Geometry needs LengthGeometryConfiguration")
        side_length = self.configuration.geometry.side_length

        height = np.sqrt(side_length ** 2 - (side_length / 2) ** 2)
        z_position = 0.0

        positions = pd.DataFrame(
            [
                {
                    'x': -side_length / 2,
                    'y': -height / 3,
                    'z': z_position
                },
                {
                    'x': side_length / 2,
                    'y': -height / 3,
                    'z': z_position
                },
                {
                    'x': 0,
                    'y': height * 2 / 3,
                    'z': z_position
                },
            ]
        )
        return Vectors3D(df=positions)


class HexagonalDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for hexagonal detectors."""

    def _get_string_locations(self) -> Vectors3D:
        """Get string locations for hexagonal detector.

        Returns:
            Locations of the hexagonal strings.

        Raises:
            ValueError: Hexagonal Geometry needs SidedGeometryConfiguration
        """
        if not isinstance(self.configuration.geometry, SidedGeometryConfiguration):
            raise ValueError("Hexagonal Geometry needs SidedGeometryConfiguration")
        string_locations = []
        number_per_side = self.configuration.geometry.number_of_strings_per_side
        distance_between_strings = self.configuration.geometry.distance_between_strings

        z_position = 0.0

        for row_index in range(0, number_per_side):
            i_this_row = 2 * (number_per_side - 1) - row_index
            x_positions = np.linspace(
                -(i_this_row - 1) / 2 * distance_between_strings,
                (i_this_row - 1) / 2 * distance_between_strings,
                i_this_row,
            )
            y_position = row_index * distance_between_strings * np.sqrt(3) / 2
            for x_position in x_positions:
                string_locations.append(
                    {
                        'x': x_position,
                        'y': y_position,
                        'z': z_position
                    }
                )

            if row_index != 0:
                x_positions = np.linspace(
                    -(i_this_row - 1) / 2 * distance_between_strings,
                    (i_this_row - 1) / 2 * distance_between_strings,
                    i_this_row,
                )
                y_position = -row_index * distance_between_strings * np.sqrt(3) / 2

                for x_position in x_positions:
                    string_locations.append(
                        {
                            'x': x_position,
                            'y': y_position,
                            'z': z_position
                        }
                    )

        return Vectors3D(df=pd.DataFrame(string_locations))


class RhombusDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for rhombus detectors."""

    def _get_string_locations(self) -> Vectors3D:
        """Get string locations for rhombus detector.

        Returns:
            Locations of the rhombus strings.

        Raises:
            ValueError: Rhombus Geometry needs LengthGeometryConfiguration
        """
        if not isinstance(self.configuration.geometry, LengthGeometryConfiguration):
            raise ValueError("Rhombus Geometry needs LengthGeometryConfiguration")
        side_length = self.configuration.geometry.side_length
        z_position = 0.0
        positions = pd.DataFrame(
            [
                {'x': -side_length / 2, 'y': -0.0, 'z': z_position},
                {'x': +side_length / 2, 'y': 0.0, 'z': z_position},
                {'x': 0, 'y': np.sqrt(3) / 2 * side_length, 'z': z_position},
                {'x': 0, 'y': -np.sqrt(3) / 2 * side_length, 'z': z_position},
            ]
        )

        return Vectors3D(df=positions)


class GridDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for grid detectors."""

    def _get_string_locations(self) -> Vectors3D:
        """Get string locations for grid detector.

        Returns:
            Locations of the grid strings.

        Raises:
            ValueError: Rhombus Geometry needs SidedGeometryConfiguration
        """
        if not isinstance(self.configuration.geometry, SidedGeometryConfiguration):
            raise ValueError("Rhombus Geometry needs SidedGeometryConfiguration")
        string_locations = []
        number_per_side = self.configuration.geometry.number_of_strings_per_side
        distance_between_strings = self.configuration.geometry.distance_between_strings
        x_positions = np.linspace(
            -number_per_side / 2 * distance_between_strings,
            number_per_side / 2 * distance_between_strings,
            number_per_side,
        )
        y_positions = x_positions

        for x_position, y_position in itertools.product(x_positions, y_positions):
            string_locations.append({'x': x_position, 'y': y_position, 'z': 0.0})

        return Vectors3D(df=pd.DataFrame(string_locations))


DetectorTypeVar = TypeVar('DetectorTypeVar', bound=Detector)


class DetectorBuilderService:
    """Class responsible for building detectors."""

    def __init__(self, detector_subclass: Optional[Type[Detector]] = None) -> None:
        """Constructor for the DetectorBuilderService.

        Args:
            detector_subclass: Subclass by which detector should be generated.
        """
        self.__builders: Mapping[str, Type[AbstractDetectorBuilder]] = {
            DetectorGeometries.GRID: GridDetectorBuilder,
            DetectorGeometries.SINGLE: SingleStringDetectorBuilder,
            DetectorGeometries.HEXAGONAL: HexagonalDetectorBuilder,
            DetectorGeometries.TRIANGULAR: TriangularDetectorBuilder,
            DetectorGeometries.RHOMBUS: RhombusDetectorBuilder,
        }
        self.detector_subclass = detector_subclass

    def get(self, configuration: DetectorConfiguration) -> DetectorTypeVar:
        """Returns a detector based on a given configuration.

        Args:
            configuration: Configuration to create detector from.

        Returns:
            Detector based on the given configuration.

        Raises:
            ValueError: Configuration must be of type detector configuration

        """
        if not isinstance(configuration, DetectorConfiguration):
            raise ValueError('Configuration must be of type detector configuration')

        builder = self.__builders[configuration.geometry.type](
            configuration=configuration,
            detector_subclass=self.detector_subclass
        )

        return builder.get()
