"""Module containing all the Services for a detector."""
import itertools

from abc import ABC, abstractmethod
from typing import List, Mapping, Type, Optional

import numpy as np
import scipy

from ananke.models.detector import PMT, Detector, Module, String
from ananke.models.geometry import Vector3D
from ananke.schemas.detector import (
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
    def _get_string_locations(self) -> List[Vector3D]:
        """Abstract method supposed to return an array of string locations.

        Returns:
            List containing the string locations for the detector to build

        Raises:
            NotImplementedError: get string locations not implemented

        """
        raise NotImplementedError("get string locations not implemented")

    def __get_noise_rate_for_pmt(self) -> float:
        """Generates a noise rate based on input or gamma distribution.

        Returns:
            Noise rate for a PMT
        """
        noise_rate = 0

        # Randomize noise level with given parameters
        if (
                self.configuration.pmt.noise_rate > 0
                and self.configuration.pmt.gamma_scale > 0
        ):
            noise_rate = (
                    scipy.stats.gamma.rvs(
                        1, self.configuration.pmt.gamma_scale, random_state=self.rng
                    )
                    * self.configuration.pmt.noise_rate
            )
        return noise_rate

    def _get_pmts_for_module_location(
            self, module_location: Vector3D, module_as_PMT=False
    ) -> List[PMT]:
        """Build the PMTs for a given module.

        The method is as follows. At the moment, we have two layers at each half of
        the module. An inner one and an outer one. Starting from the vertical
        separation ring, the inner PMTs have an angle of 25° and the outer ones one
        of 57.5°. There is four each spread out evenly. The outer and inner PMTs are
        shifted by 45°. The inner modules start at an azimuthal angle of 45°. In total
        16 PMTs are generated as we have two halves

        Args:
            module_location: Location of the general for which to generate PMTs
            module_as_PMT: When there should only be one PMT at module position

        Returns:
            List containing all PMTs for a given module
        """
        # Return only one central PMT if module should not contain any
        if module_as_PMT:
            return [
                PMT(
                    location=module_location,
                    orientation=Vector3D(x=0, y=0, z=0),
                    noise_rate=self.__get_noise_rate_for_pmt(),
                    efficiency=self.configuration.pmt.efficiency,
                    area=self.configuration.pmt.area,
                    ID=0
                )
            ]
        module_radius = self.configuration.module.radius  # TODO: Better place?

        # We start with having the module "flat" as the ring is horizontal.
        # Afterwards we upright the module otherwise my head explodes :D

        orientations = []  # type: List[Vector3D]

        for i in range(8):
            phi = 2 * np.pi / 8 * i
            if i % 2:
                theta = (1 - 57.5 / 90) * np.pi / 2
            else:
                theta = (1 - 25 / 90) * np.pi / 2

            theta_start = np.pi / 2

            original_vector_top = Vector3D.from_spherical(
                module_radius, phi, theta_start + theta
            )
            original_vector_bottom = Vector3D.from_spherical(
                module_radius, phi, theta_start - theta
            )

            # rotate 90° around y-axis (x,y,z) -> (-z, y, x)

            rotated_vector_top = Vector3D(
                x=-original_vector_top.z,
                y=original_vector_top.y,
                z=original_vector_top.x,
            )

            rotated_vector_bottom = Vector3D(
                x=-original_vector_bottom.z,
                y=original_vector_bottom.y,
                z=original_vector_bottom.x,
            )

            orientations.append(rotated_vector_top)
            orientations.append(rotated_vector_bottom)

        PMTs = []

        for ID, orientation in enumerate(orientations):
            PMTs.append(
                PMT(
                    ID=ID,
                    location=module_location,
                    orientation=orientation,
                    efficiency=self.configuration.pmt.efficiency,
                    noise_rate=self.__get_noise_rate_for_pmt(),
                    area=self.configuration.pmt.area
                )
            )

        return PMTs

    def _get_modules_for_string_location(
            self, string_location: Vector3D
    ) -> List[Module]:
        """Build the modules for a given string.

        Args:
            string_location: location of the string for which to generate modules.

        Returns:
            List containing all Modules for a given string

        """
        string_configuration = self.configuration.string

        modules = []

        for module_ID in range(0, string_configuration.module_number):
            z_location = (
                    module_ID * string_configuration.module_distance
                    + string_configuration.z_offset
            )
            module_location = Vector3D(
                x=string_location.x, y=string_location.y, z=z_location
            )
            PMTs = self._get_pmts_for_module_location(
                module_location=module_location,
                module_as_PMT=self.configuration.module.module_as_PMT
            )

            module = Module(
                ID=module_ID,
                location=module_location,
                radius=self.configuration.module.radius,
                PMTs=PMTs,
            )
            modules.append(module)

        return modules

    def _get_strings(self, string_locations: List[Vector3D]) -> List[String]:
        """Build the strings for a detector.

        Args:
            string_locations: List of string locations

        Returns:
            List containing all strings for a set of given string locations

        """
        locations = self._get_string_locations()

        strings = []

        for index, location in enumerate(locations):
            string = String(
                ID=index,
                location=location,
                modules=self._get_modules_for_string_location(location),
            )
            strings.append(string)

        return strings

    def get(self) -> Detector:
        """Builds a detector based on a given configuration.

        Returns:
            Detector containing all strings and modules

        """
        string_locations = self._get_string_locations()
        return self.detector_class(
            strings=self._get_strings(string_locations=string_locations)
        )


class SingleStringDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for one string detectors."""

    def _get_string_locations(self) -> List[Vector3D]:
        """Get string locations for string detector."""
        position = Vector3D(
            x=self.configuration.geometry.start_position.x,
            y=self.configuration.geometry.start_position.y,
            z=0.0,
        )

        return [position]


class TriangularDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for triangular detectors."""

    def _get_string_locations(self) -> List[Vector3D]:
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
        return [
            Vector3D(x=-side_length / 2, y=-height / 3, z=z_position),
            Vector3D(x=+side_length / 2, y=-height / 3, z=z_position),
            Vector3D(x=0, y=+height * 2 / 3, z=z_position),
        ]


class HexagonalDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for hexagonal detectors."""

    def _get_string_locations(self) -> List[Vector3D]:
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

        for row_index in range(0, number_per_side):
            i_this_row = 2 * (number_per_side - 1) - row_index
            x_positions = np.linspace(
                -(i_this_row - 1) / 2 * distance_between_strings,
                (i_this_row - 1) / 2 * distance_between_strings,
                i_this_row,
            )
            y_position = row_index * distance_between_strings * np.sqrt(3) / 2
            for x_position in x_positions:
                string_locations.append(Vector3D(x=x_position, y=y_position, z=0.0))

            if row_index != 0:
                x_positions = np.linspace(
                    -(i_this_row - 1) / 2 * distance_between_strings,
                    (i_this_row - 1) / 2 * distance_between_strings,
                    i_this_row,
                )
                y_position = -row_index * distance_between_strings * np.sqrt(3) / 2

                for x_position in x_positions:
                    string_locations.append(Vector3D(x=x_position, y=y_position, z=0.0))

        return string_locations


class RhombusDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for rhombus detectors."""

    def _get_string_locations(self) -> List[Vector3D]:
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
        return [
            Vector3D(x=-side_length / 2, y=-0.0, z=z_position),
            Vector3D(x=+side_length / 2, y=0.0, z=z_position),
            Vector3D(x=0, y=np.sqrt(3) / 2 * side_length, z=z_position),
            Vector3D(x=0, y=-np.sqrt(3) / 2 * side_length, z=z_position),
        ]


class GridDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for grid detectors."""

    def _get_string_locations(self) -> List[Vector3D]:
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
            string_locations.append(Vector3D(x=x_position, y=y_position, z=0.0))

        return string_locations


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

    def get(self, configuration: DetectorConfiguration) -> Detector:
        """Returns a detector based on a given configuration.

        Args:
            configuration: Configuration to create detector from.

        Returns:
            Detector based on the given configuration.

        """
        builder = self.__builders[configuration.geometry.type](
            configuration=configuration,
            detector_subclass=self.detector_subclass
        )

        return builder.get()
