"""Module containing all the Services for a detector."""
import itertools
from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import numpy as np
import scipy

from ananke.models.detector import Detector, String, Module, PMT
from ananke.models.geometry import Vector3D
from ananke.schemas.detector import DetectorConfiguration, DetectorGeometries


class AbstractDetectorBuilder(ABC):
    """Detector builder interface.

    Todo:
        - Implement Calibration module
    """

    def __init__(self, configuration: DetectorConfiguration):
        """Constructor of the Detector builder

        Args:
            configuration:
        """
        self.configuration = configuration
        self.rng = np.random.default_rng(configuration.seed)

    @abstractmethod
    def _get_string_locations(self) -> List[Vector3D]:
        """Abstract method supposed to return an array of string locations in child classes.

        Returns:
            List containing the string locations for the detector to build

        """
        raise NotImplementedError('get string locations not implemented')

    def _get_pmts_for_module_location(self, module_location: Vector3D) -> List[PMT]:
        """Build the PMTs for a given module.

        Args:
            module_location: Location of the general for which to generate PMTs

        Returns:
            List containing all PMTs for a given module
        """
        efficiency = self.configuration.pmt.efficiency
        noise_rate = 0

        if self.configuration.pmt.noise_rate > 0 and self.configuration.pmt.gamma_scale > 0:
            noise_rate = scipy.stats.gamma.rvs(1, self.configuration.pmt.gamma_scale,
                                               random_state=self.rng) * self.configuration.pmt.noise_rate

        PMTs = [PMT(ID=0, location=module_location, orientation=Vector3D(x=1, y=0, z=0),
                    efficiency=self.configuration.pmt.efficiency, noise_rate=noise_rate)]
        return PMTs

    def _get_modules_for_string_location(self, string_location: Vector3D) -> List[Module]:
        """Build the modules for a given string.

        Args:
            string_location: location of the string for which to generate modules.

        Returns:
            List containing all Modules for a given string

        """
        string_configuration = self.configuration.string

        modules = []

        for module_ID in range(string_configuration.module_number):
            z_location = module_ID * string_configuration.module_distance + string_configuration.z_offset
            module_location = Vector3D(x=string_location.x, y=string_location.y, z=z_location)
            module = Module(
                ID=module_ID,
                location=module_location,
                PMTs=self._get_pmts_for_module_location(module_location)
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
            modules = self._get_modules_for_string_location(location)
            string = String(
                ID=index,
                location=location,
                modules=self._get_modules_for_string_location(location)
            )
            strings.append(string)

        return strings

    def get(self) -> Detector:
        """Builds a detector based on a given configuration.

        Returns:
            Detector containing all strings and modules

        """
        string_locations = self._get_string_locations()
        return Detector(
            strings=self._get_strings(string_locations=string_locations)
        )


class SingleStringDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for one string detectors."""

    def _get_string_locations(self) -> List[Vector3D]:
        """Get string locations for string detector."""
        position = Vector3D(
            x=self.configuration.geometry.start_position.x,
            y=self.configuration.geometry.start_position.y,
            z=0.0
        )

        return [position]


class TriangularDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for triangular detectors."""

    def _get_string_locations(self) -> List[Vector3D]:
        """Get string locations for triangular detector."""
        side_length = self.configuration.geometry.side_length

        height = np.sqrt(side_length ** 2 - (side_length / 2) ** 2)
        z_position = 0.0
        return [
            Vector3D(
                x=- side_length / 2,
                y=- height / 3,
                z=z_position
            ),
            Vector3D(
                x=+ side_length / 2,
                y=- height / 3,
                z=z_position
            ),
            Vector3D(
                x=0,
                y=+ height * 2 / 3,
                z=z_position
            ),
        ]


class HexagonalDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for hexagonal detectors."""

    def _get_string_locations(self) -> List[Vector3D]:
        """Get string locations for hexagonal detector."""
        string_locations = []
        number_per_side = self.configuration.geometry.number_of_strings_per_side
        distance_between_strings = self.configuration.geometry.distance_between_strings

        for row_index in range(0, number_per_side):
            i_this_row = 2 * (number_per_side - 1) - row_index
            x_positions = np.linspace(
                -(i_this_row - 1) / 2 * distance_between_strings,
                (i_this_row - 1) / 2 * distance_between_strings,
                i_this_row
            )
            y_position = row_index * distance_between_strings * np.sqrt(3) / 2
            for x_position in x_positions:
                string_locations.append(
                    Vector3D(
                        x=x_position,
                        y=y_position,
                        z=0.0
                    )
                )

            if row_index != 0:
                x_positions = np.linspace(
                    -(i_this_row - 1) / 2 * distance_between_strings,
                    (i_this_row - 1) / 2 * distance_between_strings,
                    i_this_row
                )
                y_position = -row_index * distance_between_strings * np.sqrt(3) / 2

                for x_position in x_positions:
                    string_locations.append(
                        Vector3D(
                            x=x_position,
                            y=y_position,
                            z=0.0
                        )
                    )

        return string_locations


class RhombusDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for rhombus detectors."""

    def _get_string_locations(self) -> List[Vector3D]:
        """Get string locations for rhombus detector."""
        side_length = self.configuration.geometry.side_length

        height = np.sqrt(side_length ** 2 - (side_length / 2) ** 2)
        z_position = 0.0
        return [
            Vector3D(
                x=-side_length / 2,
                y=-0.0,
                z=z_position
            ),
            Vector3D(
                x=+side_length / 2,
                y=0.0,
                z=z_position
            ),
            Vector3D(
                x=0,
                y=np.sqrt(3) / 2 * side_length,
                z=z_position
            ),
            Vector3D(
                x=0,
                y=-np.sqrt(3) / 2 * side_length,
                z=z_position
            ),
        ]


class GridDetectorBuilder(AbstractDetectorBuilder):
    """Implementation of the detector builder for grid detectors."""

    def _get_string_locations(self) -> List[Vector3D]:
        """Get string locations for grid detector."""
        string_locations = []
        number_per_side = self.configuration.geometry.number_of_strings_per_side
        distance_between_strings = self.configuration.geometry.distance_between_strings
        x_positions = np.linspace(
            -number_per_side / 2 * distance_between_strings,
            number_per_side / 2 * distance_between_strings,
            number_per_side
        )
        y_positions = x_positions

        for x_position, y_position in itertools.product(x_positions, y_positions):
            string_locations.append(
                Vector3D(
                    x=x_position,
                    y=y_position,
                    z=0.0
                )
            )

        return string_locations


class DetectorBuilderService:
    """Class responsible for building detectors."""

    def __init__(self):
        """Constructor for the DetectorBuilderService."""
        self.__builders = {
            DetectorGeometries.GRID: GridDetectorBuilder,
            DetectorGeometries.SINGLE: SingleStringDetectorBuilder,
            DetectorGeometries.HEXAGONAL: HexagonalDetectorBuilder,
            DetectorGeometries.TRIANGULAR: TriangularDetectorBuilder,
            DetectorGeometries.RHOMBUS: RhombusDetectorBuilder
        }

    def get(self, configuration: DetectorConfiguration) -> Detector:
        """Returns a detector based on a given configuration.

        Args:
            configuration: Configuration to create detector from.

        Returns:
            Detector based on the given configuration.

        """
        builder = self.__builders[configuration.geometry.type](configuration)

        return builder.get()
