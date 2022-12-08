"""Module containing relevant functions for plotting a detector."""

from typing import List

import numpy
import numpy as np
import plotly.graph_objects as go

from ananke.models.detector import Detector


def get_detector_scatter3ds(
    detector: Detector, include_pmts: bool = False
) -> List[go.Scatter3d]:
    """Paint the detectors modules and eventually strings onto a 3d-scatter trace.

    Args:
        detector: Detector to draw
        include_pmts: Want to show the individual PMTs

    Returns:
        List of traces containing module and eventually pmt trace.
    """
    traces = []

    module_coordinates = np.array(
        detector.module_locations
    )  # type: numpy.typing.NDArray[numpy.float64]
    traces.append(
        go.Scatter3d(
            x=module_coordinates[:, 0],
            y=module_coordinates[:, 1],
            z=module_coordinates[:, 2],
            name="modules",
            mode="markers",
            marker=dict(
                size=5,
                color="black",  # set color to an array/list of desired values
                colorscale="Viridis",  # choose a colorscale
                opacity=0.4,
            ),
        ),
    )

    if include_pmts:
        pmt_coordinates = np.array(
            detector.pmt_locations
        )  # type: numpy.typing.NDArray[numpy.float64]
        traces.append(
            go.Scatter3d(
                x=pmt_coordinates[:, 0],
                y=pmt_coordinates[:, 1],
                z=pmt_coordinates[:, 2],
                name="PMTs",
                mode="markers",
                marker=dict(
                    size=5,
                    color="red",  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.6,
                ),
            ),
        )

    return traces
