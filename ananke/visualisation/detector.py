"""Module containing relevant functions for plotting a detector"""

from typing import List
import plotly.graph_objects as go
import numpy as np

from ananke.models.detector import Detector


def get_detector_scatter3ds(detector: Detector, include_pmts: bool = False) -> List[go.Scatter3d]:
    traces = []
    module_coordinates = np.array(detector.module_locations)
    radius = detector.strings[0].modules[0].radius
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
        pmt_coordinates = np.array(detector.pmt_locations)
        traces.append(
            go.Scatter3d(
                x=pmt_coordinates[:, 0],
                y=pmt_coordinates[:, 1],
                z=pmt_coordinates[:, 2],
                name="PMTs",
                mode="markers",
                marker=dict(
                    size=1,
                    color="red",  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.6,
                ),
            ),
        )

    return traces
