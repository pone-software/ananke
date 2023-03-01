"""Module containing relevant functions for plotting a detector."""

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ananke.models.detector import Detector
from ananke.models.event import Sources


def get_detector_scatter3ds(
        detector: Detector,
        include_pmts: bool = False,
        include_modules: bool = True,
        pmt_color: Optional[pd.Series] = None,
        sources: Optional[Sources] = None
) -> List[go.Scatter3d]:
    """Paint the detectors modules and eventually strings onto a 3d-scatter trace.

    Args:
        detector: Detector to draw
        include_pmts: Want to show the individual PMTs

    Returns:
        List of traces containing module and eventually pmt trace.
    """
    traces = []

    if include_modules:
        module_coordinates = detector.module_locations.to_numpy(np.float32)
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
        pmt_coordinates = detector.pmt_locations.to_numpy(np.float32)
        marker = dict(
            size=5,
            color="red",  # set color to an array/list of desired values
            colorscale="Inferno",  # choose a colorscale
            opacity=0.6,
        )
        if pmt_color is not None:
            marker['color'] = pmt_color
            marker['colorbar'] = dict(thickness=20)
        traces.append(
            go.Scatter3d(
                x=pmt_coordinates[:, 0],
                y=pmt_coordinates[:, 1],
                z=pmt_coordinates[:, 2],
                name="PMTs",
                mode="markers",
                marker=marker,
            ),
        )

    if sources is not None:
        source_coordinates = sources.locations.to_numpy(np.float32)
        traces.append(
            go.Scatter3d(
                x=source_coordinates[:, 0],
                y=source_coordinates[:, 1],
                z=source_coordinates[:, 2],
                name="modules",
                mode="markers",
                marker=dict(
                    size=5,
                    color="blue",  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.4,
                ),
            ),
        )


    return traces
