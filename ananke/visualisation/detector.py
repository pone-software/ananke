"""Module containing relevant functions for plotting a detector."""

from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ananke.models.detector import Detector
from ananke.models.event import Sources, Hits


def get_detector_scatter3ds(
        detector: Detector,
        include_pmts: bool = False,
        include_modules: bool = True,
        hits: Optional[Hits] = None,
        sources: Optional[Sources] = None,
        size: int = 5,
        start_time: float = 0.0,
        end_time: float = 1000.0,
        pmt_opacity: float = 0.6
) -> go.Figure:
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
                    size=size,
                    color="black",  # set color to an array/list of desired values
                    colorscale="Viridis",  # choose a colorscale
                    opacity=0.4,
                ),
            ),
        )

    if include_pmts:
        pmt_coordinates = detector.pmt_locations.to_numpy(np.float32)
        marker = dict(
            size=size,
            color="red",  # set color to an array/list of desired values
            colorscale="Inferno",  # choose a colorscale
            opacity=pmt_opacity
        )
        if hits is not None:
            grouped_hits_dataframe = hits.df \
                .groupby(['string_id', 'module_id', 'pmt_id']).aggregate(
                {
                    'pmt_id': 'count',
                    'time': 'min',
                }
            )
            grouped_hits_dataframe.rename(
                columns={
                    'pmt_id': 'count',
                    'time': 'min_time'
                },
                inplace=True
            )
            for row in detector.indices.itertuples():
                current_index_tuple = (row.string_id, row.module_id, row.pmt_id)
                try:
                    grouped_hits_dataframe.loc[current_index_tuple]
                except:
                    grouped_hits_dataframe.at[current_index_tuple, 'count'] = 0
                    grouped_hits_dataframe.at[current_index_tuple, 'min_time'] = 0
            grouped_hits_dataframe.sort_index(inplace=True)
            marker["colorbar"] = dict(thickness=20)
            marker['cmin'] = start_time
            marker['cmax'] = end_time
            marker['color'] = grouped_hits_dataframe['min_time']
            marker['colorbar'] = dict(
                title=dict(
                    text='Time [ns]'
                )
            )
            marker["size"] = grouped_hits_dataframe['count'] / \
                             grouped_hits_dataframe['count'].max() * 2 * size + size
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
                    size=size / 2.0,
                    color=sources.times,
                    # set color to an array/list of desired values
                    cmin=start_time,
                    cmax=end_time,
                    colorscale=[[0, 'rgba(0,0,255,0.0)'], [1, 'rgba(0,0,255,1.0)']],
                    # choose a colorscale
                    opacity=1,
                    showscale=False
                ),
            ),
        )

    fig = go.Figure(
        data=traces
    )

    camera = dict(
        eye=dict(x=2, y=1, z=1),
    )

    showlegend = False

    if include_pmts and include_modules:
        showlegend = True

    fig.update_layout(
        autosize=False,
        width=600,
        height=600,
        showlegend=showlegend,
        scene=go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='x [m]'),
            yaxis=go.layout.scene.YAxis(title='y [m]'),
            zaxis=go.layout.scene.ZAxis(title='z [m]')
        ),
        template='simple_white',
        scene_camera=camera
    )

    return fig
