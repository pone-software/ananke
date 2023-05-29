"""Module containing functions to display events."""

import numpy as np
import matplotlib.pyplot as plt

from ananke.models.detector import Detector
from ananke.models.event import Hits


def draw_hit_distribution(hits: Hits):
    counts, bins = np.histogram(hits.df['time'],bins=50, range=(0,1000))
    fig, ax = plt.subplots()
    ax.stairs(counts, bins, fill=True)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('Hit count')
    return fig

def draw_hit_histogram(hits: Hits, detector: Detector, colorbar_step=None):
    grouped_dataframe = hits.df.groupby(['string_id', 'module_id', 'pmt_id']).agg(list)[['time']]
    for row in detector.indices.itertuples():
        current_index_tuple = (row.string_id, row.module_id, row.pmt_id)
        try:
            grouped_dataframe.loc[current_index_tuple]
        except:
            grouped_dataframe.at[current_index_tuple, :] = [-1]

    grouped_dataframe = grouped_dataframe.sort_index()

    histrograms = grouped_dataframe['time'].map(lambda x: np.histogram(x, bins=50, range=(0,1000))[0] if x != -1 else np.zeros(50)).reset_index()['time']
    stacked_histograms = np.stack(histrograms)
    fig, ax = plt.subplots()
    c = ax.pcolormesh(stacked_histograms)
    ax.set_xticklabels(np.arange(6)*200)
    ax.set_xlabel('Time [ns]')
    ax.set_ylabel('PMT Number')
    if colorbar_step is None:
        fig.colorbar(c, ax=ax, label='Hit count')
    else:
        fig.colorbar(c, ax=ax, ticks=range(0,int(np.ceil(np.max(stacked_histograms))),colorbar_step), label='Hit count')
    return fig