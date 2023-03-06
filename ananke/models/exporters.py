"""Module containing all models for exporters."""
from ananke.models.event import RecordIds, Records, RecordTimes
from ananke.schemas.exporters import GraphNetDetectorSchema, GraphNetTruthSchema
from pandera.typing import DataFrame


class GraphNetTruth(Records):
    """Description of the graph net truth."""

    df: DataFrame[GraphNetTruthSchema]


class GraphNetDetector(RecordIds, RecordTimes):
    """Description of the graph net detector."""

    df: DataFrame[GraphNetDetectorSchema]
