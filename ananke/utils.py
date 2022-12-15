"""Module containing all utils for the models"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ananke.models.geometry import Vectors3D


def df_columns_to_vectors3d(df: pd.DataFrame, prefix: str = '') -> Vectors3D:
    """Returns a DataFrame of 3d vectors.

    As many of the classes have a prefix in front of the coordinates,
    this method strips the prefix and returns a valid 3dVectors object

    Args:
        df: DataFrame with columns
        prefix: prefix to be stripped

    Returns:
        Valid Vectors3D Object
    """
    mapping = {
        prefix + 'x': 'x',
        prefix + 'y': 'y',
        prefix + 'z': 'z',
    }
    renamed_df = df[mapping.keys()].rename(mapping)
    return Vectors3D(df=renamed_df)


def vectors3d_to_df_columns(vectors3d: Vectors3D, prefix: str = '') -> pd.DataFrame:
    """Gets DataFrame with prefixed columns for later use.

    Args:
        vectors3d: (x,y,z)-columns to prefix
        prefix: prefix to prepend (x,y,z)-columns

    Returns:
        DataFrame with prefixed columns.
    """
    mapping = {
        'x': prefix + 'x',
        'y': prefix + 'y',
        'z': prefix + 'z',
    }
    return vectors3d.df[mapping.keys()].rename(mapping)


def get_repeated_df(df_to_repeat: pd.DataFrame, number_of_replications: int) -> pd.DataFrame:
    """Repeats each row x times.

    Args:
        df_to_repeat: Base DataFrame to be repeated
        number_of_replications: How often should each row be replicated?

    Returns:
        New Dataframe with replicated rows
    """
    extended_df = pd.DataFrame(np.repeat(df_to_repeat.values, number_of_replications, axis=0))
    extended_df.columns = df_to_repeat.columns
    return extended_df
