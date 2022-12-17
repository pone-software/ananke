"""Module containing all utils for the models"""
from __future__ import annotations
import pandas as pd
import numpy as np


def get_repeated_df(
        df_to_repeat: pd.DataFrame,
        number_of_replications: int
) -> pd.DataFrame:
    """Repeats each row x times.

    Args:
        df_to_repeat: Base DataFrame to be repeated
        number_of_replications: How often should each row be replicated?

    Returns:
        New Dataframe with replicated rows
    """
    extended_df = pd.DataFrame(
        np.repeat(df_to_repeat.values, number_of_replications, axis=0)
    )
    extended_df.columns = df_to_repeat.columns
    return extended_df
