"""Module containing all utils for the models"""
from __future__ import annotations

from typing import Optional, Callable, Any

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

def percentile(n: float, name: Optional[str] = None) -> Callable[[Any], Any]:
    """Helper function to aggregate DFs bei percentile.

    Args:
        n: percentile to aggregate by
        name: name to override automatically generated one

    Returns:
        callable that aggregates by quantile with a name
    """
    def percentile_(x):
        return x.quantile(n)
    if name is None:
        percentile_.__name__ = 'percentile_{:2.0f}'.format(n*100)
    else:
        percentile_.__name__ = name
    return percentile_