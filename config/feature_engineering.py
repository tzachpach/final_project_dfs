import pandas as pd

from config.dfs_categories import same_game_cols


def clean_numeric_columns(df, columns):
    """
    Convert columns to numeric, forcing errors to NaN, and handle specific non-numeric values.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert non-numeric to NaN
    return df