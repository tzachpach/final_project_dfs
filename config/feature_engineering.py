import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


def clean_numeric_columns(df, columns):
    """
    Convert columns to numeric, forcing errors to NaN, and handle specific non-numeric values.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert non-numeric to NaN
    return df


class RollingScaler(BaseEstimator, TransformerMixin):
    """
    Fit-on-train, transform-on-both scaler for walk‑forward loops.
    """

    def __init__(self, numeric_cols):
        self.numeric_cols = numeric_cols
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X[self.numeric_cols])
        return self

    def transform(self, X):
        X = X.copy()
        X[self.numeric_cols] = self.scaler.transform(X[self.numeric_cols])
        return X
