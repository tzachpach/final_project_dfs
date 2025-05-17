import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression


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


def reduce_features(
    X_train: pd.DataFrame,
    y_train,
    keep_ratio: float = 0.70,
    cap: int = 100,
    corr_thresh: float = 0.9,
) -> list[str]:
    """
    1.  Keep only numeric cols.
    2.  Drop rows that have NaN in X *or* y  (index is ignored → no KeyError).
    3.  Optional correlation filter: for any pair |ρ| ≥ corr_thresh
        keep the first and drop the others.
    4.  Rank remaining cols with mutual-information and keep top-k
        where k = ceil(keep_ratio · n_cols) capped at `cap`.

    Returns
    -------
    list[str]   names of the selected numeric features
    """
    # ---------- 1. numeric slice -----------------------------------------
    X_num = X_train.select_dtypes(include="number")
    if X_num.shape[1] == 0:
        raise ValueError("No numeric columns to select from.")

    # ---------- 2. drop rows with NaNs -----------------------------------
    y_arr = np.asarray(y_train).ravel()
    valid_mask = (~np.isnan(X_num).any(axis=1)) & (~np.isnan(y_arr))
    X_num = X_num.loc[valid_mask].reset_index(drop=True)
    y_arr = y_arr[valid_mask]

    # ---------- 3. correlation pruning  ----------------------------------
    if corr_thresh < 1.0:
        corr = X_num.corr(method="spearman").abs()  # robust for funky dists
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if (upper[col] >= corr_thresh).any()]
        X_num = X_num.drop(columns=to_drop)

    # ---------- 4. MI ranking  ------------------------------------------
    k = min(cap, max(1, int(np.ceil(keep_ratio * X_num.shape[1]))))
    selector = SelectKBest(mutual_info_regression, k=k).fit(X_num, y_arr)

    return list(X_num.columns[selector.get_support()])
