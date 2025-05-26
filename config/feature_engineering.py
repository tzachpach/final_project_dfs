import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
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


def fit_feature_reducer(
    X_train: pd.DataFrame,
    y_train,
    *,
    mode,  # 'Kbest' | 'PCA' | None | False
    keep_ratio: float = 0.70,
    cap: int = 300,
    corr_thresh: float = 0.95,
):
    """
    Learn which columns / PCA projection to keep on *training data only*.
    Returns:
        transf : ('cols', list[str]) | ('pca', fitted_PCA)
        aux    : list[str]   #   'pos-' + 'salary' columns to bolt back on
    """
    if not mode:
        return ("cols", X_train.columns.tolist()), []
    # --- numeric & NA mask --------------------------------------------
    X_num = X_train.select_dtypes(include="number")
    y_arr = np.asarray(y_train).ravel()
    mask = (~np.isnan(X_num).any(axis=1)) & (~np.isnan(y_arr))
    X_num = X_num.loc[mask]
    y_arr = y_arr[mask]

    # --- correlation filter ------------------------------------------
    if corr_thresh < 1.0 and X_num.shape[1] > 1:
        c = X_num.corr("spearman").abs()
        upper = c.where(np.triu(np.ones(c.shape), 1).astype(bool))
        drop = [col for col in upper.columns if (upper[col] >= corr_thresh).any()]
        X_num = X_num.drop(columns=drop)

    # --- choose transformer ------------------------------------------
    if mode == "PCA":
        n_comp = max(1, int(0.30 * X_num.shape[1]))
        transf = ("pca", PCA(n_components=n_comp).fit(X_num))
    elif mode == "Kbest":
        k = min(cap, max(1, int(np.ceil(keep_ratio * X_num.shape[1]))))
        skb = SelectKBest(mutual_info_regression, k=k).fit(X_num, y_arr)
        cols = X_num.columns[skb.get_support()].tolist()
        transf = ("cols", cols)
    else:  # mode is False ➜ keep everything numeric
        transf = ("cols", X_num.columns.tolist())

    aux_cols = [c for c in X_train.columns if ("pos-" in c) or ("salary" in c)]
    return transf, aux_cols


def apply_feature_reducer(
    X: pd.DataFrame,
    transf,
    aux_cols: list[str],
) -> pd.DataFrame:
    """
    Project / select on *any* dataframe using the fitted transformer.
    Keeps original row-order; rows that could not be transformed
    (because they had NaNs in numeric cols) are filled with NaNs.
    """
    kind, obj = transf  # ('pca', fitted_PCA) | ('cols', list)
    if kind == "pca":
        # ── select numeric subset expected by PCA ───────────────────────
        X_num = X.select_dtypes(include="number")[obj.feature_names_in_]
        good_mask = ~X_num.isna().any(axis=1)

        # initialise empty frame (all-NaN) with full index
        pca_cols = [f"pca_{i+1}" for i in range(obj.n_components_)]
        X_red = pd.DataFrame(np.nan, index=X.index, columns=pca_cols)

        # run PCA only on valid rows, write back into correct positions
        X_red.loc[good_mask, :] = obj.transform(X_num.loc[good_mask])

    else:  # kind == 'cols'
        X_red = X[obj].copy()

    # ── bolt the always-keep columns back on (pos-* / salary) ────────────
    for c in aux_cols:
        if c not in X_red:  # avoid accidental overwrite
            X_red[c] = X[c].values

    return X_red
