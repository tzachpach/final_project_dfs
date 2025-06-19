import os
import pickle
import numpy as np
import pandas as pd
from config.dfs_categories import dfs_cats
import shap
import xgboost as xgb

from sklearn.metrics import root_mean_squared_error, mean_absolute_error

###############################################################################
# Helpers
###############################################################################


def bias(y_true, y_pred):
    return float(np.mean(y_pred - y_true))


def compute_overall_metrics(df, categories):
    """
    For each category in 'categories', compute RMSE, MAE, Bias on the entire df.
    Expects df to have columns: cat, cat+'_pred'.
    Returns a dict of metric_name -> value.
    """
    results = {}
    for cat in categories:
        col_actual = cat
        col_pred = cat + "_pred"

        # skip if columns missing
        if col_actual not in df.columns or col_pred not in df.columns:
            results[f"{cat}_RMSE"] = np.nan
            results[f"{cat}_MAE"] = np.nan
            results[f"{cat}_Bias"] = np.nan
            continue

        sub = df.dropna(subset=[col_actual, col_pred])
        if sub.empty:
            results[f"{cat}_RMSE"] = np.nan
            results[f"{cat}_MAE"] = np.nan
            results[f"{cat}_Bias"] = np.nan
        else:
            y_true = sub[col_actual].values
            y_pred = sub[col_pred].values
            results[f"{cat}_RMSE"] = root_mean_squared_error(y_true, y_pred)
            results[f"{cat}_MAE"] = mean_absolute_error(y_true, y_pred)
            results[f"{cat}_Bias"] = bias(y_true, y_pred)

    return results


def compute_percentile_metrics(
    df, categories, percentiles, salary_col="fanduel_salary"
):
    """
    For each p in percentiles (like [10, 20]), we look at top p% salary,
    plus 'all_pop'.
    Returns a DataFrame with columns:
       percentile, [cat_RMSE, cat_MAE, cat_Bias for each category].
    """
    # Ensure we have the needed columns
    needed_cols = []
    for cat in categories:
        needed_cols += [cat, cat + "_pred"]
    needed_cols.append(salary_col)
    df = df.dropna(subset=needed_cols)

    results = []
    for p in percentiles:
        if p == 100:
            # entire population
            label = "all_pop"
            subset = df
        else:
            # top p% means salary >= (100 - p)th percentile
            thresh = np.percentile(df[salary_col], 100 - p)
            label = f"top_{p}"
            subset = df[df[salary_col] >= thresh]

        row = {"percentile": label}
        if subset.empty:
            # fill with NaNs
            for cat in categories:
                row[f"{cat}_RMSE"] = np.nan
                row[f"{cat}_MAE"] = np.nan
                row[f"{cat}_Bias"] = np.nan
            results.append(row)
            continue

        # compute metrics
        for cat in categories:
            col_actual = cat
            col_pred = cat + "_pred"

            tmp = subset.dropna(subset=[col_actual, col_pred])
            if tmp.empty:
                row[f"{cat}_RMSE"] = np.nan
                row[f"{cat}_MAE"] = np.nan
                row[f"{cat}_Bias"] = np.nan
            else:
                y_true = tmp[col_actual].values
                y_pred = tmp[col_pred].values
                row[f"{cat}_RMSE"] = root_mean_squared_error(y_true, y_pred)
                row[f"{cat}_MAE"] = mean_absolute_error(y_true, y_pred)
                row[f"{cat}_Bias"] = bias(y_true, y_pred)

        results.append(row)

    df_percentiles = pd.DataFrame(results)
    return df_percentiles


def evaluate_lineups_vs_contests(
    lineup_df: pd.DataFrame,
    contests_df: pd.DataFrame,
    solver_prefixes=("ga", "ilp", "pulp"),
    platform: str = "fanduel",
):
    """
    Compare (GA / ILP / PULP …) line-ups with contest results *per solver*.

    Returns
    -------
    dict flat { '<solver>_<kpi>': value, ... } + a few global counters
    """

    # ── 1 . Pre-clean  ─────────────────────────────────────────────
    contests_df = contests_df.rename(columns={"period": "game_date"}).copy()
    lineup_df = lineup_df.rename(columns={"date": "game_date"}).copy()

    valid = ["Main", "After Hours", "Express"]
    cdf = contests_df[contests_df["Title"].isin(valid)].loc[
        lambda d: (d["total_entrants"] > 50) & (d["cost"] >= 1)
    ]
    cdf["game_date"] = pd.to_datetime(cdf["game_date"])
    lineup_df["game_date"] = pd.to_datetime(lineup_df["game_date"])

    merged = pd.merge(
        cdf[cdf["game_date"].isin(lineup_df["game_date"])],
        lineup_df,
        on="game_date",
        how="left",
    )

    out = {}
    out["num_contests_raw"] = len(merged)
    total_dup_dropped = 0

    # ── 2 .  Iterate over every requested solver  ─────────────────
    for pref in solver_prefixes:
        pfx = f"{pref}_{platform}"
        cols_needed = {
            "gt_dup": f"{pfx}_GT_duplicates",
            "pd_dup": f"{pfx}_predicted_duplicates",
            "gt_pts": f"{pfx}_GT_points",
            "pd_pts": f"{pfx}_predicted_lineup_GT_points",
        }
        if not all(c in merged.columns for c in cols_needed.values()):
            # nothing logged for this solver → skip
            continue

        # --- filter duplicate line-ups for *this* solver
        df = merged[
            (merged[cols_needed["gt_dup"]] == 0) & (merged[cols_needed["pd_dup"]] == 0)
        ].copy()
        total_dup_dropped += len(merged) - len(df)

        if df.empty:
            for k in ("pred_win_rate", "pred_cash_rate", "total_profit", "avg_profit"):
                out[f"{pref}_{k}"] = np.nan
            continue

        # --- metrics
        df["pred_lineup_would_win"] = df[cols_needed["pd_pts"]] >= df["winning_score"]
        df["pred_lineup_would_cash"] = df[cols_needed["pd_pts"]] >= df["mincash_score"]

        def _profit(row):
            if row["pred_lineup_would_win"]:
                return row["winning_payout"] - row["cost"]
            if row["pred_lineup_would_cash"]:
                return row["mincash_payout"] - row["cost"]
            return -row["cost"]

        df["pred_lineup_profit"] = df.apply(_profit, axis=1)

        out[f"{pref}_pred_win_rate"] = df["pred_lineup_would_win"].mean()
        out[f"{pref}_pred_cash_rate"] = df["pred_lineup_would_cash"].mean()
        out[f"{pref}_total_profit"] = df["pred_lineup_profit"].sum()
        out[f"{pref}_avg_profit"] = df["pred_lineup_profit"].mean()

    out["rows_dropped_dup"] = total_dup_dropped
    return out


def compute_shap_importances(model_pickle_path, df_sample, n_top=10):
    """
    Loads a pickled XGBoost model, runs SHAP on df_sample, returns
    a dict of the top N features -> mean(|SHAP|).
    - df_sample should have the same columns (in the same order) the model expects.
    - This is purely a demonstration snippet. In real usage,
      you must ensure the feature columns match EXACTLY how the model was trained
      (same transformations, dtypes, etc.).
    """

    # Load model
    if not os.path.exists(model_pickle_path):
        return {}

    with open(model_pickle_path, "rb") as f:
        booster = pickle.load(f)
    if not isinstance(booster, xgb.Booster):
        # if you saved the model as an xgb.XGBRegressor or something, adapt accordingly
        # e.g. if "booster" is an XGBRegressor, shap.TreeExplainer(booster)
        return {}

    # Convert df_sample to DMatrix or just a NumPy array
    # But shap needs raw columns typically. We'll do a DMatrix for XGBoost if needed
    # For shap, we actually prefer raw DataFrame to get feature names
    # (Make sure columns match the order used in training.)
    X_sample = df_sample.copy()

    # If the model was purely an xgb.Booster, we might not have 'feature_names' in the right order.
    # Possibly you had code: booster.feature_names = X_sample.columns.tolist(), etc.
    # We'll proceed with a best-effort approach:

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_sample)

    # shap_values is (n_rows, n_features)
    # we compute mean absolute shap per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Pair each feature with its SHAP
    if len(X_sample.columns) != len(mean_abs_shap):
        # mismatch => big trouble
        return {}

    feat_shap_pairs = list(zip(X_sample.columns, mean_abs_shap))
    # Sort descending
    feat_shap_pairs.sort(key=lambda x: x[1], reverse=True)

    # pick top n
    top_pairs = feat_shap_pairs[:n_top]
    # build dict { feature_name: shap_value }
    top_dict = {k: float(v) for (k, v) in top_pairs}

    return top_dict


###############################################################################
# Main user-facing function
###############################################################################
def evaluate_results(
    prediction_df,
    lineup_df,
    contests_df,
    top_percentiles=[20, 10],
    salary_col="fanduel_salary",
    # model_pickle_path=None,
    # shap_data=None,
    # shap_top_n=10
):
    """
    1) Compute overall RMSE/MAE/Bias for each category on entire population.
    2) Compute percentile data (e.g. top_20%, top_10%, + all_pop).
    3) Evaluate lineup vs contests (win rate, profit).
    4) Optionally run SHAP on 'shap_data' with the loaded model to get top N features.

    Returns:
      df_overall_results: 1-row DataFrame with columns:
         - [cat_RMSE, cat_MAE, cat_Bias for each cat],
         - contest metrics,
         - model_pickle_path
      df_percentile_data: multiple rows (one for top_20, one for top_10, one for all_pop),
         each row has columns [percentile, cat_RMSE, cat_MAE, cat_Bias,...]
      top_features_dict: a dict { feature_name: shap_value } for the top N features, or {}
                         if no shap_data or if something fails.
    """

    # 0) Prepare data
    # We'll ensure the needed columns exist in prediction_df
    # Must have cat and cat+"_pred" for each cat, plus 'salary_col' for percentile logic
    # Then we do overall metrics
    categories = [
        cat for cat in dfs_cats + ["fp_fanduel"] if cat in prediction_df.columns
    ]
    overall_stats = compute_overall_metrics(prediction_df, categories)

    # 1) Contest results
    lineup_metrics = evaluate_lineups_vs_contests(lineup_df, contests_df)

    # 2) Combine overall + lineup
    res_dict = {}
    res_dict.update(overall_stats)
    res_dict.update(lineup_metrics)
    # combined_dict["model_pickle"] = os.path.basename(model_pickle_path) if model_pickle_path else "None"

    # 3) Build percentile data
    #    We'll also include p=100 to represent "all_pop"
    p_list = top_percentiles + [100]
    df_percentile_data = compute_percentile_metrics(
        prediction_df, categories, p_list, salary_col
    )

    # 4) If possible, run SHAP
    # top_features_dict = {}
    # if model_pickle_path and shap_data is not None:
    #     try:
    #         top_features_dict = compute_shap_importances(model_pickle_path, shap_data, n_top=shap_top_n)
    #     except Exception as e:
    #         print(f"[WARN] SHAP failed: {e}")
    #         top_features_dict = {}
    # else:
    #     top_features_dict = {}

    return res_dict, df_percentile_data  # , top_features_dict

# trying this script out for better visualization of KPI's of outputs on mlflow and artifacts etc
def format_metrics_for_logging(metrics):
    formatted = {}
    for k, v in metrics.items():
        if k.endswith("_win_rate") or k.endswith("_cash_rate"):
            if v is not None and not pd.isna(v):
                formatted[k] = f"{100 * v:.1f} %"
            else:
                formatted[k] = "N/A"
        elif k.endswith("_total_profit") or k.endswith("_profit"):
            if v is not None and not pd.isna(v):
                formatted[k] = "${:,.0f}".format(v)
            else:
                formatted[k] = "N/A"
        elif "RMSE" in k:
            if v is not None and not pd.isna(v):
                formatted[k] = f"{v:.2f}"
            else:
                formatted[k] = "N/A"
        else:
            formatted[k] = v
    return formatted
