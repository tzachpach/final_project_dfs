import pandas as pd
import os
from datetime import datetime

from config.dfs_categories import same_game_cols, dfs_cats
from config.fantasy_point_calculation import (
    calculate_fp_fanduel,
    calculate_fp_yahoo,
    calculate_fp_draftkings
)
from src.test_train_utils import rolling_train_test_for_xgb

def predict_fp_xgb_q(
    enriched_df,
    mode="daily",
    train_window_days=30,
    train_window_weeks=4,
    salary_thresholds=None,
    save_model=True,
    xgb_param_dict=None
):
    """
    Creates multiple sub-DataFrames based on a descending list of salary quantile thresholds,
    trains a rolling XGBoost model on each sub-DataFrame, and concatenates the predictions.

    Example usage:
        # Suppose each row in enriched_df has a column 'salary_quantile'
        # that indicates the bracket of their current salary (0.95, 0.72, 0.3, etc.)
        # And we want to create 2 bins: top 10% => salary_quantile >= 0.90,
        # second bin => [0.6, 0.90), ignoring everything below 0.6
        # Then call:
        predictions_df = predict_fp_xgb(
            enriched_df,
            mode='daily',
            train_window_days=10,
            train_window_weeks=4,
            salary_thresholds=[0.9, 0.6],
            save_model=False,
            xgb_param_dict={'max_depth': 5, 'learning_rate': 0.05}
        )

    Args:
        enriched_df (pd.DataFrame): The main data set, which must have a 'salary_quantile' column.
        mode (str): 'daily' or 'weekly'. Affects how the rolling is grouped.
        train_window_days (int): Rolling window if mode='daily'.
        train_window_weeks (int): Rolling window if mode='weekly'.
        salary_thresholds (list[float]): Descending list of quantile cutoffs, e.g. [0.9, 0.6, 0.0].
            - Each threshold defines a bin: bin #1 => [thresholds[0], 1.0]
            - bin #2 => [thresholds[1], thresholds[0])
            - bin #3 => [thresholds[2], thresholds[1]) etc.
        save_model (bool): Whether to pickle each bin's model.
        xgb_param_dict (dict): Optional XGBoost hyperparameters to pass to rolling_train_test_for_xgb
                               if your function supports it (otherwise ignored).

    Returns:
        pd.DataFrame: The vertically concatenated predictions for each bin. Each row belongs to
                      one bin in which that player's salary_quantile fell.
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_csv/xgb_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    enriched_df = enriched_df.dropna(subset=['salary_quantile'])
    # Determine grouping + rolling window
    if mode not in ["daily", "weekly"]:
        raise ValueError("mode must be either 'daily' or 'weekly'.")

    group_col = "date" if mode == "daily" else "week"
    rolling_window = train_window_days if mode == "daily" else train_window_weeks

    # If user didn't pass thresholds, default to a single bin: all data
    if not salary_thresholds:
        salary_thresholds = [0.0]   # i.e. everything

    # Ensure the thresholds are in descending order, e.g. [0.9, 0.6, 0.3, 0.0]
    # If the user always passes them in descending, we skip sorting.
    # But let's do a check:
    if any(salary_thresholds[i] < salary_thresholds[i+1] for i in range(len(salary_thresholds)-1)):
        raise ValueError("salary_thresholds must be in descending order, e.g. [0.9, 0.6, 0.0].")

    final_bin_dfs = []  # store partial predictions from each bin

    def train_bin(df_bin, bin_label):
        """Perform rolling train/test for each DFS category on this bin's data."""
        if df_bin.empty:
            print(f"[WARN] Bin '{bin_label}' is empty. Skipping.")
            return pd.DataFrame()

        combined_df = pd.DataFrame()
        # Prepare features for XGBoost (exclude same_game_cols)
        features = df_bin.columns.difference(same_game_cols).tolist()

        print(f"\n=== Training bin '{bin_label}' with {len(df_bin)} rows. ===")
        for cat in dfs_cats:
            X = df_bin[features]
            y = df_bin[cat]
            # If rolling_train_test_for_xgb can handle xgb_param_dict, pass it in.
            # Otherwise we ignore it. We'll show an example ignoring for now:
            cat_results = rolling_train_test_for_xgb(
                X, y, df_bin,
                group_by=group_col,
                train_window=rolling_window,
                save_model=save_model,
                model_dir="models",
                xgb_param_dict=xgb_param_dict,
                output_dir = output_dir,
                quantile_label = bin_label
            )

            cat_results.rename(columns={"y": cat, "y_pred": f"{cat}_pred"}, inplace=True)

            if combined_df.empty:
                combined_df = cat_results
            else:
                combined_df = pd.merge(
                    combined_df,
                    cat_results,
                    on=[
                        'player_name','minutes_played','game_date','game_id',
                        'fanduel_salary','draftkings_salary','yahoo_salary',
                        'fanduel_position','draftkings_position','yahoo_position'
                    ],
                    how='outer',
                    suffixes=('', f'_{cat}')
                )

        # Now compute final fantasy points
        combined_df['fp_fanduel_pred'] = combined_df.apply(lambda row: calculate_fp_fanduel(row, pred_mode=True), axis=1)
        combined_df['fp_yahoo_pred']   = combined_df.apply(lambda row: calculate_fp_yahoo(row, pred_mode=True), axis=1)
        combined_df['fp_draftkings_pred'] = combined_df.apply(lambda row: calculate_fp_draftkings(row, pred_mode=True), axis=1)

        combined_df['fp_fanduel']      = combined_df.apply(calculate_fp_fanduel, axis=1)
        combined_df['fp_yahoo']        = combined_df.apply(calculate_fp_yahoo, axis=1)
        combined_df['fp_draftkings']   = combined_df.apply(calculate_fp_draftkings, axis=1)

        # Optionally add a column indicating the bin label
        combined_df["_bin_label"] = bin_label

        final_output_file = os.path.join(output_dir, "final_fp_xgb.csv")
        final_df.to_csv(final_output_file, index=False)

        return combined_df

    # For i in range(len(thresholds)):
    #   bin i => [ thresholds[i], thresholds[i-1] )  in 'salary_quantile'
    # If i=0 => top bin => [ thresholds[0], 1.0]
    # If we never included 0.0 in the last element, there's no leftover bin below it
    # This is exactly the slicing logic you described:
    #   if thresholds=[0.9,0.6], bin #1 => quantile>=0.9, bin #2 => quantile>=0.6 & <0.9
    for i in range(len(salary_thresholds)):
        lower_q = salary_thresholds[i]
        if i == 0:
            # top bin => salary_quantile >= lower_q
            bin_label = f"bin_top_{lower_q}"
            df_bin = enriched_df[enriched_df['salary_quantile'] >= lower_q].copy()
            part_df = train_bin(df_bin, bin_label)
            final_bin_dfs.append(part_df)
        else:
            # middle bins => [ thresholds[i], thresholds[i-1] )
            higher_q = salary_thresholds[i-1]  # the next bigger threshold
            bin_label = f"bin_{lower_q}_to_{higher_q}"
            df_bin = enriched_df[
                (enriched_df['salary_quantile'] >= lower_q) &
                (enriched_df['salary_quantile'] < higher_q)
            ].copy()
            part_df = train_bin(df_bin, bin_label)
            final_bin_dfs.append(part_df)

    # Combine partial bin predictions
    if not final_bin_dfs:
        print("[WARN] All bins were empty. Returning empty.")
        return pd.DataFrame()
    final_df = pd.concat(final_bin_dfs, ignore_index=True)

    # Save final combined results
    final_output_file = os.path.join(output_dir, "final_fp_xgb.csv")
    final_df.to_csv(final_output_file, index=False)
    print(f"Saved final results to {final_output_file}")

    return final_df
