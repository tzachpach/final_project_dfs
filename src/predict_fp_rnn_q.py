import pandas as pd
import os
from datetime import datetime

from src.predict_fp_rnn_weekly import rolling_train_test_rnn


def predict_fp_rnn_q(
    df: pd.DataFrame,
    mode: str,
    train_window_days: int,
    train_window_weeks: int,
    salary_thresholds: list,
    hidden_size: int,
    num_layers: int,
    learning_rate: float,
    dropout_rate: float,
    epochs: int,
    batch_size: int,
    rnn_type: str,
    multi_target_mode: bool,
    predict_ahead: int,
    step_size: int,
    platform: str = "fanduel"
):
    """
    Multi-bin RNN predictions based on 'salary_quantile' slicing,
    similar to how your XGBoost multi-bin function works.

    Steps:
      1) For each bin i => [salary_thresholds[i], salary_thresholds[i-1])
         we slice the data to that bin,
         call rolling_train_test_rnn with the chosen RNN hyperparams,
         get partial predictions.
      2) We merge these partial predictions with the original DataFrame
         so that we keep the same columns:
         [player_name, game_id, game_date, minutes_played,
          salary-fanduel, pos-fanduel, etc.].
      3) Concatenate the partial results from all bins.
      4) Return the final merged predictions DataFrame with columns
         [fp_fanduel_pred, fp_fanduel, etc.], plus an optional _bin_label.
      5) Save results to a timestamped folder in output_csv/rnn_{timestamp}/.

    Args:
        df (pd.DataFrame): Must contain 'salary_quantile' for each row, in addition to RNN fields.
        mode (str): 'daily' or 'weekly' for grouping.
        train_window_days (int): Rolling window if mode='daily'.
        train_window_weeks (int): Rolling window if mode='weekly'.
        salary_thresholds (list[float]): descending e.g. [0.9,0.6,0.0].
            If None => single bin includes everything.
        hidden_size (int), num_layers, learning_rate, dropout_rate, epochs, batch_size, rnn_type:
            RNN hyperparams passed into rolling_train_test_rnn.
        multi_target_mode (bool): If True, we do a multicategory approach.
        predict_ahead (int): # of steps to predict forward in sequences.
        step_size (int): step for the rolling loop (e.g. 1 => every group, 2 => skip).
        platform (str): e.g. 'fanduel' => we train to predict fp_fanduel.

    Returns:
        pd.DataFrame: The final DataFrame with partial predictions from all bins,
                      containing columns like [player_name, game_date,
                      fp_<platform>, fp_<platform>_pred, etc.].
                      Also includes _bin_label if you want to see which bin each row came from.
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"output_csv/rnn_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Basic checks
    if mode not in ["daily", "weekly"]:
        raise ValueError("mode must be 'daily' or 'weekly'.")

    if not salary_thresholds:
        salary_thresholds = [0.0]  # one bin => entire data

    # Ensure descending order
    if any(salary_thresholds[i] < salary_thresholds[i+1] for i in range(len(salary_thresholds)-1)):
        raise ValueError("salary_thresholds must be in descending order, e.g. [0.9,0.6,0.0].")

    # We'll unify final results here
    all_bin_results = []

    # We'll define a helper function that:
    # 1) calls rolling_train_test_rnn
    # 2) merges partial predictions with the original columns
    # 3) adds a bin label
    def train_rnn_for_bin(bin_df: pd.DataFrame, bin_label: str) -> pd.DataFrame:
        if bin_df.empty:
            print(f"[WARN] Bin '{bin_label}' is empty. Skipping.")
            return pd.DataFrame()

        print(f"\n=== Training RNN bin '{bin_label}' with {len(bin_df)} rows. ===")

        results_df = rolling_train_test_rnn(
            df=bin_df,
            train_window=(train_window_days if mode=="daily" else train_window_weeks),
            hidden_size=hidden_size,
            num_layers=num_layers,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            epochs=epochs,
            batch_size=batch_size,
            rnn_type=rnn_type,
            multi_target_mode=multi_target_mode,
            group_by=("date" if mode=="daily" else "week"),
            predict_ahead=predict_ahead,
            platform=platform,
            step_size=step_size,
            quantile_label=bin_label,
            output_dir=output_dir  # Pass output directory for saving
        )

        if results_df.empty:
            return pd.DataFrame()

        # Now we want to merge partial results with the original columns
        # so we keep [player_name, game_id, game_date, etc.].
        keep_cols = [
            "player_name", "game_id", "game_date", "minutes_played",
            "salary-fanduel", "salary-draftkings", "salary-yahoo",
            "pos-fanduel", "pos-draftkings", "pos-yahoo"
        ]
        keep_cols = [c for c in keep_cols if c in bin_df.columns]
        df_lookup = bin_df[keep_cols].drop_duplicates(subset=["player_name", "game_id", "game_date"])

        # Now rename the columns in results_df so we have "fp_<platform>" and "fp_<platform>_pred"
        results_df = results_df.rename(columns={
            "y_true": f"fp_{platform}",
            "y_pred": f"fp_{platform}_pred"
        })

        merged_df = pd.merge(
            results_df[["player_name", "game_date", f"fp_{platform}", f"fp_{platform}_pred"]],
            df_lookup,
            on=["player_name", "game_date"],
            how="left"
        )

        # Optionally rename for consistent columns
        renamed = {
            "salary-fanduel": "fanduel_salary",
            "salary-draftkings": "draftkings_salary",
            "salary-yahoo": "yahoo_salary",
            "pos-fanduel": "fanduel_position",
            "pos-draftkings": "draftkings_position",
            "pos-yahoo": "yahoo_position"
        }
        merged_df = merged_df.rename(columns={k: v for k, v in renamed.items() if k in merged_df.columns})

        # Add a bin_label column
        merged_df["_bin_label"] = bin_label

        # Save bin-specific results
        if not multi_target_mode:
            return results_df
        return merged_df

    # We must have a 'salary_quantile' column
    if "salary_quantile" not in df.columns:
        raise ValueError("DataFrame must have 'salary_quantile' column for bin slicing.")

    # For i in range(len(salary_thresholds)):
    #   top bin => [ thresholds[0], 1.0 ]
    #   next => [ thresholds[1], thresholds[0] )
    # etc.
    local_df = df.dropna(subset=["salary_quantile"]).copy()

    for i in range(len(salary_thresholds)):
        lower_q = salary_thresholds[i]
        if i == 0:
            # top bin => quantile >= threshold
            bin_label = f"bin_top_{lower_q}"
            bin_slice = local_df[local_df["salary_quantile"] >= lower_q].copy()
        else:
            higher_q = salary_thresholds[i-1]
            bin_label = f"bin_{lower_q}_to_{higher_q}"
            bin_slice = local_df[
                (local_df["salary_quantile"] >= lower_q) &
                (local_df["salary_quantile"] < higher_q)
            ].copy()

        part_df = train_rnn_for_bin(bin_slice, bin_label)
        if not part_df.empty:
            all_bin_results.append(part_df)

    if not all_bin_results:
        print("[WARN] No bins had data. Returning empty DataFrame.")
        return pd.DataFrame()

    # Vertically concatenate partial results from all bins
    final_df = pd.concat(all_bin_results, ignore_index=True)

    # Save final combined results
    final_output_file = os.path.join(output_dir, f"final_fp_{platform}.csv")
    final_df.to_csv(final_output_file, index=False)
    print(f"Saved final results to {final_output_file}")

    return final_df