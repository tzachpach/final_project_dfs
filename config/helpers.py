import logging

import numpy as np
import pandas as pd
import mlflow

from config.constants import salary_constraints
from src.lineup_optimizer import get_best_lineup
from src.predict_fp_rnn_q import predict_fp_rnn_q
from src.predict_fp_xgb_q import predict_fp_xgb_q


def get_predictions_df(cfg, enriched_df):
    """
    Based on cfg['model_type'], call the appropriate function to get predictions.
    Return a pd.DataFrame or a list of (param_dict, DataFrame).
    """

    model_type = cfg.get("model_type", "XGBoost")

    if model_type.lower() == "xgboost":
        # Unpack the main fields for xgboost
        predictions = predict_fp_xgb_q(
            enriched_df,
            mode=cfg.get("mode", "daily"),
            train_window_days=cfg.get("train_window_days", 30),
            train_window_weeks=cfg.get("train_window_weeks", 4),
            salary_thresholds=cfg.get("salary_thresholds", [0.9, 0.6, 0.0]),
            save_model=cfg.get("save_model", True),
            xgb_param_dict=cfg.get("xgb_param_dict", {}),
            reduce_features_flag=cfg.get("reduce_features_flag"),
            multi_target_mode=cfg.get("multi_target_mode", False),
        )

        return predictions

    elif model_type.lower() == "rnn":
        # Placeholder: call your RNN method
        # e.g.:
        predictions = predict_fp_rnn_q(
            enriched_df,
            mode=cfg.get("mode", "daily"),
            train_window_days=cfg.get("train_window_days", 12),
            train_window_weeks=cfg.get("train_window_weeks", 4),
            salary_thresholds=cfg.get("salary_thresholds", [0.0]),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 1),
            lookback_weekly=cfg.get("lookback_weekly", 15),
            lookback_daily=cfg.get("lookback_daily", 5),
            learning_rate=cfg.get("learning_rate", 0.001),
            dropout_rate=cfg.get("dropout_rate", 0.0),
            epochs=cfg.get("epochs", 10),
            batch_size=cfg.get("batch_size", 32),
            rnn_type=cfg.get("rnn_type", "LSTM"),
            multi_target_mode=cfg.get("multi_target_mode", False),
            predict_ahead=cfg.get("predict_ahead", 1),
            step_size=cfg.get("step_size", 1),
            reduce_features_flag=cfg.get("reduce_features_flag", True),
            platform=cfg.get("platform", "fanduel"),
        )
        return predictions

    print(f"[WARN] Unknown model_type: {model_type}, returning empty DataFrame.")
    return pd.DataFrame()


def get_lineup(df, solvers=("GA", "ILP", "PULP")):
    """
    solvers: tuple/list like ("GA",) or ("GA","ILP")
    """
    df = df.sort_values("game_date").reset_index(drop=True)
    out_rows = []

    for date in df["game_date"].unique():

        # check that there are enough players for a full roster, given positional constraints
        df_date = df[df["game_date"] == date]
        insufficient_positions = []
        skip_to_next_date = False

        for platform in ["fanduel"]:
            pos_req = salary_constraints[platform]["positions"]
            roster_size = sum(pos_req.values())

            # Check if we have enough players overall
            if len(df_date) < roster_size:
                logging.warning(
                    f"Not enough players ({len(df_date)}) for a full roster ({roster_size}) on {date}"
                )
                skip_to_next_date = True
                break  # Break out of platform loop since we need to skip to next date

            # Check if we have enough players for each position
            position_counts = {pos: 0 for pos in pos_req}
            position_column = f"{platform}_position"

            for _, player in df_date.iterrows():
                if pd.isna(player[position_column]):
                    continue
                positions = str(player[position_column]).split("/")
                for pos in positions:
                    if pos in position_counts:
                        position_counts[pos] += 1
                    # Handle special positions G and F
                    if pos in ("PG", "SG") and "G" in position_counts:
                        position_counts["G"] += 1
                    if pos in ("SF", "PF") and "F" in position_counts:
                        position_counts["F"] += 1

            # Check if each position has enough players
            for pos, required in pos_req.items():
                if pos == "UTIL":  # UTIL can be any position
                    continue
                if position_counts.get(pos, 0) < required + 3:
                    insufficient_positions.append(pos)

        if skip_to_next_date or insufficient_positions:
            logging.warning(
                f"Not enough players for positions {insufficient_positions} on {date}"
            )
            continue  # Skip to next date

        row = {"date": date}
        logging.info(f"Solving {date} with solvers={solvers}")
        for platform in ["fanduel"]:
            for solver in solvers:
                try:
                    # Get ground truth lineup
                    df_gt, idx_gt = get_best_lineup(date, df, platform, False, solver)
                    if not idx_gt:
                        logging.warning(
                            f"No ground truth lineup for {date}/{platform} with {solver}"
                        )
                        continue

                    # Get predicted lineup
                    df_pred, idx_pred = get_best_lineup(
                        date, df, platform, True, solver
                    )
                    if not idx_pred:
                        logging.warning(
                            f"No predicted lineup for {date}/{platform} with {solver}"
                        )
                        mlflow.log_metric("lineup_missing", 1)
                        continue

                    # Make sure indices exist in both DataFrames
                    valid_gt_idx = [i for i in idx_gt if i in df_gt.index]
                    valid_pred_idx = [i for i in idx_pred if i in df_pred.index]

                    if not valid_gt_idx or not valid_pred_idx:
                        logging.warning(
                            f"No valid indices for {date}/{platform} with {solver}"
                        )
                        continue

                    # Use the valid indices for all operations
                    prefix = f"{solver.lower()}_{platform}"
                    row.update(
                        {
                            f"{prefix}_player_pool_count": len(df_gt),
                            f"{prefix}_GT_players": df_gt.loc[
                                valid_gt_idx, "player_name"
                            ].tolist(),
                            f"{prefix}_predicted_players": df_pred.loc[
                                valid_pred_idx, "player_name"
                            ].tolist(),
                            f"{prefix}_GT_points": df_gt.loc[
                                valid_gt_idx, f"fp_{platform}"
                            ].sum(),
                            f"{prefix}_predicted_points": df_pred.loc[
                                valid_pred_idx, f"fp_{platform}_pred"
                            ].sum(),
                            f"{prefix}_predicted_lineup_GT_points": df_gt.loc[
                                valid_pred_idx, f"fp_{platform}"
                            ].sum(),
                            f"{prefix}_GT_lineup_predicted_points": df_gt.loc[
                                valid_gt_idx, f"fp_{platform}_pred"
                            ].sum(),
                            f"{prefix}_GT_salary": df_gt.loc[
                                valid_gt_idx, f"{platform}_salary"
                            ].sum(),
                            f"{prefix}_predicted_salary": df_pred.loc[
                                valid_pred_idx, f"{platform}_salary"
                            ].sum(),
                            f"{prefix}_GT_duplicates": len(valid_gt_idx)
                            - len(np.unique(valid_gt_idx)),
                            f"{prefix}_predicted_duplicates": len(valid_pred_idx)
                            - len(np.unique(valid_pred_idx)),
                        }
                    )

                    # overlap metrics - use valid indices
                    overlap = set(valid_gt_idx) & set(valid_pred_idx)
                    if overlap:
                        row.update(
                            {
                                f"{prefix}_overlap_players": df_gt.loc[
                                    list(overlap), "player_name"
                                ].tolist(),
                                f"{prefix}_overlap_count": len(overlap),
                                f"{prefix}_overlap_GT_points": df_gt.loc[
                                    list(overlap), f"fp_{platform}"
                                ].sum(),
                                f"{prefix}_overlap_predicted_points": df_pred.loc[
                                    list(overlap), f"fp_{platform}_pred"
                                ].sum(),
                            }
                        )
                    else:
                        row.update(
                            {
                                f"{prefix}_overlap_players": [],
                                f"{prefix}_overlap_count": 0,
                                f"{prefix}_overlap_GT_points": 0,
                                f"{prefix}_overlap_predicted_points": 0,
                            }
                        )
                except Exception as e:
                    # Add detailed logging to track exactly when errors occur
                    logging.error(
                        f"Error processing {date}/{platform} with {solver}: {e}"
                    )
                    # Skip this solver and continue with the next one
                    continue

        out_rows.append(row)

    return pd.DataFrame(out_rows)
