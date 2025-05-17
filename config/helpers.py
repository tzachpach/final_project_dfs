import logging

import numpy as np
import pandas as pd
import mlflow

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
        )

        return predictions

    elif model_type.lower() == "rnn":
        # Placeholder: call your RNN method
        # e.g.:
        predictions = predict_fp_rnn_q(
            enriched_df,
            mode=cfg.get("mode", "daily"),
            # train_window_days=cfg.get("train_window_days", 30),
            train_window_weeks=cfg.get("train_window_weeks", 4),
            train_window_days=cfg.get("train_window_days", 12),
            salary_thresholds=cfg.get("salary_thresholds", [0.0]),
            hidden_size=cfg.get("hidden_size", 128),
            num_layers=cfg.get("num_layers", 1),
            learning_rate=cfg.get("learning_rate", 0.001),
            dropout_rate=cfg.get("dropout_rate", 0.0),
            epochs=cfg.get("epochs", 10),
            batch_size=cfg.get("batch_size", 32),
            rnn_type=cfg.get("rnn_type", "LSTM"),
            multi_target_mode=cfg.get("multi_target_mode", False),
            predict_ahead=cfg.get("predict_ahead", 1),
            step_size=cfg.get("step_size", 1),
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
        if date == "2017-11-21":
            # Skip this date for now
            continue
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
