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


def get_lineup(df, solvers=("GA", "ILP", "MIP", "PULP")):
    """
    solvers: tuple/list like ("GA",) or ("GA","ILP")
    """
    df = df.sort_values("game_date").reset_index(drop=True)
    out_rows = []

    for date in df["game_date"].unique():
        row = {"date": date}
        logging.info(f"Solving {date} with solvers={solvers}")
        for platform in ["fanduel"]:
            for solver in solvers:
                df_g, idx_gt = get_best_lineup(date, df, platform, False, solver)
                df_g, idx = get_best_lineup(
                    date, df, "fanduel", pred_flag=True, solver=["ILP", "GA"]
                )
                if not idx:
                    mlflow.log_metric("lineup_missing", 1)
                    continue

                df_p, idx_pred = get_best_lineup(date, df, platform, True, solver)

                prefix = f"{solver.lower()}_{platform}"
                row.update(
                    {
                        f"{prefix}_player_pool_count": len(df_g),
                        f"{prefix}_GT_players": df_g.loc[
                            idx_gt, "player_name"
                        ].tolist(),
                        f"{prefix}_predicted_players": df_p.loc[
                            idx_pred, "player_name"
                        ].tolist(),
                        f"{prefix}_GT_points": df_g.loc[idx_gt, f"fp_{platform}"].sum(),
                        f"{prefix}_predicted_points": df_p.loc[
                            idx_pred, f"fp_{platform}_pred"
                        ].sum(),
                        f"{prefix}_predicted_lineup_GT_points": df_g.loc[
                            idx_pred, f"fp_{platform}"
                        ].sum(),
                        f"{prefix}_GT_lineup_predicted_points": df_g.loc[
                            idx_gt, f"fp_{platform}_pred"
                        ].sum(),
                        f"{prefix}_GT_salary": df_g.loc[
                            idx_gt, f"{platform}_salary"
                        ].sum(),
                        f"{prefix}_predicted_salary": df_p.loc[
                            idx_pred, f"{platform}_salary"
                        ].sum(),
                        f"{prefix}_GT_duplicates": len(idx_gt) - len(np.unique(idx_gt)),
                        f"{prefix}_predicted_duplicates": len(idx_pred)
                        - len(np.unique(idx_pred)),
                    }
                )

                # overlap metrics
                overlap = set(idx_gt) & set(idx_pred)
                row.update(
                    {
                        f"{prefix}_overlap_players": df_g.loc[
                            list(overlap), "player_name"
                        ].tolist(),
                        f"{prefix}_overlap_count": len(overlap),
                        f"{prefix}_overlap_GT_points": df_g.loc[
                            list(overlap), f"fp_{platform}"
                        ].sum(),
                        f"{prefix}_overlap_predicted_points": df_p.loc[
                            list(overlap), f"fp_{platform}_pred"
                        ].sum(),
                    }
                )
        out_rows.append(row)

    return pd.DataFrame(out_rows)
