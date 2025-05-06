import itertools
from datetime import datetime
import pandas as pd


from config.model_configs import model_configs
from src.data_enrichment import (
    add_anticipated_defense_features,
    add_last_season_data_with_extras,
    add_time_dependent_features_v2,
    add_running_season_stats,
)
from src.evaluate_results import evaluate_results
from src.lineup_genetic_optimizer import get_lineup
from src.predict_fp_rnn_q import predict_fp_rnn_q
from src.predict_fp_xgb_q import predict_fp_xgb_q
from src.preprocessing import merge_all_seasons, preprocess_all_seasons_data


def preprocess_pipeline():
    all_seasons_df = merge_all_seasons()
    return preprocess_all_seasons_data(all_seasons_df)


def enrich_pipeline(df):
    """
    Adds rolling/lags/diffs features (via add_time_dependent_features_v2)
    and merges previous-season data (via add_last_season_data_with_extras).
    Also adds running season stats (via add_running_season_stats).
    """
    # Rolling window stats
    df = add_time_dependent_features_v2(df, rolling_window=10)

    # Running season stats (all within same season)
    df = add_running_season_stats(df)

    # Enrich across seasons
    enriched_seasons = []
    all_seasons = [s for s in df["season_year"].unique()]  # Store full season strings

    for current_season_year in df["season_year"].unique():
        print(f"Processing season year: {current_season_year}")
        season_df = df[df["season_year"] == current_season_year].copy()

        season_start_year = int(current_season_year.split("-")[0])  # Get as integer
        prev_season_start_year = str(season_start_year - 1)
        prev_season_year = f"{prev_season_start_year}-{str(season_start_year)[-2:]}"  # Full prev season

        if prev_season_year in all_seasons:  # Correct comparison
            print(f"Adding stats from previous season: {prev_season_year}")
            prev_season_df = df[df["season_year"] == prev_season_year]
            season_df = add_last_season_data_with_extras(season_df, prev_season_df)
        else:
            print(f"No data for previous season: {prev_season_year}")

        enriched_seasons.append(season_df)

    # Concatenate all enriched season DataFrames
    enriched_df = pd.concat(enriched_seasons, ignore_index=True)

    enriched_df = add_anticipated_defense_features(enriched_df)

    enriched_df = enriched_df.sort_values(["game_date"]).reset_index(drop=True)
    print("All seasons enriched successfully!")

    return enriched_df


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

    else:
        print(f"[WARN] Unknown model_type: {model_type}, returning empty DataFrame.")
        return pd.DataFrame()


def main():
    # Step 1: Preprocess data
    print("Starting pipeline...")
    preprocessed_df = preprocess_pipeline()

    preprocessed_df = preprocessed_df[
        preprocessed_df["season_year"].isin(["2016-17", "2017-18"])
    ]
    preprocessed_df = preprocessed_df.sort_values(["game_date"]).reset_index(drop=True)
    print("Preprocessing completed successfully!")
    # Step 2: Enrich data
    enriched_df = enrich_pipeline(preprocessed_df)
    print("Enrichment completed successfully!")

    contests_df = pd.read_csv("data/contests_data/fanduel_nba_contests.csv")

    # Prepare a list or DataFrame to store results across runs
    all_runs = []
    # 2) Loop over each config
    for cfg_index, cfg in enumerate(model_configs, start=1):
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = cfg["model_type"].lower()

        if model_type == "xgboost":
            # We'll build a cartesian product over the relevant fields in cfg
            # For example:
            thresholds_list = cfg["thresholds"]  # e.g. [[0.9,0.6,0.0], [0.7,0.0], ...]
            mode_list = cfg["mode"]  # e.g. ["daily"]
            tw_days_list = cfg["train_window_days"]  # e.g. [20, 30]
            tw_weeks_list = cfg["train_window_weeks"]  # e.g. [4]
            save_model_list = cfg["save_model"]  # e.g. [True]
            xgb_params_list = cfg["xgb_params"]  # e.g. [ {}, {"max_depth":3}, ...]
            model_dir_list = cfg["model_dir"]  # e.g. ["models"]

            # We produce a product of them. For each combination we call get_predictions_df.
            product_iter = itertools.product(
                thresholds_list,
                mode_list,
                tw_days_list,
                tw_weeks_list,
                save_model_list,
                xgb_params_list,
                model_dir_list,
            )

            for (
                thresholds_val,
                mode_val,
                tw_days_val,
                tw_weeks_val,
                save_model_val,
                xgb_params_val,
                model_dir_val,
            ) in product_iter:

                # Build a sub-config with single, chosen values
                sub_cfg = {
                    "model_type": "XGBoost",
                    "salary_thresholds": thresholds_val,
                    "mode": mode_val,
                    "train_window_days": tw_days_val,
                    "train_window_weeks": tw_weeks_val,
                    "save_model": save_model_val,
                    "xgb_params": xgb_params_val,
                    "model_dir": model_dir_val,
                }

                print(
                    f"\n=== Running XGBoost sub-run: thresholds={thresholds_val}, "
                    f"mode={mode_val}, tw_days={tw_days_val}, xgb_params={xgb_params_val} ==="
                )

                predictions_df = get_predictions_df(sub_cfg, enriched_df)
                if not isinstance(predictions_df, pd.DataFrame) or predictions_df.empty:
                    print(
                        "[WARN] No predictions / empty results for this sub-run. Skipping lineup/eval."
                    )
                    continue

                # lineups + evaluate
                lineup_df = get_lineup(predictions_df)
                res_dict, df_percentiles = evaluate_results(
                    prediction_df=predictions_df,
                    lineup_df=lineup_df,
                    contests_df=contests_df,
                )

                # add sub-run param info to res_dict
                res_dict["cfg_thresholds"] = str(thresholds_val)
                res_dict["cfg_mode"] = mode_val
                res_dict["cfg_tw_days"] = tw_days_val
                # res_dict["cfg_tw_weeks"] = tw_weeks_val
                # res_dict["cfg_save_model"] = save_model_val
                res_dict["cfg_xgb_params"] = str(xgb_params_val)
                # res_dict["cfg_model_dir"] = model_dir_val

                # store or save df_percentiles
                df_percentiles_filename = (
                    f"output_csv/percentiles_xgb_"
                    f"{mode_val}_{tw_days_val}_{thresholds_val}_{xgb_params_val}_{now_str}.csv"
                )
                df_percentiles.to_csv(df_percentiles_filename, index=False)

                all_runs.append(res_dict)

        elif model_type == "rnn":
            # We do the same approach but for the RNN fields
            mode_val_list = cfg["mode"]
            tw_weeks_list = cfg["train_window_weeks"]
            tw_days_list = cfg["train_window_days"]
            hidden_size_list = cfg["hidden_size"]
            num_layers_list = cfg["num_layers"]
            learning_rate_list = cfg["learning_rate"]
            dropout_rate_list = cfg["dropout_rate"]
            epochs_list = cfg["epochs"]
            batch_size_list = cfg["batch_size"]
            rnn_type_list = cfg["rnn_type"]
            salary_thresh_list = cfg["salary_thresholds"]
            multi_target_list = cfg["multi_target_mode"]
            predict_ahead_list = cfg["predict_ahead"]

            product_iter = itertools.product(
                mode_val_list,
                tw_weeks_list,
                tw_days_list,
                hidden_size_list,
                num_layers_list,
                learning_rate_list,
                dropout_rate_list,
                epochs_list,
                batch_size_list,
                rnn_type_list,
                salary_thresh_list,
                multi_target_list,
                predict_ahead_list,
            )

            for (
                mode_val,
                tw_weeks_val,
                tw_days_val,
                hidden_val,
                layer_val,
                lr_val,
                drop_val,
                epoch_val,
                bs_val,
                rnn_type_val,
                sal_thresh_val,
                multi_val,
                pred_ahead_val,
            ) in product_iter:

                sub_cfg = {
                    "model_type": "RNN",
                    "mode": mode_val,
                    "train_window_weeks": tw_weeks_val,  # Changed from "tw_weeks"
                    "train_window_days": tw_days_val,  # Changed from "tw_days"
                    "hidden_size": hidden_val,
                    "num_layers": layer_val,
                    "learning_rate": lr_val,
                    "dropout_rate": drop_val,
                    "epochs": epoch_val,
                    "batch_size": bs_val,
                    "rnn_type": rnn_type_val,
                    "salary_thresholds": sal_thresh_val,
                    "multi_target_mode": multi_val,
                    "predict_ahead": pred_ahead_val,
                }

                print(
                    f"\n=== Running RNN sub-run: tw_weeks={tw_weeks_val}, hidden_size={hidden_val}, "
                    f"layers={layer_val}, lr={lr_val}, drop={drop_val}, epochs={epoch_val}, "
                    f"batch={bs_val}, rnn_type={rnn_type_val}, sal_thresh={sal_thresh_val}, multi={multi_val}, "
                    f"pred_ahead={pred_ahead_val} ==="
                )

                predictions_df = get_predictions_df(sub_cfg, enriched_df)
                if not isinstance(predictions_df, pd.DataFrame) or predictions_df.empty:
                    print(
                        "[WARN] No predictions / empty results for this sub-run. Skipping."
                    )
                    continue

                lineup_df = get_lineup(predictions_df)
                res_dict, df_percentiles = evaluate_results(
                    prediction_df=predictions_df,
                    lineup_df=lineup_df,
                    contests_df=contests_df,
                )

                # store the sub-run param info
                res_dict["cfg_tw_weeks"] = tw_weeks_val
                res_dict["cfg_hidden_size"] = hidden_val
                res_dict["cfg_num_layers"] = layer_val
                res_dict["cfg_learning_rate"] = lr_val
                res_dict["cfg_dropout_rate"] = drop_val
                res_dict["cfg_epochs"] = epoch_val
                res_dict["cfg_batch_size"] = bs_val
                res_dict["cfg_rnn_type"] = rnn_type_val
                res_dict["cfg_salary_thresholds"] = str(sal_thresh_val)
                res_dict["cfg_multi_target"] = multi_val
                res_dict["cfg_predict_ahead"] = pred_ahead_val

                df_percentiles_filename = (
                    f"output_csv/percentiles_rnn_"
                    f"{rnn_type_val}_{tw_weeks_val}_{lr_val}_{drop_val}_{now_str}.csv"
                )
                df_percentiles.to_csv(df_percentiles_filename, index=False)

                all_runs.append(res_dict)

        else:
            print(f"[WARN] Unknown model type: {model_type}. Skipping.")

    # 3.4) Build a master results DF
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_master = pd.DataFrame(all_runs)
    out_path = f"output_csv/master_results_{now_str}.csv"
    df_master.to_csv(out_path, index=False)
    print(f"\nAll runs complete. Master results saved to {out_path}.")


if __name__ == "__main__":
    main()
