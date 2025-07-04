import pandas as pd
import os
from datetime import datetime
import mlflow
import json
from typing import Tuple, List, Dict

from config.constants import PROJECT_ROOT
from src.test_train_utils import prepare_train_test_rnn_data_fixed
from models.tst_model import TimeSeriesTransformer
import torch
import torch.optim as optim
import numpy as np
from functools import reduce
from config.dfs_categories import dfs_cats
from config.fantasy_point_calculation import calculate_fp_fanduel
from config.constants import select_device

import warnings
warnings.filterwarnings("ignore")

def player_key_to_name(player_key: str) -> str:
    """Convert a player key (player_name_TEAM) back to just the player name."""
    return player_key.rsplit("_", 1)[0]

def train_tst_model(
    model,
    X_train,
    y_train,
    epochs,
    batch_size,
    learning_rate,
    device,
    X_val=None,
    y_val=None,
):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    has_val = (X_val is not None) and (y_val is not None)
    if has_val:
        X_v = torch.tensor(X_val, dtype=torch.float32, device=device)
        y_v = torch.tensor(y_val, dtype=torch.float32, device=device)
    for epoch in range(1, epochs + 1):
        model.train()
        indices = np.random.permutation(len(X_t))
        num_batches = int(np.ceil(len(X_t) / batch_size))
        epoch_loss = 0.0
        for b in range(num_batches):
            batch_idx = indices[b * batch_size : (b + 1) * batch_size]
            X_batch = X_t[batch_idx]
            y_batch = y_t[batch_idx]
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if has_val:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_v)
                val_loss = criterion(val_preds, y_v).item()
            print(f"Epoch {epoch}/{epochs}, Train Loss: {epoch_loss/num_batches:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {epoch_loss/num_batches:.4f}")

def rolling_train_test_tst_fixed(
    df: pd.DataFrame,
    train_window: int,
    model_dim: int,
    num_heads: int,
    num_layers: int,
    learning_rate: float,
    dropout_rate: float,
    epochs: int,
    batch_size: int,
    multi_target_mode: bool,
    quantile_label: str,
    reduce_features_flag: bool,
    lookback: int,
    group_by: str = "weekly",
    predict_ahead: int = 1,
    platform: str = "fanduel",
    step_size: int = 1,
    output_dir: str = "output_csv",
    save_csv: bool = True,
):
    if multi_target_mode:
        os.makedirs(output_dir, exist_ok=True)
        all_category_results = []
        for cat in dfs_cats:
            cat_df = df.copy()
            # Rename columns to match expected format
            if f"{cat}_position" in cat_df.columns:
                cat_df[f"pos-{cat}"] = cat_df[f"{cat}_position"]
                cat_df.drop(columns=[f"{cat}_position"], inplace=True)
            if f"{cat}_salary" in cat_df.columns:
                cat_df[f"salary-{cat}"] = cat_df[f"{cat}_salary"]
                cat_df.drop(columns=[f"{cat}_salary"], inplace=True)

            cat_res = rolling_train_test_tst_fixed(
                df=cat_df,
                train_window=train_window,
                model_dim=model_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                epochs=epochs,
                batch_size=batch_size,
                multi_target_mode=False,
                quantile_label=quantile_label,
                reduce_features_flag=reduce_features_flag,
                lookback=lookback,
                group_by=group_by,
                predict_ahead=predict_ahead,
                platform=cat,
                step_size=step_size,
                output_dir=output_dir,
                save_csv=False,
            )
            cat_res = cat_res.rename(
                columns={"y_true": f"{cat}", "y_pred": f"{cat}_pred"}
            )
            out_path = os.path.join(output_dir, f"{cat}_{quantile_label}.csv")
            cat_res.to_csv(out_path, index=False)
            print(f"[multi-target] Saved {out_path}")
            all_category_results.append(cat_res)

        combined = reduce(
            lambda l, r: pd.merge(l, r, on=["player_name", "game_date"], how="outer"),
            all_category_results,
        ).drop_duplicates(["player_name", "game_date"])
        combined["fp_fanduel_pred"] = combined.apply(
            lambda row: calculate_fp_fanduel(row, pred_mode=True), axis=1
        )
        combined["fp_fanduel"] = combined.apply(calculate_fp_fanduel, axis=1)
        combined["game_date"] = pd.to_datetime(combined["game_date"])
        final_path = os.path.join(output_dir, f"fp_fanduel_{quantile_label}.csv")
        combined.to_csv(final_path, index=False)
        print(f"[multi-target] Combined predictions saved → {final_path}")
        return combined

    # Create standardized output directory structure
    if output_dir and save_csv:
        # Build model name from parameters
        model_name = f"tst_{group_by}"
        model_name += f"_tw{train_window}"
        model_name += f"_lb{lookback}"
        model_name += f"_md{model_dim}"
        model_name += f"_h{num_heads}"
        model_name += f"_l{num_layers}"
        model_name += f"_lr{str(learning_rate).replace('.', 'p')}"
        model_name += f"_drop{str(dropout_rate).replace('.', 'p')}"
        model_name += f"_ep{epochs}"
        model_name += f"_b{batch_size}"
        if multi_target_mode:
            model_name += "_multi"
        if reduce_features_flag:
            model_name += f"_feat{reduce_features_flag.lower()}"
        if quantile_label:
            model_name += f"_sal{quantile_label}"
        
        # Create directories
        base_dir = os.path.join("results", model_name)
        models_dir = os.path.join(base_dir, "models")
        predictions_dir = os.path.join(base_dir, "predictions")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(predictions_dir, exist_ok=True)
        
        # Save config
        config = {
            "model_type": "TST",
            "mode": group_by,
            "train_window": train_window,
            "model_dim": model_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "multi_target_mode": multi_target_mode,
            "reduce_features_flag": reduce_features_flag,
            "lookback": lookback,
            "platform": platform,
            "quantile_label": quantile_label
        }
        with open(os.path.join(base_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    device = select_device()
    df = df.copy().dropna()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Rename columns to match expected format
    if f"{platform}_position" in df.columns:
        df[f"pos-{platform}"] = df[f"{platform}_position"]
        df.drop(columns=[f"{platform}_position"], inplace=True)
    if f"{platform}_salary" in df.columns:
        df[f"salary-{platform}"] = df[f"{platform}_salary"]
        df.drop(columns=[f"{platform}_salary"], inplace=True)

    if group_by == "weekly":
        df["group_col"] = (
            df["game_date"].dt.year.astype(str) + "_" +
            df["game_date"].dt.isocalendar().week.astype(str).str.zfill(2)
        )
    elif group_by == "daily":
        df["group_col"] = df["game_date"]
    else:
        raise ValueError("group_by must be 'weekly' or 'daily'.")

    unique_groups = sorted(df["group_col"].unique())
    results = []

    for i in range(train_window, len(unique_groups), step_size):
        current_group = unique_groups[i]
        print(f"\n── step {i}/{len(unique_groups)-1} | test={current_group} ──")

        train_groups = unique_groups[i - train_window : i]
        feature_groups = train_groups + [current_group]

        train_df   = df[df["group_col"].isin(train_groups)].copy()
        feature_df = df[df["group_col"].isin(feature_groups)].copy()
        label_df   = df[df["group_col"] == current_group].copy()

        if train_df.empty or label_df.empty:
            print("   · skipped (empty split)")
            continue

        X_tr, y_tr, X_te, y_te, p_te, d_te, scalers = prepare_train_test_rnn_data_fixed(
            train_df=train_df,
            feature_df=feature_df,
            label_df=label_df,
            target_platform=platform,
            lookback=lookback,
            predict_ahead=predict_ahead,
            reduce_features_flag=reduce_features_flag,
        )

        if len(X_tr) == 0 or len(X_te) == 0:
            print("   · skipped (no valid sequences)")
            continue

        model = TimeSeriesTransformer(
            input_size=X_tr.shape[2],
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout_rate,
            output_size=1,
        )
        train_tst_model(
            model,
            X_tr,
            y_tr,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

        model.eval()
        with torch.no_grad():
            y_pred_s = model(torch.tensor(X_te, dtype=torch.float32, device=device)).cpu().numpy()

        y_pred, y_true = [], []
        for k, ps, ts in zip(p_te, y_pred_s, y_te):
            inv_pred = scalers[k]["y"].inverse_transform(ps.reshape(-1, 1))[0, 0]
            inv_true = scalers[k]["y"].inverse_transform(ts.reshape(-1, 1))[0, 0]
            y_pred.append(inv_pred)
            y_true.append(inv_true)

        step_df = pd.DataFrame({
            "player_name": [player_key_to_name(pk) for pk in p_te],
            "game_date": d_te,
            "y_true": y_true,
            "y_pred": y_pred,
        })
        results.append(step_df)

    results_df = pd.concat(results, ignore_index=True)

    if save_csv:
        os.makedirs(output_dir, exist_ok=True)
        fname = os.path.join(output_dir, f"fp_{platform}_{quantile_label}.csv")
        results_df.to_csv(fname, index=False)
        print(f"[single-target] Saved → {fname}")

    return results_df

def predict_fp_tst_q(
    df: pd.DataFrame,
    mode: str,
    train_window_days: int,
    train_window_weeks: int,
    lookback_daily: int,
    lookback_weekly: int,
    salary_thresholds: list,
    multi_target_mode: bool,
    predict_ahead: int,
    step_size: int,
    reduce_features_flag: bool,
    tst_config: dict,
    platform: str = "fanduel",
    save_model: bool = True,
    run_name: str = None,
):
    """
    Creates multiple sub-DataFrames based on a descending list of salary quantile thresholds,
    trains a rolling TST model on each sub-DataFrame, and concatenates the predictions.
    """
    if not salary_thresholds:
        salary_thresholds = [0.0]  # Default: use all players

    # Determine lookback based on mode
    lookback = lookback_daily if mode == "daily" else lookback_weekly

    # Use provided run_name or generate one
    if run_name is None:
        run_name = f"tst_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    output_dir = os.path.join("results", run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config = {
        "model_type": "TST",
        "mode": mode,
        "train_window_days": train_window_days,
        "train_window_weeks": train_window_weeks,
        "lookback_daily": lookback_daily,
        "lookback_weekly": lookback_weekly,
        "salary_thresholds": salary_thresholds,
        "multi_target_mode": multi_target_mode,
        "predict_ahead": predict_ahead,
        "step_size": step_size,
        "reduce_features_flag": reduce_features_flag,
        "tst_config": tst_config,
        "platform": platform,
        "save_model": save_model
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Create model and predictions directories
    models_dir = os.path.join(output_dir, "models")
    predictions_dir = os.path.join(output_dir, "predictions")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)

    # Prepare data
    local_df = df.copy()
    local_df["salary_quantile"] = local_df.groupby("game_date")["salary-fanduel"].transform(
        lambda x: x.rank(pct=True)
    )

    all_bin_results = []

    def train_tst_for_bin(bin_df: pd.DataFrame, bin_label: str) -> pd.DataFrame:
        if bin_df.empty:
            print(f"[WARN] Bin '{bin_label}' is empty. Skipping.")
            return pd.DataFrame()
        print(f"\n=== Training TST bin '{bin_label}' with {len(bin_df)} rows. ===")

        # Rename columns to match expected format
        if f"{platform}_position" in bin_df.columns:
            bin_df[f"pos-{platform}"] = bin_df[f"{platform}_position"]
            bin_df.drop(columns=[f"{platform}_position"], inplace=True)
        if f"{platform}_salary" in bin_df.columns:
            bin_df[f"salary-{platform}"] = bin_df[f"{platform}_salary"]
            bin_df.drop(columns=[f"{platform}_salary"], inplace=True)

        results_df = rolling_train_test_tst_fixed(
            df=bin_df,
            train_window=(train_window_days if mode == "daily" else train_window_weeks),
            model_dim=tst_config["model_dim"],
            num_heads=tst_config["num_heads"],
            num_layers=tst_config["num_layers"],
            learning_rate=tst_config["learning_rate"],
            dropout_rate=tst_config["dropout"],
            epochs=tst_config["epochs"],
            batch_size=tst_config["batch_size"],
            multi_target_mode=multi_target_mode,
            group_by=mode,
            lookback=lookback,
            predict_ahead=predict_ahead,
            platform=platform,
            step_size=step_size,
            quantile_label=bin_label,
            output_dir=predictions_dir,
            reduce_features_flag=reduce_features_flag,
        )

        if results_df.empty:
            return pd.DataFrame()

        # Now we want to merge partial results with the original columns
        # so we keep [player_name, game_id, game_date, etc.].
        keep_cols = [
            "player_name",
            "game_id",
            "game_date",
            "minutes_played",
            "team_abbreviation",
            "salary-fanduel",
            "salary-draftkings",
            "salary-yahoo",
            "pos-fanduel",
            "pos-draftkings",
            "pos-yahoo",
        ]
        keep_cols = [c for c in keep_cols if c in bin_df.columns]
        df_lookup = bin_df[keep_cols].drop_duplicates(
            subset=["player_name", "team_abbreviation", "game_id", "game_date"]
        )

        # Now rename the columns in results_df so we have "fp_<platform>" and "fp_<platform>_pred"
        results_df = results_df.rename(
            columns={"y_true": f"fp_{platform}", "y_pred": f"fp_{platform}_pred"}
        )

        merged_df = pd.merge(
            results_df[
                ["player_name", "game_date", f"fp_{platform}", f"fp_{platform}_pred"]
            ],
            df_lookup,
            on=["player_name", "game_date"],
            how="left",
        )

        # Add a bin_label column
        merged_df["_bin_label"] = bin_label

        # Save bin results
        bin_output_file = os.path.join(predictions_dir, f"fp_{platform}_{bin_label}.csv")
        merged_df.to_csv(bin_output_file, index=False)
        print(f"[INFO] Saved bin results to {bin_output_file}")

        return merged_df

    # Process each salary threshold bin
    for i in range(len(salary_thresholds)):
        lower_q = salary_thresholds[i]
        if i == 0:
            # top bin => quantile >= threshold
            bin_label = f"bin_top_{lower_q}"
            bin_slice = local_df[local_df["salary_quantile"] >= lower_q].copy()
        else:
            higher_q = salary_thresholds[i - 1]
            bin_label = f"bin_{lower_q}_to_{higher_q}"
            bin_slice = local_df[
                (local_df["salary_quantile"] >= lower_q)
                & (local_df["salary_quantile"] < higher_q)
            ].copy()

        part_df = train_tst_for_bin(bin_slice, bin_label)
        if not part_df.empty:
            all_bin_results.append(part_df)

    if not all_bin_results:
        print("[WARN] No bins had data. Returning empty DataFrame.")
        return pd.DataFrame()

    # Vertically concatenate partial results from all bins
    final_df = pd.concat(all_bin_results, ignore_index=True)

    renamed = {
        "salary-fanduel": "fanduel_salary",
        "salary-draftkings": "draftkings_salary",
        "salary-yahoo": "yahoo_salary",
        "pos-fanduel": "fanduel_position",
        "pos-draftkings": "draftkings_position",
        "pos-yahoo": "yahoo_position",
    }
    final_df = final_df.rename(
        columns={k: v for k, v in renamed.items() if k in final_df.columns}
    )

    # Save final combined results with original column names
    final_output_file = os.path.join(predictions_dir, f"final_fp_{platform}.csv")
    final_df.to_csv(final_output_file, index=False)
    print(f"Saved final results to {final_output_file}")

    return final_df 