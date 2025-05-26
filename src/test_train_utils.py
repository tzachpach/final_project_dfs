import os
import pickle
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn, optim

from config.constants import select_device
from config.dfs_categories import dfs_cats, same_game_cols
from config.fantasy_point_calculation import calculate_fp_fanduel
from config.feature_engineering import fit_feature_reducer, apply_feature_reducer


def rolling_train_test_for_xgb(
    X,
    y,
    df,
    group_by: str,
    train_window: int,
    save_model: bool,
    model_dir: str,
    reduce_features_flag: bool,
    xgb_param_dict=None,
    output_dir=None,
    quantile_label=None,
):
    os.makedirs(model_dir, exist_ok=True)

    # ---------------- group-col construction --------------
    if group_by == "date":
        X["group_col"] = X["game_date"]
        df["group_col"] = df["game_date"]
    elif group_by == "week":
        season_start = X.groupby("season_year")["game_date"].transform("min")
        X["group_col"] = ((X["game_date"] - season_start).dt.days // 7) + 1
        df["group_col"] = ((df["game_date"] - season_start).dt.days // 7) + 1
    else:
        raise ValueError("Invalid value for group_by. Use 'date' or 'week'.")

    unique_groups = sorted(X["group_col"].unique())

    # ---------------- bookkeeping -------------------------
    all_predictions, all_true_values = [], []
    all_game_ids, all_game_dates, all_player_names, all_minutes_played = [], [], [], []
    all_fanduel_salaries, all_draftkings_salaries, all_yahoo_salaries = [], [], []
    all_fanduel_positions, all_draftkings_positions, all_yahoo_positions = [], [], []

    cat_cols = [
        "team_abbreviation",
        "player_name",
        "opponent_abbr",
        "pos-draftkings",
        "pos-fanduel",
        "pos-yahoo",
        "season_year",
    ]

    # ---------------- rolling window loop -----------------------------
    for idx in range(train_window, len(unique_groups)):
        current_group = unique_groups[idx]
        training_groups = unique_groups[idx - train_window : idx]

        X_train = X[X["group_col"].isin(training_groups)].copy()
        y_train = y.loc[X_train.index].reset_index(drop=True)

        X_test = X[X["group_col"] == current_group].copy()
        y_test = y.loc[X_test.index].reset_index(drop=True)

        if X_train.empty or X_test.empty:
            continue

        # -- drop helper cols, cast dtypes --------------------------------
        X_train = X_train.drop(columns=["game_date", "group_col"]).reset_index(
            drop=True
        )
        X_test = X_test.drop(columns=["game_date", "group_col"]).reset_index(drop=True)

        for col in X_train.columns:
            if col in cat_cols:
                X_train[col] = X_train[col].astype("category")
                X_test[col] = X_test[col].astype("category")
            else:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        # -- ***feature reduction*** --------------------------------------
        if reduce_features_flag:
            transf, aux_cols = fit_feature_reducer(  # fit on train only
                X_train,
                y_train,
                mode=reduce_features_flag,
                keep_ratio=0.70,
                cap=300,
            )
            # always keep categories & salary cols
            aux_cols.extend([c for c in cat_cols if c in X_train])
            aux_cols = list(dict.fromkeys(aux_cols))  # dedupe

            X_train = apply_feature_reducer(X_train, transf, aux_cols)
            X_test = apply_feature_reducer(X_test, transf, aux_cols)
        # else: keep original X_train / X_test

        # -- IDs for later -------------------------------------------------
        ids = df[df["group_col"] == current_group][
            ["player_name", "game_date", "game_id", "minutes_played"]
        ].drop_duplicates()

        # -- XGBoost -------------------------------------------------------
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "enable_categorical": True,
        }
        if xgb_param_dict:
            params.update(
                {
                    k: v
                    for k, v in xgb_param_dict.items()
                    if k not in ("num_boost_round", "n_estimators")
                }
            )

        n_rounds = (
            (xgb_param_dict or {}).get("num_boost_round")
            or (xgb_param_dict or {}).get("n_estimators")
            or 100
        )

        model = xgb.train(params, dtrain, num_boost_round=n_rounds)
        y_pred = model.predict(dtest)

        # -- bookkeeping ---------------------------------------------------
        all_predictions.extend(y_pred.tolist())
        all_true_values.extend(y_test.tolist())
        all_game_ids.extend(ids["game_id"])
        all_game_dates.extend(ids["game_date"])
        all_player_names.extend(ids["player_name"])
        all_minutes_played.extend(ids["minutes_played"])
        all_fanduel_salaries.extend(X_test["salary-fanduel"])
        all_draftkings_salaries.extend(X_test["salary-draftkings"])
        all_yahoo_salaries.extend(X_test["salary-yahoo"])
        all_fanduel_positions.extend(X_test["pos-fanduel"])
        all_draftkings_positions.extend(X_test["pos-draftkings"])
        all_yahoo_positions.extend(X_test["pos-yahoo"])

        if save_model:
            model_path = f"{model_dir}/model_{group_by}_{current_group}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        print(f"[{group_by}={current_group}]  RMSE={rmse:.2f}  R²={r2:.2f}")

    # ---------------- assemble output -----------------------------------
    results_df = pd.DataFrame(
        {
            "player_name": all_player_names,
            "minutes_played": all_minutes_played,
            "game_id": all_game_ids,
            "game_date": all_game_dates,
            "y": all_true_values,
            "y_pred": all_predictions,
            "fanduel_salary": all_fanduel_salaries,
            "draftkings_salary": all_draftkings_salaries,
            "yahoo_salary": all_yahoo_salaries,
            "fanduel_position": all_fanduel_positions,
            "draftkings_position": all_draftkings_positions,
            "yahoo_position": all_yahoo_positions,
        }
    )

    if output_dir and quantile_label:
        out = Path(output_dir) / f"fp_xgb_{quantile_label}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(out, index=False)
        print(f"Saved XGB intermediate results → {out}")

    return results_df


def player_key_to_name(player_key):
    """
    Convert a player_key (e.g., 'AARON_GORDON_ORL') back to the original player name (e.g., 'Aaron Gordon').
    Args:
        player_key (str): The player key in the format "FIRSTNAME_LASTNAME_TEAM"

    Returns:
        str: The player name in the format "Firstname Lastname"
    """
    # Split the player_key by underscores
    parts = player_key.split("_")
    # Take all parts except the last one (which is the team abbreviation)
    name_parts = parts[:-1]
    # Join these parts with spaces
    uppercase_name = "_".join(name_parts)
    # Replace underscores with spaces
    player_name = uppercase_name.replace("_", " ")
    return player_name


def prepare_train_test_rnn_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    reduce_features_flag,
    target_platform: str = "fanduel",
    lookback: int = 15,
    predict_ahead: int = 1,
    use_standard_scaler: bool = False,
):
    """
    Build train / test tensors for the RNN with per-player scaling
    and (optionally) numeric feature reduction.

    Returns
    -------
    X_train, y_train, X_test, y_test, players_test, dates_test, scalers
    """

    # 0) stable player key -------------------------------------------------
    for df in (train_df, test_df):
        df["player_key"] = df["player_name"] + "_" + df["team_abbreviation"].str.upper()
    KEY = "player_key"

    # 1) sort + target column --------------------------------------------
    train_df = train_df.sort_values([KEY, "game_date"]).reset_index(drop=True)
    test_df = test_df.sort_values([KEY, "game_date"]).reset_index(drop=True)

    target_col = (
        target_platform
        if target_platform in train_df.columns
        else f"fp_{target_platform}"
    )
    if target_col not in train_df.columns:
        raise ValueError(f"column “{target_col}” not found")

    # 2) one-hot categorical columns --------------------------------------
    CAT = [
        c
        for c in train_df.columns
        if "pos-" in c or c in ("team_abbreviation", "opponent_abbr")
    ]
    train_df = pd.get_dummies(train_df, columns=CAT, drop_first=True)
    DESIGN = train_df.columns  # frozen
    test_df = pd.get_dummies(test_df, columns=CAT, drop_first=True).reindex(
        columns=DESIGN, fill_value=0
    )

    # 3) numeric feature set (raw) ---------------------------------------
    EXCLUDE = {
        "player_name",
        KEY,
        "game_id",
        "game_date",
        "group_col",
        "available_flag",
        "season_year",
        "fp_draftkings",
        "fp_fanduel",
        "fp_yahoo",
    }
    NUM_FEATS = [c for c in DESIGN if c not in EXCLUDE and c not in same_game_cols]

    # 4) fit / apply reducer (optional) -----------------------------------
    if reduce_features_flag is False:
        red_train = train_df[NUM_FEATS]
        red_test = test_df[NUM_FEATS]
    else:
        transf, aux_cols = fit_feature_reducer(
            X_train=train_df[NUM_FEATS],
            y_train=train_df[target_col],
            mode=reduce_features_flag,  # "Kbest", "PCA", …
            keep_ratio=0.70,
            cap=300,
        )
        # always append pos-/salary columns
        aux_cols += [c for c in DESIGN if ("pos-" in c or "salary" in c)]
        aux_cols = list(dict.fromkeys(aux_cols))  # dedupe

        red_train = apply_feature_reducer(train_df, transf, aux_cols)
        red_test = apply_feature_reducer(test_df, transf, aux_cols)

    FEATS = red_train.columns.tolist()
    assert not (set(FEATS) & set(same_game_cols))

    # 5) per-player scalers ----------------------------------------------
    scaler_cls = StandardScaler if use_standard_scaler else MinMaxScaler
    scalers = {}
    for k, grp in train_df.groupby(KEY):
        if len(grp) < 2:  # cannot scale a singleton
            continue
        scalers[k] = {
            "X": scaler_cls().fit(red_train.loc[grp.index, FEATS]),
            "y": scaler_cls().fit(grp[[target_col]].values),
        }

    # 6) sequence builder -------------------------------------------------
    def build_sequences(source_idx, is_train: bool):
        X_l, y_l, p_l, d_l = [], [], [], []
        df_here = train_df if is_train else test_df
        red_here = red_train if is_train else red_test

        for k, grp in df_here.groupby(KEY):
            if k not in scalers:  # unseen in train
                continue
            X_scaled = scalers[k]["X"].transform(red_here.loc[grp.index, FEATS])
            y_scaled = scalers[k]["y"].transform(grp[[target_col]].values).flatten()
            dates = grp["game_date"].values

            for i in range(lookback, len(grp) - predict_ahead + 1):
                X_l.append(X_scaled[i - lookback : i])
                y_l.append(y_scaled[i + predict_ahead - 1])
                if not is_train:
                    p_l.append(k)
                    d_l.append(dates[i + predict_ahead - 1])

        X_arr = np.asarray(X_l, dtype="float32")
        y_arr = np.asarray(y_l, dtype="float32")
        return (X_arr, y_arr) if is_train else (X_arr, y_arr, p_l, d_l)

    X_train, y_train = build_sequences(train_df.index, True)
    X_test, y_test, players_test, dates_test = build_sequences(test_df.index, False)

    # 7) sanity checks ----------------------------------------------------
    assert X_train.shape[0] == y_train.shape[0]
    assert set(players_test).issubset(scalers.keys())

    return X_train, y_train, X_test, y_test, players_test, dates_test, scalers


def rolling_train_test_rnn(
    df: pd.DataFrame,
    train_window,
    hidden_size,
    num_layers,
    learning_rate,
    dropout_rate,
    epochs,
    batch_size,
    rnn_type,
    multi_target_mode,
    quantile_label,
    reduce_features_flag,
    group_by="week",
    predict_ahead=1,
    platform="fanduel",
    step_size=1,
    output_dir="output_csv",
    save_csv=True,
):
    """
    Minimal rolling approach:
    1) Group data by 'week' or 'date'.
    2) For each step i in range(train_window, len(unique_groups), step_size):
        - Use last 'train_window' groups for training
        - Use the current group for testing (plus 'lookback' history).
    3) Build & train an RNN, then predict on the test set.
    4) Inverse-transform predictions => store real fantasy point values.
    5) Save results to the specified output_dir.

    Args:
        output_dir (str): Directory to save CSV results.
        ... (other args remain unchanged)
    """
    device = select_device()

    df = df.copy().dropna()
    df["game_date"] = pd.to_datetime(df["game_date"])

    if group_by == "week":
        df["group_col"] = (
            df["game_date"].dt.year.astype(str)
            + "_"
            + df["game_date"].dt.isocalendar().week.astype(str)
        )
        df["group_col"] = df["group_col"].apply(
            lambda x: (
                f'{x.split("_")[0]}_0{x.split("_")[1]}'
                if len(x.split("_")[1]) == 1
                else x
            )
        )
    elif group_by == "date":
        df["group_col"] = df["game_date"]
    else:
        raise ValueError("group_by must be 'week' or 'date'.")

    unique_groups = sorted(df["group_col"].unique())
    all_preds = []
    all_trues = []
    all_players = []
    all_dates = []

    if multi_target_mode:
        all_category_results = []
        for cat in dfs_cats:  # Assuming dfs_cats is defined elsewhere
            cat_df = df.copy()
            cat_result = rolling_train_test_rnn(
                df=cat_df,
                train_window=train_window,
                hidden_size=hidden_size,
                num_layers=num_layers,
                learning_rate=learning_rate,
                dropout_rate=dropout_rate,
                epochs=epochs,
                batch_size=batch_size,
                rnn_type=rnn_type,
                multi_target_mode=False,
                group_by=group_by,
                predict_ahead=predict_ahead,
                platform=cat,
                step_size=step_size,
                output_dir=output_dir,
                quantile_label=quantile_label,
                reduce_features_flag=reduce_features_flag,
                save_csv=False,
            )
            cat_result = cat_result.rename(
                columns={"y_true": f"{cat}", "y_pred": f"{cat}_pred"}
            )
            # Save category-specific results
            cat_output_file = os.path.join(output_dir, f"{cat}_{quantile_label}.csv")
            cat_result.to_csv(cat_output_file, index=False)
            print(f"Saved category results to {cat_output_file}")
            all_category_results.append(cat_result)

        combined_df = reduce(
            lambda left, right: pd.merge(
                left, right, on=["player_name", "game_date"], how="outer"
            ),
            all_category_results,
        )
        combined_df = combined_df.drop_duplicates(["player_name", "game_date"])
        combined_df["fp_fanduel_pred"] = combined_df.apply(
            lambda row: calculate_fp_fanduel(row, pred_mode=True), axis=1
        )
        combined_df["fp_fanduel"] = combined_df.apply(calculate_fp_fanduel, axis=1)
        combined_df["game_date"] = pd.to_datetime(combined_df["game_date"])

        # Save combined multi-target results
        combined_output_file = os.path.join(
            output_dir, f"fp_fanduel_{quantile_label}.csv"
        )
        combined_df.to_csv(combined_output_file, index=False)
        print(f"Saved combined multi-target results to {combined_output_file}")

        return combined_df

    # Rest of the function (single-target mode) remains unchanged
    for i in range(train_window, len(unique_groups), step_size):
        current_group = unique_groups[i]
        print(
            f"\n=== Rolling step index {i}/{len(unique_groups)-1} - Testing group={current_group} ==="
        )

        train_groups = unique_groups[i - train_window : i]
        train_df = df[df["group_col"].isin(train_groups)]

        if group_by == "week":
            test_groups = unique_groups[max(0, i - train_window + 1) : i + 1]
            test_df = df[df["group_col"].isin(test_groups)]
        else:
            cg_date = df.loc[df["group_col"] == current_group, "game_date"].max()
            if pd.isnull(cg_date):
                print(f"Skipping group={current_group} - no date found.")
                continue
            test_start = cg_date - pd.Timedelta(days=train_window - 1)
            test_df = df[(df["game_date"] >= test_start) & (df["game_date"] <= cg_date)]

        if train_df.empty or test_df.empty:
            print("Skipping - train_df or test_df is empty.")
            continue

        X_train, y_train, X_test, y_test, p_test, d_test, scalers = (
            prepare_train_test_rnn_data(
                train_df,
                test_df,
                target_platform=platform,
                lookback=train_window,
                predict_ahead=predict_ahead,
                use_standard_scaler=False,
                reduce_features_flag=reduce_features_flag,
            )
        )

        if len(X_train) == 0 or len(X_test) == 0:
            print("Skipping - no valid sequences in train/test.")
            continue

        input_size = X_train.shape[2]
        model = SimpleRNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=rnn_type,
            dropout=dropout_rate,
        )

        train_rnn_model(
            model,
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

        model.eval()
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
        with torch.no_grad():
            y_pred_scaled = model(X_test_t).cpu().numpy()

        y_pred_unscaled = []
        y_true_unscaled = []
        for idx, pred_val in enumerate(y_pred_scaled):
            player_name = p_test[idx]
            if player_name not in scalers:
                continue
            pred_inv = scalers[player_name]["y"].inverse_transform(
                pred_val.reshape(-1, 1)
            )[0, 0]
            true_inv = scalers[player_name]["y"].inverse_transform(
                y_test[idx].reshape(-1, 1)
            )[0, 0]
            y_pred_unscaled.append(pred_inv)
            y_true_unscaled.append(true_inv)

        all_preds.extend(y_pred_unscaled)
        all_trues.extend(y_true_unscaled)
        all_players.extend(p_test)
        all_dates.extend(d_test)

    results_df = pd.DataFrame(
        {
            "player_name": [player_key_to_name(p) for p in all_players],
            "game_date": all_dates,
            "y_true": all_trues,
            "y_pred": all_preds,
        }
    )

    # Save single-target results
    if not multi_target_mode and save_csv:
        output_file = os.path.join(output_dir, f"fp_{platform}_{quantile_label}.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Saved single-target results to {output_file}")
    return results_df


class SimpleRNN(nn.Module):
    """
    A simple LSTM or GRU-based model for regression on time-series sequences.
    """

    def __init__(
        self, input_size, hidden_size, num_layers=1, rnn_type="LSTM", dropout=0.0
    ):
        super(SimpleRNN, self).__init__()
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")

        self.fc = nn.Linear(hidden_size, 1)  # single output for regression

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size)
        last_hidden = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(last_hidden)  # (batch_size, 1)
        return out.squeeze()  # (batch_size,)


def train_rnn_model(
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
    """
    Standard training loop with MSE loss and Adam optimizer, on the chosen 'device'.
    """
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

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
            print(
                f"Epoch {epoch}/{epochs}, Train Loss: {epoch_loss/num_batches:.4f}, Val Loss: {val_loss:.4f}"
            )
        else:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {epoch_loss/num_batches:.4f}")
