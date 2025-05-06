import os
import pickle
from functools import reduce

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch import nn, optim

from config.constants import select_device
from config.dfs_categories import dfs_cats
from config.fantasy_point_calculation import calculate_fp_fanduel


def rolling_train_test_for_xgb(
    X,
    y,
    df,
    group_by="date",
    train_window=10,
    save_model=False,
    model_dir="models",
    xgb_param_dict=None,
    output_dir=None,
    quantile_label=None,
):
    """
    Rolling train-test function for both daily and weekly training, based on a grouping parameter.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.Series): Target variable.
        df (pd.DataFrame): Original DataFrame containing game metadata.
        group_by (str): "date" for daily predictions or "week" for weekly predictions.
        train_window (int): Number of previous groups (days/weeks) to use for training.
        save_model (bool): Whether to save the trained models.
        model_dir (str): Directory to save models.

    Returns:
        pd.DataFrame: DataFrame containing predictions, actual values, and metadata.
    """
    os.makedirs(model_dir, exist_ok=True)

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

    # Initialize lists to store results
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

    # Loop over groups with a rolling window
    for idx in range(train_window, len(unique_groups)):
        current_group = unique_groups[idx]
        training_groups = unique_groups[idx - train_window : idx]

        # Training data: all data in the rolling window
        X_train = X[X["group_col"].isin(training_groups)].copy()
        y_train = y.loc[X_train.index]

        # Test data: data for the current group
        X_test = X[X["group_col"] == current_group].copy()
        y_test = y.loc[X_test.index]

        if X_train.empty or X_test.empty:
            continue

        # Drop non-numeric columns before model training
        X_train = X_train.drop(columns=["game_date", "group_col"]).reset_index(
            drop=True
        )
        X_test = X_test.drop(columns=["game_date", "group_col"]).reset_index(drop=True)

        # Ensure categorical columns are of type 'category'
        for col in X_train.columns:
            if col in cat_cols:
                X_train[col] = X_train[col].astype("category")
                X_test[col] = X_test[col].astype("category")
            else:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        identifying_test_data = df[df["group_col"] == current_group][
            ["player_name", "game_date", "game_id", "minutes_played"]
        ].drop_duplicates()

        # Create DMatrix for training and testing
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # Model parameters
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",  # use "gpu_hist" on a GPU box
            "enable_categorical": True,
        }

        # --- user overrides ---
        if xgb_param_dict:  # None or {}
            # all keys except those that belong to the booster loop length
            non_round_keys = {
                k: v
                for k, v in xgb_param_dict.items()
                if k not in ("num_boost_round", "n_estimators")
            }
            params.update(non_round_keys)

        num_rounds = (
            xgb_param_dict.get("num_boost_round")
            or xgb_param_dict.get("n_estimators")
            or 100  # ← previous implicit default
        )

        # Train the model
        model = xgb.train(params, dtrain, num_boost_round=num_rounds)

        # Make predictions
        y_pred = model.predict(dtest)

        # Store predictions and metadata
        all_predictions.extend(y_pred.tolist())
        all_true_values.extend(y_test.tolist())  # Ensure list format
        all_game_ids.extend(identifying_test_data["game_id"].tolist())
        all_game_dates.extend(identifying_test_data["game_date"].tolist())
        all_player_names.extend(identifying_test_data["player_name"].tolist())
        all_minutes_played.extend(identifying_test_data["minutes_played"].tolist())
        all_fanduel_salaries.extend(X_test["salary-fanduel"].tolist())
        all_draftkings_salaries.extend(X_test["salary-draftkings"].tolist())
        all_yahoo_salaries.extend(X_test["salary-yahoo"].tolist())
        all_fanduel_positions.extend(X_test["pos-fanduel"].tolist())
        all_draftkings_positions.extend(X_test["pos-draftkings"].tolist())
        all_yahoo_positions.extend(X_test["pos-yahoo"].tolist())

        # Save the model if requested
        if save_model:
            model_filename = f"{model_dir}/model_{group_by}_{current_group}.pkl"
            with open(model_filename, "wb") as file:
                pickle.dump(model, file)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f"Training up to {group_by}: {current_group}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"R-squared (R²): {r2:.2f}")
        print("")

    # Compile results into a DataFrame
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
        fname = f"fp_xgb_{quantile_label}.csv"
        results_df.to_csv(os.path.join(output_dir, fname), index=False)
        print(f"Saved XGB intermediate results → {fname}")

    return results_df


def make_unique_player_key(df: pd.DataFrame) -> pd.Series:
    """
    Robust surrogate key: name_team_season[_n].
    """
    base = (
        df["player_name"].astype(str).str.replace(" ", "_")
        + "_"
        + df["team_abbreviation"].astype(str)
        + "_"
        + df["season_year"].str[:4]  # "2023-24" -> "2023"
    )

    dup_idx = df.groupby(base).cumcount()  # 0,1,2,...
    # Only append suffix where duplicates exist (dup_idx > 0)
    key = np.where(dup_idx == 0, base, base + "_" + dup_idx.astype(str))
    return pd.Series(key, index=df.index, name="player_key")


def prepare_train_test_rnn_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_platform="fanduel",
    lookback=15,
    predict_ahead=1,
    use_standard_scaler=False,
):
    """
    Prepares (X_train, y_train) and (X_test, y_test) for an RNN by:
      1) Doing minimal cleansing + get_dummies on train_df & test_df.
      2) Building sequences WITH per-player scaling (fit on train, apply on test).

    Returns:
        X_train, y_train, X_test, y_test, players_test, dates_test, scalers
    """

    # ---------- 0 | create an ad‑hoc unique key ----------
    #  (concat name + current team; good enough to split 99 % of name clashes)
    for df in (train_df, test_df):
        df["player_key"] = make_unique_player_key(df)

    key_col = "player_key"
    #################### 1) Sort & Basic Checks ####################
    train_df = train_df.sort_values([key_col, "game_date"]).reset_index(drop=True)
    test_df = test_df.sort_values([key_col, "game_date"]).reset_index(drop=True)

    target_col = f"fp_{target_platform}"
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Missing '{target_col}' in train/test DataFrame columns.")

    exclude = {
        "player_name",
        key_col,
        "game_id",
        "game_date",
        "available_flag",
        "group_col",
        "season_year",
        "fp_draftkings",
        "fp_fanduel",
        "fp_yahoo",
    }
    cat_cols = [
        c
        for c in train_df.columns
        if "pos-" in c or c in ["team_abbreviation", "opponent"]
    ]

    #################### 2) One-hot encode ####################
    train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
    train_cols = train_df.columns  # freeze final train design matrix

    test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)
    test_df = test_df.reindex(columns=train_cols, fill_value=0)  # align

    feature_cols = [c for c in train_cols if c not in exclude]

    # make sure numeric dtypes are float32 early (faster later)
    train_df[feature_cols] = train_df[feature_cols].astype(np.float32)
    test_df[feature_cols] = test_df[feature_cols].astype(np.float32)

    #################### 3) Fit Per-Player Scalers on Train ####################
    # We'll store a dict of scalers keyed by player_name
    scaler_cls = StandardScaler if use_standard_scaler else MinMaxScaler
    scalers = {}

    for player, grp in train_df.groupby(key_col):
        X_scaler = scaler_cls()
        y_scaler = scaler_cls()
        X_scaler.fit(grp[feature_cols].values)
        y_scaler.fit(grp[[target_col]].values)
        scalers[player] = {"X": X_scaler, "y": y_scaler}

    #################### 4) Build Train Sequences (scaled) ####################
    X_tr, y_tr = [], []
    for player, grp in train_df.groupby(key_col):
        if player not in scalers:
            continue
        feat_scaled = scalers[player]["X"].transform(grp[feature_cols].values)
        targ_scaled = scalers[player]["y"].transform(grp[[target_col]].values).flatten()

        for i in range(lookback, len(grp) - predict_ahead + 1):
            X_tr.append(feat_scaled[i - lookback : i])
            y_tr.append(targ_scaled[i + predict_ahead - 1])

    X_train = np.asarray(X_tr, dtype=np.float32)
    y_train = np.asarray(y_tr, dtype=np.float32)

    #################### 5) Build Test Sequences (scaled) ####################
    X_te, y_te, players_test, dates_test = [], [], [], []

    for player, grp in test_df.groupby(key_col):
        if player not in scalers:
            continue
        feat_scaled = scalers[player]["X"].transform(grp[feature_cols].values)
        targ_scaled = scalers[player]["y"].transform(grp[[target_col]].values).flatten()
        date_arr = grp["game_date"].values

        for i in range(lookback, len(grp) - predict_ahead + 1):
            X_te.append(feat_scaled[i - lookback : i])
            y_te.append(targ_scaled[i + predict_ahead - 1])
            players_test.append(player)
            dates_test.append(date_arr[i + predict_ahead - 1])

    X_test = np.asarray(X_te, dtype=np.float32)
    y_test = np.asarray(y_te, dtype=np.float32)

    #################### testing ####################
    assert X_train.shape[0] == y_train.shape[0]  # sequence/label parity
    assert set(players_test).issubset(test_df.player_name)  # readable names
    assert all(k in scalers for k in train_df.player_key.unique())  # scalers built

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
    group_by="week",
    predict_ahead=1,
    platform="fanduel",
    step_size=1,
    output_dir="output_csv",  # New parameter for output directory
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
                train_window=train_window,
                predict_ahead=predict_ahead,
                use_standard_scaler=False,
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
            "player_name": all_players,
            "game_date": all_dates,
            "y_true": all_trues,
            "y_pred": all_preds,
        }
    )

    # Save single-target results
    if not multi_target_mode:
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
