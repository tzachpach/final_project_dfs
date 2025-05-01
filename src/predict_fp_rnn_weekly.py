import itertools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from functools import reduce

from config.constants import rnn_param_grid, select_device
from config.dfs_categories import dfs_cats
from config.fantasy_point_calculation import calculate_fp_fanduel


###############################################################################
# 3) Data Preparation with Per-Player Scaling
###############################################################################
def prepare_train_test_rnn_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_platform="fanduel",
    train_window=15,
    predict_ahead=1,
    use_standard_scaler=False
):
    """
    Prepares (X_train, y_train) and (X_test, y_test) for an RNN by:
      1) Doing minimal cleansing + get_dummies on train_df & test_df.
      2) Building sequences WITH per-player scaling (fit on train, apply on test).

    Returns:
        X_train, y_train, X_test, y_test, players_test, dates_test, scalers
    """
    # 1) Sort & Basic Checks
    train_df = train_df.sort_values(["player_name", "game_date"]).reset_index(drop=True)
    test_df  = test_df.sort_values(["player_name", "game_date"]).reset_index(drop=True)

    if target_platform not in ["fanduel", "draftkings", "yahoo"]:
        target_col = target_platform
    else:
        target_col = f"fp_{target_platform}"

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Missing '{target_col}' in train/test DataFrame columns.")

    # Columns to exclude from features
    exclude = {
        "player_name", "game_id", "game_date",
        "available_flag", "group_col", "season_year",
        "fp_draftkings", "fp_fanduel", "fp_yahoo"
    }

    # Possible categorical columns (positions, teams, opponents)
    cat_cols = [c for c in train_df.columns if "pos-" in c or c in ["team_abbreviation", "opponent"]]

    # 2) One-hot encode
    train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
    test_df  = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)

    # Ensure same columns in both
    all_cols = sorted(set(train_df.columns).union(test_df.columns))
    train_df = train_df.reindex(columns=all_cols, fill_value=0)
    test_df  = test_df.reindex(columns=all_cols, fill_value=0)

    feature_cols = [c for c in train_df.columns if c not in exclude]

    # 3) Fit Per-Player Scalers on Train
    scalers = {}
    for player, grp in train_df.groupby("player_name"):
        feat_arr = grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        targ_arr = grp[target_col].apply(pd.to_numeric, errors="coerce").fillna(0).values.reshape(-1, 1)

        X_scaler = StandardScaler() if use_standard_scaler else MinMaxScaler()
        y_scaler = StandardScaler() if use_standard_scaler else MinMaxScaler()

        X_scaler.fit(feat_arr)
        y_scaler.fit(targ_arr)
        scalers[player] = {"X": X_scaler, "y": y_scaler}

    # 4) Build Train Sequences
    X_train_list, y_train_list = [], []
    for player, grp in train_df.groupby("player_name"):
        if player not in scalers:
            continue

        feat_arr = grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        targ_arr = grp[target_col].apply(pd.to_numeric, errors="coerce").fillna(0).values

        feat_scaled = scalers[player]["X"].transform(feat_arr)
        targ_scaled = scalers[player]["y"].transform(targ_arr.reshape(-1, 1)).flatten()

        for i in range(train_window, len(grp) - predict_ahead + 1):
            X_seq = feat_scaled[i - train_window : i]
            y_val = targ_scaled[i + predict_ahead - 1]
            X_train_list.append(X_seq)
            y_train_list.append(y_val)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)

    # 5) Build Test Sequences
    X_test_list, y_test_list = [], []
    players_test, dates_test = [], []

    for player, grp in test_df.groupby("player_name"):
        if player not in scalers:
            continue

        feat_arr = grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        targ_arr = grp[target_col].apply(pd.to_numeric, errors="coerce").fillna(0).values
        date_arr = grp["game_date"].values

        feat_scaled = scalers[player]["X"].transform(feat_arr)
        targ_scaled = scalers[player]["y"].transform(targ_arr.reshape(-1, 1)).flatten()

        for i in range(train_window, len(grp) - predict_ahead + 1):
            X_seq = feat_scaled[i - train_window : i]
            y_val = targ_scaled[i + predict_ahead - 1]
            X_test_list.append(X_seq)
            y_test_list.append(y_val)
            players_test.append(player)
            dates_test.append(date_arr[i + predict_ahead - 1])

    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    return X_train, y_train, X_test, y_test, players_test, dates_test, scalers


###############################################################################
# 4) Simple PyTorch LSTM or GRU
###############################################################################
class SimpleRNN(nn.Module):
    """
    A simple LSTM or GRU-based model for regression on time-series sequences.
    """
    def __init__(self, input_size, hidden_size, num_layers=1, rnn_type="LSTM", dropout=0.0):
        super(SimpleRNN, self).__init__()
        if rnn_type.upper() == "LSTM":
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                               batch_first=True, dropout=dropout)
        elif rnn_type.upper() == "GRU":
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                              batch_first=True, dropout=dropout)
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")

        self.fc = nn.Linear(hidden_size, 1)  # single output for regression

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        rnn_out, _ = self.rnn(x)  # (batch_size, seq_len, hidden_size)
        last_hidden = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        out = self.fc(last_hidden)       # (batch_size, 1)
        return out.squeeze()             # (batch_size,)


###############################################################################
# 5) Training Function
###############################################################################
def train_rnn_model(
    model,
    X_train,
    y_train,
    epochs,
    batch_size,
    learning_rate,
    device,
    X_val=None,
    y_val=None
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
            print(f"Epoch {epoch}/{epochs}, Train Loss: {epoch_loss/num_batches:.4f}, Val Loss: {val_loss:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs}, Train Loss: {epoch_loss/num_batches:.4f}")


###############################################################################
# 6) Rolling Train-Test with Optionally Larger Train Window + Step Size
###############################################################################
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
    output_dir="output_csv"  # New parameter for output directory
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
        df["group_col"] = df["game_date"].dt.year.astype(str) + "_" + df["game_date"].dt.isocalendar().week.astype(str)
        df["group_col"] = df["group_col"].apply(lambda x: f'{x.split("_")[0]}_0{x.split("_")[1]}'
                                                if len(x.split("_")[1]) == 1 else x)
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
                quantile_label=quantile_label
            )
            cat_result = cat_result.rename(columns={
                "y_true": f"{cat}",
                "y_pred": f"{cat}_pred"
            })
            # Save category-specific results
            cat_output_file = os.path.join(output_dir, f"{cat}_{quantile_label}.csv")
            cat_result.to_csv(cat_output_file, index=False)
            print(f"Saved category results to {cat_output_file}")
            all_category_results.append(cat_result)

        combined_df = reduce(lambda left, right: pd.merge(
            left, right,
            on=["player_name", "game_date"],
            how="outer"
        ), all_category_results)
        combined_df = combined_df.drop_duplicates(["player_name", "game_date"])
        combined_df["fp_fanduel_pred"] = combined_df.apply(lambda row: calculate_fp_fanduel(row, pred_mode=True), axis=1)
        combined_df["fp_fanduel"] = combined_df.apply(calculate_fp_fanduel, axis=1)
        combined_df['game_date'] = pd.to_datetime(combined_df['game_date'])

        # Save combined multi-target results
        combined_output_file = os.path.join(output_dir, f"fp_fanduel_{quantile_label}.csv")
        combined_df.to_csv(combined_output_file, index=False)
        print(f"Saved combined multi-target results to {combined_output_file}")

        return combined_df

    # Rest of the function (single-target mode) remains unchanged
    for i in range(train_window, len(unique_groups), step_size):
        current_group = unique_groups[i]
        print(f"\n=== Rolling step index {i}/{len(unique_groups)-1} - Testing group={current_group} ===")

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

        X_train, y_train, X_test, y_test, p_test, d_test, scalers = prepare_train_test_rnn_data(
            train_df, test_df,
            target_platform=platform,
            train_window=train_window,
            predict_ahead=predict_ahead,
            use_standard_scaler=False
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
            dropout=dropout_rate
        )

        train_rnn_model(
            model,
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device
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
            pred_inv = scalers[player_name]["y"].inverse_transform(pred_val.reshape(-1,1))[0,0]
            true_inv = scalers[player_name]["y"].inverse_transform(y_test[idx].reshape(-1,1))[0,0]
            y_pred_unscaled.append(pred_inv)
            y_true_unscaled.append(true_inv)

        all_preds.extend(y_pred_unscaled)
        all_trues.extend(y_true_unscaled)
        all_players.extend(p_test)
        all_dates.extend(d_test)

    results_df = pd.DataFrame({
        "player_name": all_players,
        "game_date": all_dates,
        "y_true": all_trues,
        "y_pred": all_preds
    })

    # Save single-target results
    if not multi_target_mode:
        output_file = os.path.join(output_dir, f"fp_{platform}_{quantile_label}.csv")
        results_df.to_csv(output_file, index=False)
        print(f"Saved single-target results to {output_file}")

    return results_df

###############################################################################
# 7) Utility: run_rnn_and_merge_results (One Platform)
###############################################################################
def run_rnn_and_merge_results(
    df,
    group_by="week",
    platform="fanduel",
    **best_params
):
    """
    1) Runs rolling_train_test_rnn to get predictions for a single platform (e.g. "fanduel")
       using the best hyperparameters.
    2) Optionally set step_size=6 if you want to train/predict every 6 groups.
    3) Merges predictions back into the original df to produce a final table with
       [player_name, game_id, game_date, salaries, positions, etc.,
        fp_fanduel_pred, fp_fanduel].
    4) Returns that final DataFrame.
    """
    # Pass everything to rolling_train_test_rnn, including step_size
    results_df = rolling_train_test_rnn(
        df=df,
        group_by=group_by,
        platform=platform,
        **best_params
    )

    # Rename columns for the final merged output
    results_df = results_df.rename(columns={
        "y_true": f"fp_{platform}",
        "y_pred": f"fp_{platform}_pred"
    })

    # Merge with original to bring back player_name, game_id, game_date, etc.
    keep_cols = [
        "player_name", "game_id", "game_date", "minutes_played",
        "salary-fanduel", "salary-draftkings", "salary-yahoo",
        "pos-fanduel", "pos-draftkings", "pos-yahoo"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df_lookup = df[keep_cols].drop_duplicates(subset=["player_name", "game_id", "game_date"])

    merged = pd.merge(
        results_df[["player_name", "game_date", f"fp_{platform}", f"fp_{platform}_pred"]],
        df_lookup,
        on=["player_name", "game_date"],
        how="left"
    )

    merged = merged.rename(columns={
        "salary-fanduel": "fanduel_salary",
        "salary-draftkings": "draftkings_salary",
        "salary-yahoo": "yahoo_salary",
        "pos-fanduel": "fanduel_position",
        "pos-draftkings": "draftkings_position",
        "pos-yahoo": "yahoo_position"
    })

    return merged

# def tune_rnn_hyperparameters(df, param_grid=rnn_param_grid, group_by="week", platform="fanduel"):
#     """
#     Tunes hyperparameters for the RNN model using a rolling window approach.
#     """
#
#     results = []
#     # Use itertools.product to get all combinations of hyperparameters
#     keys, values = zip(*param_grid.items())
#     i = 0
#     for combination in itertools.product(*values):
#         params = dict(zip(keys, combination))  # Create dict for current combination
#         print(f"Testing parameters: {params}")
#
#         # Call run_and_merge_results with the current set of hyperparameters
#
#         results_df = run_rnn_and_merge_results(df, platform=platform, group_by=group_by, **params)
#         results_df.to_csv(f'version_{i}.csv')
#         i += 1
#
#         # Calculate evaluation metrics (RMSE, MAE, R^2)
#         rmse = np.sqrt(mean_squared_error(results_df['fp_fanduel_y'], results_df['fp_fanduel_pred']))
#         mae = mean_absolute_error(results_df['fp_fanduel_y'], results_df['fp_fanduel_pred'])
#         r2 = r2_score(results_df['fp_fanduel_y'], results_df['fp_fanduel_pred'])
#         print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R^2: {r2:.4f}")
#
#         # Store results with hyperparameter values
#         results.append({
#             **params,  # Store all hyperparameters
#             'rmse': rmse,
#             'mae': mae,
#             'r2': r2,
#         })
#
#
#     # Convert results to DataFrame and sort
#     final_results_df = pd.DataFrame(results)
#     final_results_df = final_results_df.sort_values(by='rmse', ascending=True) # Sort by RMSE
#     final_results_df.to_csv('final_results_rnn.csv')
#     return final_results_df