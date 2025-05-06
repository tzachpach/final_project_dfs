import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.test_train_utils import rolling_train_test_rnn


def prepare_train_test_rnn_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_platform="fanduel",
    train_window=15,
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
    # 1) Sort & Basic Checks
    train_df = train_df.sort_values(["player_name", "game_date"]).reset_index(drop=True)
    test_df = test_df.sort_values(["player_name", "game_date"]).reset_index(drop=True)

    if target_platform not in ["fanduel", "draftkings", "yahoo"]:
        target_col = target_platform
    else:
        target_col = f"fp_{target_platform}"

    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Missing '{target_col}' in train/test DataFrame columns.")

    # Columns to exclude from features
    exclude = {
        "player_name",
        "game_id",
        "game_date",
        "available_flag",
        "group_col",
        "season_year",
        "fp_draftkings",
        "fp_fanduel",
        "fp_yahoo",
    }

    # Possible categorical columns (positions, teams, opponents)
    cat_cols = [
        c
        for c in train_df.columns
        if "pos-" in c or c in ["team_abbreviation", "opponent"]
    ]

    # 2) One-hot encode
    train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
    test_df = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)

    # Ensure same columns in both
    all_cols = sorted(set(train_df.columns).union(test_df.columns))
    train_df = train_df.reindex(columns=all_cols, fill_value=0)
    test_df = test_df.reindex(columns=all_cols, fill_value=0)

    feature_cols = [c for c in train_df.columns if c not in exclude]

    # 3) Fit Per-Player Scalers on Train
    scalers = {}
    for player, grp in train_df.groupby("player_name"):
        feat_arr = (
            grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        )
        targ_arr = (
            grp[target_col]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .values.reshape(-1, 1)
        )

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

        feat_arr = (
            grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        )
        targ_arr = (
            grp[target_col].apply(pd.to_numeric, errors="coerce").fillna(0).values
        )

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

        feat_arr = (
            grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        )
        targ_arr = (
            grp[target_col].apply(pd.to_numeric, errors="coerce").fillna(0).values
        )
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


def run_rnn_and_merge_results(df, group_by="week", platform="fanduel", **best_params):
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
        df=df, group_by=group_by, platform=platform, **best_params
    )

    # Rename columns for the final merged output
    results_df = results_df.rename(
        columns={"y_true": f"fp_{platform}", "y_pred": f"fp_{platform}_pred"}
    )

    # Merge with original to bring back player_name, game_id, game_date, etc.
    keep_cols = [
        "player_name",
        "game_id",
        "game_date",
        "minutes_played",
        "salary-fanduel",
        "salary-draftkings",
        "salary-yahoo",
        "pos-fanduel",
        "pos-draftkings",
        "pos-yahoo",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    df_lookup = df[keep_cols].drop_duplicates(
        subset=["player_name", "game_id", "game_date"]
    )

    merged = pd.merge(
        results_df[
            ["player_name", "game_date", f"fp_{platform}", f"fp_{platform}_pred"]
        ],
        df_lookup,
        on=["player_name", "game_date"],
        how="left",
    )

    merged = merged.rename(
        columns={
            "salary-fanduel": "fanduel_salary",
            "salary-draftkings": "draftkings_salary",
            "salary-yahoo": "yahoo_salary",
            "pos-fanduel": "fanduel_position",
            "pos-draftkings": "draftkings_position",
            "pos-yahoo": "yahoo_position",
        }
    )

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
