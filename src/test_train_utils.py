import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score


def rolling_train_test(X, y, df, group_by="date", train_window=10, save_model=False, model_dir="models"):
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

    cat_cols = ["team_abbreviation", "player_name", "opponent", "pos-draftkings", "pos-fanduel", "pos-yahoo", "season_year"]

    # Loop over groups with a rolling window
    for idx in range(train_window, len(unique_groups)):
        current_group = unique_groups[idx]
        training_groups = unique_groups[idx - train_window:idx]

        # Training data: all data in the rolling window
        X_train = X[X["group_col"].isin(training_groups)].copy()
        y_train = y.loc[X_train.index]

        # Test data: data for the current group
        X_test = X[X["group_col"] == current_group].copy()
        y_test = y.loc[X_test.index]

        if X_train.empty or X_test.empty:
            continue

        # Drop non-numeric columns before model training
        X_train = X_train.drop(columns=["game_date", "group_col"]).reset_index(drop=True)
        X_test = X_test.drop(columns=["game_date", "group_col"]).reset_index(drop=True)

        # Ensure categorical columns are of type 'category'
        for col in X_train.columns:
            if col in cat_cols:
                X_train[col] = X_train[col].astype("category")
                X_test[col] = X_test[col].astype("category")
            else:
                X_train[col] = pd.to_numeric(X_train[col], errors="coerce")
                X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

        identifying_test_data = df[df["group_col"] == current_group][["player_name", "game_date", "game_id", "minutes_played"]].drop_duplicates()

        # Create DMatrix for training and testing
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # Model parameters
        params = {
            "tree_method": "hist", # Use "gpu_hist" if you have a GPU
            "enable_categorical": True,
        }

        # Train the model
        model = xgb.train(params, dtrain)

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
    results_df = pd.DataFrame({
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
    })

    return results_df


def prepare_train_test_rnn_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_platform="fanduel",
    lookback=15,
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
    #################### 1) Sort & Basic Checks ####################
    train_df = train_df.sort_values(["player_name", "game_date"]).reset_index(drop=True)
    test_df  = test_df.sort_values(["player_name", "game_date"]).reset_index(drop=True)

    target_col = f"fp_{target_platform}"
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(f"Missing '{target_col}' in train/test DataFrame columns.")

    exclude = {
        "player_name", "game_id", "game_date",
        "available_flag", "group_col", "season_year",
        "fp_draftkings", "fp_fanduel", "fp_yahoo"
    }
    cat_cols = [c for c in train_df.columns if "pos-" in c or c in ["team_abbreviation", "opponent"]]

    #################### 2) One-hot encode ####################
    train_df = pd.get_dummies(train_df, columns=cat_cols, drop_first=True)
    test_df  = pd.get_dummies(test_df, columns=cat_cols, drop_first=True)

    # Ensure same columns in both
    all_cols = sorted(set(train_df.columns).union(test_df.columns))
    train_df = train_df.reindex(columns=all_cols, fill_value=0)
    test_df  = test_df.reindex(columns=all_cols, fill_value=0)

    feature_cols = [c for c in train_df.columns if c not in exclude]

    #################### 3) Fit Per-Player Scalers on Train ####################
    # We'll store a dict of scalers keyed by player_name
    # e.g. scalers[player_name] = {"X": <scaler>, "y": <scaler>}
    scalers = {}

    # We'll first do a pass to fit scalers for each player on the training data.
    for player, grp in train_df.groupby("player_name"):
        # convert to numeric
        feat_arr = grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        targ_arr = grp[target_col].apply(pd.to_numeric, errors="coerce").fillna(0).values.reshape(-1,1)

        # define scalers
        X_scaler = StandardScaler() if use_standard_scaler else MinMaxScaler()
        y_scaler = StandardScaler() if use_standard_scaler else MinMaxScaler()

        # fit on the player's train features/target
        X_scaler.fit(feat_arr)   # shape: (num_rows, num_features)
        y_scaler.fit(targ_arr)   # shape: (num_rows, 1)

        scalers[player] = {"X": X_scaler, "y": y_scaler}

    #################### 4) Build Train Sequences (scaled) ####################
    X_train_list, y_train_list = [], []
    for player, grp in train_df.groupby("player_name"):
        feat_arr = grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        targ_arr = grp[target_col].apply(pd.to_numeric, errors="coerce").fillna(0).values

        # If no scaler for this player, skip
        if player not in scalers:
            continue
        # apply scaling
        feat_scaled = scalers[player]["X"].transform(feat_arr)  # shape: (rows, features)
        targ_scaled = scalers[player]["y"].transform(targ_arr.reshape(-1,1)).flatten()

        for i in range(lookback, len(grp) - predict_ahead + 1):
            X_seq = feat_scaled[i - lookback : i]             # shape: (lookback, features)
            y_val = targ_scaled[i + predict_ahead - 1]       # single float
            X_train_list.append(X_seq)
            y_train_list.append(y_val)

    X_train = np.array(X_train_list, dtype=np.float32)
    y_train = np.array(y_train_list, dtype=np.float32)

    #################### 5) Build Test Sequences (scaled) ####################
    X_test_list, y_test_list = [], []
    players_test, dates_test = [], []

    for player, grp in test_df.groupby("player_name"):
        feat_arr = grp[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        targ_arr = grp[target_col].apply(pd.to_numeric, errors="coerce").fillna(0).values
        date_arr = grp["game_date"].values

        if player not in scalers:
            # Means the player wasn't in train
            continue

        # scale with the same scaler from train
        feat_scaled = scalers[player]["X"].transform(feat_arr)
        targ_scaled = scalers[player]["y"].transform(targ_arr.reshape(-1,1)).flatten()

        for i in range(lookback, len(grp) - predict_ahead + 1):
            X_seq = feat_scaled[i - lookback : i]
            y_val = targ_scaled[i + predict_ahead - 1]

            X_test_list.append(X_seq)
            y_test_list.append(y_val)
            players_test.append(player)
            dates_test.append(date_arr[i + predict_ahead - 1])

    X_test = np.array(X_test_list, dtype=np.float32)
    y_test = np.array(y_test_list, dtype=np.float32)

    return X_train, y_train, X_test, y_test, players_test, dates_test, scalers