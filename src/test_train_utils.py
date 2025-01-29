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
    all_game_ids, all_game_dates, all_player_names = [], [], []
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

        identifying_test_data = df[df["group_col"] == current_group][["player_name", "game_date", "game_id"]].drop_duplicates()

        # Create DMatrix for training and testing
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # Model parameters
        params = {
            "tree_method": "hist",  # Use "gpu_hist" if you have a GPU
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
