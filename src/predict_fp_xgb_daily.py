import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import pickle
from sklearn.metrics import mean_squared_error, r2_score

from constants import *
from preprocess_merged_csvs import merge_all_seasons


def rolling_train_test(X, y, save_model=False, model_dir='models'):
    os.makedirs(model_dir, exist_ok=True)

    # Initialize lists to store predictions and true values
    all_predictions = []
    all_true_values = []
    all_game_ids = []
    all_game_dates = []
    all_player_names = []
    all_fanduel_salaries = []
    all_draftkings_salaries = []
    all_yahoo_salaries = []
    all_fanduel_positions = []
    all_draftkings_positions = []
    all_yahoo_positions = []

    unique_dates = sorted(X['game_date'].unique())

    # Start from a date where we have enough data
    start_index = 10  # Adjust based on your data

    cat_cols = ['team_abbreviation', 'player_name', 'opponent', 'pos-draftkings', 'pos-fanduel', 'pos-yahoo', 'season_year']

    for idx in range(start_index, len(unique_dates)):
        current_date = unique_dates[idx]

        # Training data: all data prior to current_date
        df_train = X[X['game_date'] < current_date]

        # For each player, get last 10 games
        df_train = df_train.sort_values(['player_name', 'game_date'])
        df_train = df_train.groupby('player_name').tail(10)

        # Test data: data on current_date
        df_test = X[X['game_date'] == current_date]

        if df_train.empty or df_test.empty:
            continue

        # Prepare features and target variables
        X_train = df_train.drop(columns=['game_date', 'game_id'])
        y_train = y.loc[X_train.index]

        X_test = df_test.drop(columns=['game_date', 'game_id'])
        y_test = y.loc[X_test.index]

        # Ensure categorical columns are properly set
        X_train[cat_cols] = X_train[cat_cols].astype('category')
        X_test[cat_cols] = X_test[cat_cols].astype('category')

        identifying_test_data = df_test[['player_name', 'game_date', 'game_id']]

        # Create DMatrix for training and testing data
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        params = {
            'tree_method': 'hist',  # Use 'gpu_hist' if you have a GPU
            'enable_categorical': True
        }

        # Train the model
        model = xgb.train(params, dtrain)

        # Make predictions
        y_pred = model.predict(dtest)

        # Store the predictions and true values
        all_predictions.extend(y_pred)
        all_true_values.extend(y_test)
        all_game_ids.extend(identifying_test_data['game_id'].tolist())
        all_game_dates.extend(identifying_test_data['game_date'].tolist())
        all_player_names.extend(identifying_test_data['player_name'].tolist())
        all_fanduel_salaries.extend(df_test['salary-fanduel'])
        all_draftkings_salaries.extend(df_test['salary-draftkings'])
        all_yahoo_salaries.extend(df_test['salary-yahoo'])
        all_fanduel_positions.extend(df_test['pos-fanduel'])
        all_draftkings_positions.extend(df_test['pos-draftkings'])
        all_yahoo_positions.extend(df_test['pos-yahoo'])

        # Save the model if requested
        if save_model:
            model_filename = f'{model_dir}/model_date_{current_date}.pkl'
            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f'Training up to date: {current_date}')
        print(f'Mean Squared Error (MSE): {mse:.2f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
        print(f'R-squared (R²): {r2:.2f}')
        print('')

    results_df = pd.DataFrame({
        'player_name': all_player_names,
        'game_id': all_game_ids,
        'game_date': all_game_dates,
        'y': all_true_values,
        'y_pred': all_predictions,
        'fanduel_salary': all_fanduel_salaries,
        'draftkings_salary': all_draftkings_salaries,
        'yahoo_salary': all_yahoo_salaries,
        'fanduel_position': all_fanduel_positions,
        'draftkings_position': all_draftkings_positions,
        'yahoo_position': all_yahoo_positions,
    })

    return results_df


def predict_fp(df, rolling_window=10, season_year=None, start_date=None, end_date=None):
    """
    Predict fantasy points using rolling training and testing. Allows filtering by season year and date range.

    Args:
        df (pd.DataFrame): The input DataFrame containing player and game data.
        rolling_window (int): The rolling window size for features.
        season_year (str): Specific season year to filter (e.g., "2022-2023"). If None, includes all seasons.
        start_date (str): Start date to filter data (e.g., "2023-01-01"). If None, includes all start dates.
        end_date (str): End date to filter data (e.g., "2023-04-01"). If None, includes all end dates.

    Returns:
        pd.DataFrame: DataFrame containing predictions and results for the specified filters.
    """
    df.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)  # Handle potential missing column
    df['game_date'] = pd.to_datetime(df['game_date'])
    df.sort_values(['game_date'], ascending=True, inplace=True)

    # Apply filters to the DataFrame based on input parameters
    if season_year:
        df = df[df['season_year'] == season_year]
    if start_date:
        df = df[df['game_date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['game_date'] <= pd.to_datetime(end_date)]

    if df.empty:
        raise ValueError("No data available after applying filters. Please adjust the filters.")

    all_seasons_start_years = [s.split("-")[0] for s in df['season_year'].unique()]
    results_list = []

    for current_season_year in df['season_year'].unique():
        print(f"Processing season year: {current_season_year}")
        season_df = df[df['season_year'] == current_season_year].copy()

        # Apply last season stats
        season_start_year = current_season_year.split("-")[0]
        prev_season_start_year = str(int(season_start_year) - 1)

        if prev_season_start_year in all_seasons_start_years:
            prev_season_df = df[df['season_year'] == f'{prev_season_start_year}-{season_start_year}']
            season_df = add_last_season_data_with_extras(season_df, prev_season_df)

        season_df = add_running_season_stats(season_df)

        # Clean numeric columns and add time-dependent features
        season_df = clean_numeric_columns(season_df, same_game_cols)
        season_df = add_time_dependent_features_v2(season_df, rolling_window=rolling_window)

        # Prepare features and target variables
        features = season_df.columns.difference(same_game_cols).tolist()
        combined_df = pd.DataFrame()

        for cat in dfs_cats:
            target = cat
            X = season_df[features]
            y = season_df[target]

            print(f"Training models for {cat} in season {current_season_year}")
            cat_results = rolling_train_test(X=X, y=y)
            cat_results.rename(columns={'y': cat, 'y_pred': f'{cat}_pred'}, inplace=True)

            if combined_df.empty:
                combined_df = cat_results
            else:
                combined_df = pd.merge(
                    combined_df,
                    cat_results,
                    on=['player_name', 'game_date', 'game_id', 'fanduel_salary', 'draftkings_salary', 'yahoo_salary',
                        'fanduel_position', 'draftkings_position', 'yahoo_position'],
                    how='outer',
                    suffixes=('', f'_{cat}'))

        # Compute fantasy points for the season
        combined_df['fp_fanduel'] = combined_df.apply(lambda row: calculate_fp_fanduel(row), axis=1)
        combined_df['fp_fanduel_pred'] = combined_df.apply(lambda row: calculate_fp_fanduel(row, pred_mode=True),
                                                           axis=1)

        combined_df['fp_yahoo'] = combined_df.apply(calculate_fp_yahoo, axis=1)
        combined_df['fp_yahoo_pred'] = combined_df.apply(lambda row: calculate_fp_yahoo(row, pred_mode=True), axis=1)

        combined_df['fp_draftkings'] = combined_df.apply(calculate_fp_draftkings, axis=1)
        combined_df['fp_draftkings_pred'] = combined_df.apply(lambda row: calculate_fp_draftkings(row, pred_mode=True),
                                                              axis=1)

        results_list.append(combined_df)

    # Concatenate all season results
    final_results = pd.concat(results_list, ignore_index=True)

    # Generate a descriptive output filename based on filters
    res_name = f'fp_xgb_daily_pred'
    if season_year:
        res_name += f'_{season_year}'
    if start_date:
        res_name += f'_from_{start_date}'
    if end_date:
        res_name += f'_to_{end_date}'
    res_name += '.csv'

    final_results.to_csv(f'output_csv/{res_name}', index=False)
    return final_results
