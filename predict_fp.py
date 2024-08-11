import logging

import numpy as np

import pandas as pd

import xgboost as xgb
import os
import pickle
from sklearn.metrics import mean_squared_error, r2_score

from constants import *


def assign_league_weeks(df):
    # Ensure 'season_year' and 'game_date' are available and properly formatted
    df['week'] = df['game_date'].dt.isocalendar().week
    df['season_start'] = df.groupby('season_year')['game_date'].transform('min')
    df['season_week'] = ((df['game_date'] - df['season_start']).dt.days // 7) + 1
    df = df.drop(columns=['season_start', 'week'])
    df = df.rename(columns={'season_week': 'league_week'})
    return df


def rolling_train_test(X, y, df, num_weeks_for_training=4, save_model=False, model_dir='models'):
    # Ensure the model directory exists
    os.makedirs(model_dir, exist_ok=True)

    # Initialize lists to store predictions and true values
    all_predictions = []
    all_true_values = []
    all_game_ids = []
    all_game_dates = []
    all_player_ids = []
    all_fanduel_salaries = []
    all_draftkings_salaries = []
    all_yahoo_salaries = []
    all_fanduel_positions = []
    all_draftkings_positions = []
    all_yahoo_positions = []


    # Iterate over each league week, testing on the current week and training on the previous 4 weeks
    unique_weeks = df['league_week'].unique()
    for current_week in unique_weeks:
        start_week = current_week - num_weeks_for_training
        training_weeks = list(range(start_week, current_week))

        # Select training data (previous 4 weeks)
        X_train = X[X['league_week'].isin(training_weeks)]
        y_train = y.loc[X_train.index]

        # Select test data (current week)
        X_test = X[X['league_week'] == current_week]
        y_test = y.loc[X_test.index]

        if X_train.empty or X_test.empty:
            continue

        identifying_test_data = X_test[['player_name', 'game_date', 'game_id']]

        # Remove non-feature columns
        X_train = X_train.drop(columns=['game_date', 'game_id'])
        X_test = X_test.drop(columns=['game_date', 'game_id'])

        # Create DMatrix for training and testing data
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

        # Specify the booster parameters with tree_method set to 'gpu_hist' if using GPU
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
        all_game_ids.extend(list(identifying_test_data['game_id']))
        all_game_dates.extend(list(identifying_test_data['game_date']))
        all_player_ids.extend(list(identifying_test_data['player_name']))
        all_fanduel_salaries.extend(X_test['salary-draftkings'])
        all_draftkings_salaries.extend(X_test['salary-draftkings'])
        all_yahoo_salaries.extend(X_test['salary-yahoo'])
        all_fanduel_positions.extend(X_test['pos-draftkings'])
        all_draftkings_positions.extend(X_test['pos-draftkings'])
        all_yahoo_positions.extend(X_test['pos-yahoo'])

        # Save the model if requested
        if save_model:
            model_filename = f'{model_dir}/model_week_{current_week}_trained_on_{start_week}_to_{current_week - 1}.pkl'
            with open(model_filename, 'wb') as file:
                pickle.dump(model, file)

        # Calculate evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Print the evaluation metrics
        print(f'Training weeks: {training_weeks}')
        print(f'Test week: {current_week}')
        print(f'Mean Squared Error (MSE): {mse:.2f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
        print(f'R-squared (R²): {r2:.2f}')
        print('')

    # Combine predictions and true values into a DataFrame
    results_df = pd.DataFrame({
        'player_name': all_player_ids,
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


def add_time_dependent_features(df, rolling_window):
    # Add moving average columns:
    for col in same_game_cols:
        logging.info(f"Adding features to {col}")
        gb = df.groupby('player_name')[col]
        df[f'{col}_rolling_{rolling_window}_day_avg'] = gb.transform(lambda x: x.rolling(rolling_window, min_periods=1).mean())
        df[f'{col}_rolling_{rolling_window}_day_std'] = gb.transform(lambda x: x.rolling(rolling_window, min_periods=1).std())
        df[f'{col}_lag_1'] = gb.shift(1)
        df[f'{col}_lag_2'] = gb.shift(2)
        df[f'{col}_lag_3'] = gb.shift(3)
        df[f'{col}_diff_1'] = gb.diff(1)
        df[f'{col}_diff_2'] = gb.diff(2)
        df[f'{col}_diff_3'] = gb.diff(3)
    return df


def predict_fp(df, rolling_window=rolling_window):
    df = df.drop('Unnamed: 0', axis=1)
    # Preprocess specific columns
    df['game_date'] = pd.to_datetime(df['game_date'])

    cat_cols = ['team_abbreviation', 'player_name', 'opponent', 'pos-draftkings', 'pos-fanduel', 'pos-yahoo']
    df[cat_cols] = df[cat_cols].astype('category')

    # Reverse the order of the df
    df = df.sort_values(['game_date'], ascending=True)
    df = assign_league_weeks(df)

    df = clean_numeric_columns(df, same_game_cols)

    df = add_time_dependent_features(df, rolling_window=rolling_window)

    total_results = {}
    cross_season_df = pd.DataFrame()
    for season in df['season_year'].unique():
        season_df = df[df['season_year'] == season]
        season_df = season_df.drop('season_year', axis=1)


        for cat in dfs_cats:
            target = cat
            target_related_cols = same_game_cols
            features = season_df.columns.difference(target_related_cols).tolist()

            # Split data into features and target
            X = season_df[features]
            y = season_df[target]

            # Train the model
            print(f'Training models for {cat}')
            print('---------------------------------')
            results = rolling_train_test(X=X, y=y, df=season_df)
            results.to_csv(f'output_csv/{cat}_{season}_results.csv', index=False)
            total_results.update({f'{cat}_{season}': results})

    dffs = []
    # Iterate over the dictionary items
    for key, dff in total_results.items():
        prefix = key.split('_')[0]
        dff.rename(columns={'y': prefix, 'y_pred': f'{prefix}_pred'}, inplace=True)
        dffs.append(dff)

    combined_df = dffs[0]

    for df in dffs[1:]:
        combined_df = pd.merge(combined_df, df, on=['player_name', 'game_date', 'game_id','fanduel_salary', 'draftkings_salary', 'yahoo_salary', 'draftkings_position', 'fanduel_position', 'yahoo_position'], suffixes=('', f'_{df.columns.name}'))

    # Add FP calculations:
    combined_df['fp_fanduel'] = combined_df.apply(lambda row: calculate_fp_fanduel(row), axis=1)
    combined_df['fp_fanduel_pred'] = combined_df.apply(lambda row: calculate_fp_fanduel(row, pred_mode=True), axis=1)

    combined_df['fp_yahoo'] = combined_df.apply(calculate_fp_yahoo, axis=1)
    combined_df['fp_yahoo_pred'] = combined_df.apply(lambda row: calculate_fp_yahoo(row, pred_mode=True), axis=1)

    combined_df['fp_draftkings'] = combined_df.apply(calculate_fp_draftkings, axis=1)
    combined_df['fp_draftkings_pred'] = combined_df.apply(lambda row: calculate_fp_draftkings(row, pred_mode=True),
                                                      axis=1)
    cross_season_df = pd.concat([cross_season_df, combined_df])
    return cross_season_df


df = pd.read_csv('data/gamelogs_salaries_all_seasons_merged.csv')
res = predict_fp(df)
res.to_csv('fp_pred.csv')
