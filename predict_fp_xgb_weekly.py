import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import pickle
from sklearn.metrics import mean_squared_error, r2_score

from constants import *
from preprocess_merged_csvs import merge_all_seasons


def assign_league_weeks(df):
    # Assign league weeks based on the game date and season year
    # df['week'] = df['game_date'].dt.isocalendar().week
    df['season_start'] = df.groupby('season_year')['game_date'].transform('min')
    df['season_week'] = ((df['game_date'] - df['season_start']).dt.days // 7) + 1
    df = df.drop(columns=['season_start'])
    df = df.rename(columns={'season_week': 'league_week'})
    return df


def rolling_train_test(X, y, df, num_weeks_for_training=4, save_model=False, model_dir='models'):
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
        X_train = X_train.drop(columns=['game_date', 'game_id'])
        X_test = X_test.drop(columns=['game_date', 'game_id'])

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

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        print(f'Training weeks: {training_weeks}')
        print(f'Test week: {current_week}')
        print(f'Mean Squared Error (MSE): {mse:.2f}')
        print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
        print(f'R-squared (R²): {r2:.2f}')
        print('')

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

def predict_fp(df, rolling_window=rolling_window, three_months_only=True):
    df = df.drop('Unnamed: 0', axis=1)
    df['game_date'] = pd.to_datetime(df['game_date'])

    cat_cols = ['team_abbreviation', 'player_name', 'opponent', 'pos-draftkings', 'pos-fanduel', 'pos-yahoo']
    df[cat_cols] = df[cat_cols].astype('category')

    df = df.sort_values(['game_date'], ascending=True)
    if three_months_only:
        df = df[(df['game_date'] >= '2023-01-01') & (df['game_date'] < '2023-04-01')]
    df = assign_league_weeks(df)
    df = clean_numeric_columns(df, same_game_cols)
    df = add_time_dependent_features(df, rolling_window=rolling_window)

    all_seasons_results = []

    for season in df['season_year'].unique():
        season_df = df[df['season_year'] == season]
        season_df = season_df.drop('season_year', axis=1)
        season_results = pd.DataFrame()

        for cat in dfs_cats:
            target = cat
            features = season_df.columns.difference(same_game_cols).tolist()

            X = season_df[features]
            y = season_df[target]

            print(f'Training models for {cat}')
            print('---------------------------------')
            cat_results = rolling_train_test(X=X, y=y, df=season_df)
            cat_results.rename(columns={'y': cat, 'y_pred': f'{cat}_pred'}, inplace=True)
            if len(season_results) == 0:
                season_results = cat_results
            else:
                season_results = pd.merge(
                    season_results,
                    cat_results,
                    on=['player_name', 'game_date', 'game_id', 'fanduel_salary', 'draftkings_salary', 'yahoo_salary', 'draftkings_position', 'fanduel_position', 'yahoo_position'],
                    suffixes=('', f'_{season_df.columns.name}'))
            cat_results.to_csv(f'output_csv/{cat}_{season}_results.csv', index=False)

        all_seasons_results.append(season_results)

    combined_df = pd.concat(all_seasons_results, ignore_index=True)
    combined_df['fp_fanduel'] = combined_df.apply(lambda row: calculate_fp_fanduel(row), axis=1)
    combined_df['fp_fanduel_pred'] = combined_df.apply(lambda row: calculate_fp_fanduel(row, pred_mode=True), axis=1)

    combined_df['fp_yahoo'] = combined_df.apply(calculate_fp_yahoo, axis=1)
    combined_df['fp_yahoo_pred'] = combined_df.apply(lambda row: calculate_fp_yahoo(row, pred_mode=True), axis=1)

    combined_df['fp_draftkings'] = combined_df.apply(calculate_fp_draftkings, axis=1)
    combined_df['fp_draftkings_pred'] = combined_df.apply(lambda row: calculate_fp_draftkings(row, pred_mode=True),
                                                           axis=1)
    res_name = 'fp_xgb_pred_three_months_only' if three_months_only else 'fp_xgb_pred'
    combined_df.to_csv(f'output_csv/{res_name}.csv', index=False)
    return combined_df


df = merge_all_seasons()
res = predict_fp(df, three_months_only=True)
