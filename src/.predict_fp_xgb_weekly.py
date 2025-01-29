import logging
import numpy as np
import pandas as pd
import xgboost as xgb
import os
import pickle
from sklearn.metrics import mean_squared_error, r2_score

from config.constants import rolling_window
from config.dfs_categories import same_game_cols, dfs_cats
from config.fantasy_point_calculation import calculate_fp_fanduel, calculate_fp_yahoo, calculate_fp_draftkings
from config.feature_engineering import clean_numeric_columns, add_time_dependent_features_v2
from src.test_train_utils import rolling_train_test


def predict_fp(df, rolling_window=rolling_window, three_months_only=True):
    df = df.drop('Unnamed: 0', axis=1)
    df['game_date'] = pd.to_datetime(df['game_date'])

    cat_cols = ['team_abbreviation', 'player_name', 'opponent', 'pos-draftkings', 'pos-fanduel', 'pos-yahoo']
    df[cat_cols] = df[cat_cols].astype('category')

    df = df.sort_values(['game_date'], ascending=True)
    if three_months_only:
        df = df[(df['game_date'] >= '2023-01-01') & (df['game_date'] < '2023-04-01')]
    df = clean_numeric_columns(df, same_game_cols)
    df = add_time_dependent_features_v2(df, rolling_window=rolling_window)

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
            cat_results = rolling_train_test(X, y, df, group_by="week", train_window=4, save_model=True)
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