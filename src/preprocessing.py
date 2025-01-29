import pandas as pd
import os

from config.dfs_categories import same_game_cols
from config.fantasy_point_calculation import calculate_fp_fanduel, calculate_fp_yahoo, calculate_fp_draftkings
from config.feature_engineering import clean_numeric_columns


def preprocess_basic(df):
    """
    Perform basic preprocessing on the dataset, including feature engineering and cleaning.
    """
    df['starter'] = df['starter(y/n)'].apply(lambda x: 1 if x == 'Y' else 0)
    df['venue'] = df['venue(r/h)'].apply(lambda x: 1 if x == 'H' else 0)
    df['is_playoff'] = df['regular/playoffs'].apply(lambda x: 0 if x == 'Regular' else 1)
    df['is_wl'] = df['wl'].apply(lambda x: 1 if x == 'W' else 0)
    df['days_rest_int'] = df['days_rest'].astype(int)

    # Drop unnecessary columns
    cols_to_drop = ['player_id', 'team_id', 'days_rest', 'starter(y/n)', 'venue(r/h)', 'regular/playoffs', 'wl']
    df = df.drop(cols_to_drop, axis=1)

    # Fill missing values for minutes played
    df['minutes_played'] = df['minutes_played'].fillna(0)
    return df


def merge_all_seasons(data_path='data/'):
    """
    Merges all season files into a single DataFrame and performs basic preprocessing.
    """
    dfs_to_merge = [
        pd.read_csv(os.path.join(data_path, f))
        for f in os.listdir(data_path)
        if 'merged_gamelogs_salaries_' in f and f.endswith('.csv')
    ]

    # Combine all data into one DataFrame
    df = pd.concat(dfs_to_merge, ignore_index=True)

    # Clean up unnecessary columns
    df = df.drop('Unnamed: 0', axis=1, errors='ignore')

    # Handle venue column inconsistencies
    df['venue(r/h)'] = df['venue(r/h)'].fillna(df['venue(r/h/n)'])
    df.drop(columns=['venue(r/h/n)'], inplace=True)

    # Drop rows with missing flags
    df = df.dropna(subset=['available_flag'])

    # Apply basic preprocessing
    df = preprocess_basic(df)
    return df


def preprocess_all_seasons_data(all_seasons_df):
    all_seasons_df.drop('Unnamed: 0', axis=1, errors='ignore', inplace=True)  # Handle potential missing column
    all_seasons_df['game_date'] = pd.to_datetime(all_seasons_df['game_date'])
    all_seasons_df.sort_values(['game_date'], ascending=True, inplace=True)
    all_seasons_df = clean_numeric_columns(all_seasons_df, same_game_cols)
    all_seasons_df['fp_fanduel'] = all_seasons_df.apply(lambda row: calculate_fp_fanduel(row), axis=1)
    all_seasons_df['fp_yahoo'] = all_seasons_df.apply(calculate_fp_yahoo, axis=1)
    all_seasons_df['fp_draftkings'] = all_seasons_df.apply(calculate_fp_draftkings, axis=1)
    return all_seasons_df