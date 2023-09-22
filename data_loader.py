import os

import numpy as np
import pandas as pd
from fuzzywuzzy import process

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def data_loader(seasons_to_reduce=0, n=10):
    # Load and preprocess data
    res_df = load_data(seasons_to_reduce)

    logging.info("Loaded basic data.")
    ctg_df = load_data(seasons_to_reduce, ctg_flag=True)
    ctg_df = ctg_df[['player', 'season', 'age', 'pos']]

    logging.info("Loaded ctg data.")

    res_df = standardize_names(res_df, ctg_df)
    logging.info("Standardized names.")

    res_df = pd.merge(res_df, ctg_df, on=['player', 'season'], how='left')
    logging.info("Merged dataframes.")

    res_df['dkfp'] = res_df.apply(calculate_draftkings_fantasy_points, axis=1)
    logging.info("Calculated DKFP.")

    res_df['cost'] = res_df.apply(randomly_assign_a_pricetag, axis=1)
    logging.info("Assigned random price tags.")

    res_df['date'] = pd.to_datetime(res_df['date'])
    # remove playoff games, work and learn only from regular season games:
    res_df['is_reg_season'] = res_df['date'].apply(lambda x: x.month in [10, 11, 12, 1, 2, 3, 4])
    res_df = res_df[res_df['is_reg_season'] == True]


    res_df = res_df.sort_values(by=['date', 'player'], ascending=False)

    res_df_no_mv = res_df.dropna(how='all')

    gb_player = res_df_no_mv.groupby('player').mean()
    gb_over_10_mins = gb_player[gb_player['mp'] >= n].index
    res_df_no_mv_over_10_mins = res_df_no_mv[res_df_no_mv['player'].isin(gb_over_10_mins)]
    res_df_no_mv_over_10_mins = res_df_no_mv_over_10_mins.reset_index()
    res_df_no_mv_over_10_mins = res_df_no_mv_over_10_mins.drop('index', axis=1)

    logging.info(f'Dropped {len(res_df) - len(res_df_no_mv)} empty rows')
    logging.info(f'Dropped {len(res_df_no_mv) - len(res_df_no_mv_over_10_mins)} players with under {n} minutes')

    logging.info('Number of players in the dataset: {}'.format(len(res_df_no_mv)))

    num_previous_games = [5, 10]
    columns_to_roll = [i.lower() for i in ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB',
                       'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-']]
    rolling_features = {}
    for num_prev_games in num_previous_games:
        for column in columns_to_roll:
            rolling_features.update({f'{column}_prev_{num_prev_games}': res_df_no_mv_over_10_mins.groupby('player')[column].rolling(window=num_prev_games,
                                                                            min_periods=1).mean().reset_index(drop=True)})
    rolling_features_df = pd.DataFrame().from_dict(rolling_features)

    logging.info("Added rolling features.")
    res_df_no_mv_over_10_mins = pd.concat([res_df_no_mv_over_10_mins, rolling_features_df], axis=1)
    train_data = res_df_no_mv_over_10_mins[[col for col in res_df_no_mv_over_10_mins.columns if col not in columns_to_roll]]

    return train_data

def calculate_draftkings_fantasy_points(row):
    fantasy_points = (
            row['pts'] +
            row['3p'] * 0.5 +
            row['trb'] * 1.25 +
            row['ast'] * 1.5 +
            row['stl'] * 2 +
            row['blk'] * 2 -
            row['tov'] * 0.5
    )
    cats_over_10 = sum([row['pts'] >= 10, row['trb'] >= 10, row['ast'] >= 10, row['stl'] >= 10, row['blk'] >= 10])

    if cats_over_10 >= 2:
        fantasy_points += 1.5
    if cats_over_10 >= 3:
        fantasy_points += 3
    if pd.isna(fantasy_points):
        return 0
    return fantasy_points


def randomly_assign_a_pricetag(row):
    random_number = np.random.uniform(0.7, 1.5)
    random_cut = np.random.uniform(-5, 5)
    minutes = 0 if pd.isna(row['mp']) else row['mp']
    points = 0 if pd.isna(row['pts']) else row['pts']
    rebounds = 0 if pd.isna(row['trb']) else row['trb']
    performance = minutes + points + rebounds
    return max(random_number * performance + random_cut, 2)


def load_data(seasons_to_reduce, ctg_flag=False):
    """
    Load data from either the basic dataset or the 'cleaning_the_glass' dataset based on the ctg_flag.

    Parameters:
    - seasons_to_reduce (int): Number of seasons to exclude from the end.
    - ctg_flag (bool): If True, load 'cleaning_the_glass' data. Otherwise, load basic data.

    Returns:
    - DataFrame: Loaded and processed data.
    """

    if ctg_flag:
        directory = 'data/cleaning_the_glass'
        file_prefix = ''
        season_separator = '_'
    else:
        directory = 'data'
        file_prefix = '201'
        season_separator = '-'

    all_year_files = [file for file in os.listdir(directory) if file.endswith('.csv') and file.startswith(file_prefix)]

    df = pd.DataFrame()
    for i in range(len(all_year_files) - seasons_to_reduce):
        if ctg_flag:
            file_name_parts = all_year_files[i].split('.')[0].split(season_separator)
            season = f"{file_name_parts[-2]}/{file_name_parts[-1]}"
        else:
            file_name_parts = all_year_files[i].split('.')[0]
            season = file_name_parts.replace(season_separator, '/')

        season_df = pd.read_csv(os.path.join(directory, all_year_files[i]))
        season_df['season'] = season
        df = pd.concat([df, season_df])

    df.columns = df.columns.str.lower()
    if not ctg_flag:
        df = df.drop('gamelink', axis=1)

    return df


def standardize_names(res_df, ctg_df):
    ctg_df_names = ctg_df['player'].unique()

    # create a dictionary to hold the name mappings
    name_mapping = {}

    # for each name in ctg_df, find the closest matching name in res_df
    for name in res_df['player'].unique():
        # find the closest match in res_df_names
        closest_match, score = process.extractOne(name, ctg_df_names)

        # if the score is above a certain threshold, add it to the name_mapping dictionary
        if score > 90 and name not in ctg_df_names:
            name_mapping[name] = closest_match

    # replace the names in ctg_df
    res_df['player'] = res_df['player'].replace(name_mapping)
    return res_df