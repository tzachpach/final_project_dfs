import logging

import pandas as pd

rolling_window = 10

dfs_cats = ['reb', 'pts', 'ast', 'stl', 'blk', 'tov']
same_game_cols = ['minutes_played', 'fgm', 'fga', 'fg_pct', 'fg3m', 'fg3a', 'fg3_pct', 'ftm', 'fta', 'ft_pct', 'oreb',
                   'dreb', 'reb', 'ast', 'tov', 'stl', 'blk', 'blka', 'pf', 'pfd', 'pts', 'plus_minus',
                   'nba_fantasy_pts', 'dd2', 'td3', 'wnba_fantasy_pts', 'available_flag', 'e_off_rating', 'off_rating',
                   'sp_work_off_rating', 'e_def_rating', 'def_rating', 'sp_work_def_rating', 'e_net_rating',
                   'net_rating', 'sp_work_net_rating', 'ast_pct', 'ast_to', 'ast_ratio', 'oreb_pct', 'dreb_pct',
                   'reb_pct', 'tm_tov_pct', 'e_tov_pct', 'efg_pct', 'ts_pct', 'usg_pct_x', 'e_usg_pct', 'e_pace',
                   'pace', 'pace_per40', 'sp_work_pace', 'pie', 'poss', 'fgm_pg', 'fga_pg', 'pct_fga_2pt',
                   'pct_fga_3pt', 'pct_pts_2pt', 'pct_pts_2pt_mr', 'pct_pts_3pt', 'pct_pts_fb', 'pct_pts_ft',
                   'pct_pts_off_tov', 'pct_pts_paint', 'pct_ast_2pm', 'pct_uast_2pm', 'pct_ast_3pm', 'pct_uast_3pm',
                   'pct_ast_fgm', 'pct_uast_fgm', 'pct_fgm', 'pct_fga', 'pct_fg3m', 'pct_fg3a', 'pct_ftm', 'pct_fta',
                   'pct_oreb', 'pct_dreb', 'pct_reb', 'pct_ast', 'pct_tov', 'pct_stl', 'pct_blk', 'pct_blka', 'pct_pf',
                   'pct_pfd', 'pct_pts', 'usage_rate', 'fp_draftkings', 'fp_fanduel',
                   'fp_yahoo', 'is_wl']

thresholds_for_exceptional_games = {
    'pts': 20,  # High-scoring game threshold
    'reb': 10,  # High-rebounding game threshold
    'ast': 8,  # High-assist game threshold
    'stl': 5,  # High-steal game threshold
    'blk': 5,  # High-block game threshold
    'tov': 5,  # High-turnover game threshold
    'fp_draftkings': 50,  # Exceptional DFS performance for DraftKings
    'fp_fanduel': 45,  # Exceptional DFS performance for FanDuel
    'fp_yahoo': 40,  # Exceptional DFS performance for Yahoo
}


def calculate_fp_fanduel(row, pred_mode=False):
    pred = '_pred' if pred_mode else ''
    return (row[f'pts{pred}'] +
            row[f'reb{pred}'] * 1.2 +
            row[f'ast{pred}'] * 1.5 +
            row[f'stl{pred}'] * 3 +
            row[f'blk{pred}'] * 3 -
            row[f'tov{pred}'] * 1)


def calculate_fp_yahoo(row, pred_mode=False):
    pred = '_pred' if pred_mode else ''
    return (row[f'pts{pred}'] +
            row[f'reb{pred}'] * 1.2 +
            row[f'ast{pred}'] * 1.5 +
            row[f'stl{pred}'] * 3 +
            row[f'blk{pred}'] * 3 -
            row[f'tov{pred}'] * 1)


def calculate_fp_draftkings(row, pred_mode=False):
    pred = '_pred' if pred_mode else ''
    fp = (row[f'pts{pred}'] +
          row[f'reb{pred}'] * 1.25 +
          row[f'ast{pred}'] * 1.5 +
          row[f'stl{pred}'] * 2 +
          row[f'blk{pred}'] * 2 -
          row[f'tov{pred}'] * 0.5)

    # Calculate Double-Double and Triple-Double bonuses
    stats = [row[f'pts{pred}'], row[f'reb{pred}'], row[f'ast{pred}'], row[f'stl{pred}'], row[f'blk{pred}']]
    double_double = sum([1 for stat in stats if stat >= 10]) >= 2
    triple_double = sum([1 for stat in stats if stat >= 10]) >= 3

    if double_double:
        fp += 1.5
    if triple_double:
        fp += 3

    return fp

def clean_numeric_columns(df, columns):
    """
    Convert columns to numeric, forcing errors to NaN, and handle specific non-numeric values.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert non-numeric to NaN
    return df

def add_time_dependent_features(df, rolling_window):
    for col in same_game_cols:
        logging.info(f"Adding features to {col}")
        gb = df.groupby('player_name')[col]
        df[f'{col}_rolling_{rolling_window}_day_avg'] = gb.transform(
            lambda x: x.rolling(rolling_window, min_periods=1).mean())
        df[f'{col}_rolling_{rolling_window}_day_std'] = gb.transform(
            lambda x: x.rolling(rolling_window, min_periods=1).std())
        df[f'{col}_lag_1'] = gb.shift(1)
        df[f'{col}_lag_2'] = gb.shift(2)
        df[f'{col}_lag_3'] = gb.shift(3)
        df[f'{col}_diff_1'] = gb.diff(1)
        df[f'{col}_diff_2'] = gb.diff(2)
        df[f'{col}_diff_3'] = gb.diff(3)
    return df


def add_time_dependent_features_v2(df, rolling_window):
    # Sort the DataFrame to ensure correct order for rolling calculations
    df = df.sort_values(['player_name', 'game_date']).reset_index(drop=True)

    # Initialize a list to collect new features
    new_features_list = []

    # Group by 'player_name'
    grouped = df.groupby('player_name')

    # For each group (player), compute rolling features
    for name, group in grouped:
        # Ensure the group is sorted by 'game_date'
        group = group.sort_values('game_date').reset_index(drop=True)
        # Initialize a DataFrame to hold features for this group
        features = pd.DataFrame(index=group.index)

        # Rolling mean and std
        rolling = group[same_game_cols].rolling(window=rolling_window, min_periods=1)
        rolling_mean = rolling.mean()
        rolling_std = rolling.std()

        # Rename columns
        rolling_mean.columns = [f'{col}_rolling_{rolling_window}_day_avg' for col in same_game_cols]
        rolling_std.columns = [f'{col}_rolling_{rolling_window}_day_std' for col in same_game_cols]

        # Collect rolling features
        features = pd.concat([features, rolling_mean, rolling_std], axis=1)

        # Lags and diffs
        for lag in [1, 2, 3]:
            lag_features = group[same_game_cols].shift(lag)
            lag_features.columns = [f'{col}_lag_{lag}' for col in same_game_cols]
            diff_features = group[same_game_cols].diff(lag)
            diff_features.columns = [f'{col}_diff_{lag}' for col in same_game_cols]

            # Concatenate lag and diff features
            features = pd.concat([features, lag_features, diff_features], axis=1)

        # Add 'player_name' and 'game_date' to features DataFrame
        features['player_name'] = name
        features['game_date'] = group['game_date'].values

        # Append the features DataFrame to the list
        new_features_list.append(features)

    # Concatenate all the features into a single DataFrame
    new_features_df = pd.concat(new_features_list, ignore_index=True)

    # Merge the new features DataFrame with the original DataFrame
    df = pd.merge(df, new_features_df, on=['player_name', 'game_date'], how='left')

    return df

def calculate_exceptional_games_and_doubles(group, thresholds):
    """Calculate exceptional games, double-doubles, and triple-doubles for a group of games."""
    results = {}

    # Exceptional games counts based on thresholds
    for stat, threshold in thresholds.items():
        exceptional_col = f'{stat}_exceptional_games'
        results[exceptional_col] = (group[stat] >= threshold).sum()

    # Double-doubles and triple-doubles
    double_double = ((group[['pts', 'reb', 'ast', 'stl', 'blk']] >= 10).sum(axis=1) >= 2).sum()
    triple_double = ((group[['pts', 'reb', 'ast', 'stl', 'blk']] >= 10).sum(axis=1) >= 3).sum()
    results['double_doubles'] = double_double
    results['triple_doubles'] = triple_double

    return pd.Series(results)


def add_last_season_data_with_extras(df):
    """ Add last season aggregates and additional stats to the DataFrame """
    all_cats = dfs_cats + ['fp_fanduel', 'fp_yahoo', 'fp_draftkings']

    # Predefine all new columns with None values
    for cat in all_cats:
        df[f'last_season_avg_{cat}'] = None
    df['last_season_games_played'] = None
    df['last_season_double_doubles'] = None
    df['last_season_triple_doubles'] = None

    for col in thresholds_for_exceptional_games.keys():
        df[f'last_season_{col}_exceptional_games'] = None

    # Add column for last season year
    df['last_season_year_start'] = df['season_year'].apply(lambda x: str(int(x.split("-")[0]) - 1))
    prev_season_start_year = [s.split("-")[0] for s in df['season_year'].unique()]

    for season_year in df['season_year'].unique():
        last_season_year = str(int(season_year.split("-")[0]) - 1)

        if last_season_year not in prev_season_start_year:
            continue

        # Filter rows for the current season and last season
        last_season_filter = (df['last_season_year_start'] == last_season_year)

        last_season_data = df[last_season_filter]

        if last_season_data.empty:
            continue

        # Calculate aggregate stats for last season
        agg_cols = {f'last_season_avg_{cat}': (cat, 'mean') for cat in all_cats}
        agg_cols['last_season_games_played'] = ('game_id', 'count')

        agg_data = last_season_data.groupby(['player_name', 'team_abbreviation']).agg(**agg_cols).reset_index()

        # Calculate exceptional games and doubles
        doubles_data = last_season_data.groupby(['player_name', 'team_abbreviation']).apply(
            calculate_exceptional_games_and_doubles, thresholds=thresholds_for_exceptional_games
        ).reset_index()

        doubles_data.columns = ['player_name', 'team_abbreviation'] + [
            f'last_season_{col}' for col in doubles_data.columns if col not in ['player_name', 'team_abbreviation']
        ]

        # Merge doubles data into aggregated data
        agg_data = agg_data.merge(doubles_data, on=['player_name', 'team_abbreviation'], how='left')

        # Get current season rows
        current_season_mask = df['season_year'] == season_year

        # For each column in agg_data (except player_name and team_abbreviation)
        for col in agg_data.columns:
            if col not in ['player_name', 'team_abbreviation']:
                # Create a mapping series from player_name and team_abbreviation to the stat
                stat_mapping = agg_data.set_index(['player_name', 'team_abbreviation'])[col]

                # Create a temporary index for the current season data
                temp_idx = df[current_season_mask].set_index(['player_name', 'team_abbreviation']).index

                # Map the values and update the original dataframe
                df.loc[current_season_mask, col] = temp_idx.map(stat_mapping)

    # Fill NaN values in newly added columns only
    last_season_cols = [col for col in df.columns if col.startswith('last_season_')]
    df[last_season_cols] = df[last_season_cols].fillna(0)

    return df

def add_last_season_data_with_extras(current_df, prev_df):
    """ Add last season aggregates and additional stats to the current DataFrame using data from the previous season. """
    all_cats = dfs_cats + ['fp_fanduel', 'fp_yahoo', 'fp_draftkings']

    # Predefine all new columns in current_df with None values
    for cat in all_cats:
        current_df[f'last_season_avg_{cat}'] = None
    current_df['last_season_games_played'] = None
    current_df['last_season_double_doubles'] = None
    current_df['last_season_triple_doubles'] = None

    for col in thresholds_for_exceptional_games.keys():
        current_df[f'last_season_{col}_exceptional_games'] = None

    # Calculate aggregate stats for the previous season
    agg_cols = {f'last_season_avg_{cat}': (cat, 'mean') for cat in all_cats}
    agg_cols['last_season_games_played'] = ('game_id', 'count')

    agg_data = prev_df.groupby(['player_name', 'team_abbreviation']).agg(**agg_cols).reset_index()

    # Calculate exceptional games and doubles
    doubles_data = prev_df.groupby(['player_name', 'team_abbreviation']).apply(
        calculate_exceptional_games_and_doubles, thresholds=thresholds_for_exceptional_games
    ).reset_index()

    doubles_data.columns = ['player_name', 'team_abbreviation'] + [
        f'last_season_{col}' for col in doubles_data.columns if col not in ['player_name', 'team_abbreviation']
    ]

    # Merge doubles data into aggregated data
    agg_data = agg_data.merge(doubles_data, on=['player_name', 'team_abbreviation'], how='left')

    # Update current_df with last season's stats
    for col in agg_data.columns:
        if col not in ['player_name', 'team_abbreviation']:
            # Map values from agg_data to current_df based on player_name and team_abbreviation
            stat_mapping = agg_data.set_index(['player_name', 'team_abbreviation'])[col]
            temp_idx = current_df.set_index(['player_name', 'team_abbreviation']).index
            current_df[col] = temp_idx.map(stat_mapping)

    # Fill NaN values in newly added columns only
    last_season_cols = [col for col in current_df.columns if col.startswith('last_season_')]
    current_df[last_season_cols] = current_df[last_season_cols].fillna(0)

    return current_df



def add_running_season_stats(df):
    """ Add running season aggregates and stats up to but not including the current game """
    all_cats = dfs_cats + ['fp_fanduel', 'fp_yahoo', 'fp_draftkings']

    # Predefine all new columns with zeros (more efficient than None)
    for cat in all_cats:
        df[f'running_season_avg_{cat}'] = 0
        df[f'running_season_total_{cat}'] = 0
    df['running_season_games_played'] = 0
    df['running_season_double_doubles'] = 0
    df['running_season_triple_doubles'] = 0

    for col in thresholds_for_exceptional_games.keys():
        df[f'running_season_{col}_exceptional_games'] = 0

    # Sort the entire dataframe once instead of multiple times
    df = df.sort_values(['player_name', 'team_abbreviation', 'season_year', 'game_date'])

    # Process each group more efficiently
    for (player_name, team_abbreviation, season_year), group in df.groupby(
            ['player_name', 'team_abbreviation', 'season_year'], observed=True):

        group_idx = group.index

        # Vectorized operations for all stats at once
        for cat in all_cats:
            # Calculate running averages and totals
            cumsum = group[cat].cumsum()
            cumcount = pd.Series(range(1, len(group) + 1), index=group.index)

            # Shift to exclude current game
            df.loc[group_idx, f'running_season_total_{cat}'] = cumsum.shift(1).fillna(0)
            df.loc[group_idx, f'running_season_avg_{cat}'] = (cumsum.shift(1) / cumcount.shift(1)).fillna(0)

        # Calculate games played (vectorized)
        df.loc[group_idx, 'running_season_games_played'] = pd.Series(range(len(group)), index=group_idx).shift(
            1).fillna(0)

        # Calculate double-doubles and triple-doubles more efficiently
        stats_matrix = group[['pts', 'reb', 'ast', 'stl', 'blk']] >= 10
        double_doubles = (stats_matrix.sum(axis=1) >= 2).cumsum().shift(1)
        triple_doubles = (stats_matrix.sum(axis=1) >= 3).cumsum().shift(1)

        df.loc[group_idx, 'running_season_double_doubles'] = double_doubles.fillna(0)
        df.loc[group_idx, 'running_season_triple_doubles'] = triple_doubles.fillna(0)

        # Calculate exceptional games more efficiently
        for stat, threshold in thresholds_for_exceptional_games.items():
            exceptional_games = (group[stat] >= threshold).cumsum().shift(1)
            df.loc[group_idx, f'running_season_{stat}_exceptional_games'] = exceptional_games.fillna(0)

    return df

