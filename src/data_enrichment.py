import pandas as pd

from config.constants import thresholds_for_exceptional_games
from config.dfs_categories import dfs_cats, same_game_cols
from config.fantasy_point_calculation import calculate_exceptional_games_and_doubles


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

    # Calculate exceptional games and doubles using a modified approach
    doubles_data = []
    for (player, team), group in prev_df.groupby(['player_name', 'team_abbreviation']):
        result = calculate_exceptional_games_and_doubles(group, thresholds=thresholds_for_exceptional_games)
        doubles_data.append({
            'player_name': player,
            'team_abbreviation': team,
            **result
        })

    doubles_df = pd.DataFrame(doubles_data)

    if doubles_df.empty:
        # Handle the case where no doubles data was found
        return current_df

    # Rename columns to add 'last_season_' prefix
    rename_cols = {col: f'last_season_{col}' for col in doubles_df.columns
                   if col not in ['player_name', 'team_abbreviation']}
    doubles_df = doubles_df.rename(columns=rename_cols)

    # Merge doubles data into aggregated data
    agg_data = agg_data.merge(doubles_df, on=['player_name', 'team_abbreviation'], how='left')

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