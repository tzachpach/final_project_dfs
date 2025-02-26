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
    """
    Adds last season aggregates and additional stats to current_df using prev_df (the previous season).
    """
    all_cats = dfs_cats + ['fp_fanduel', 'fp_yahoo', 'fp_draftkings']

    # Predefine new columns in current_df (important for consistent structure)
    for cat in all_cats:
        current_df[f'last_season_avg_{cat}'] = 0  # Initialize with 0
    current_df['last_season_games_played'] = 0
    current_df['last_season_double_doubles'] = 0
    current_df['last_season_triple_doubles'] = 0
    for col in thresholds_for_exceptional_games.keys():
        current_df[f'last_season_{col}_exceptional_games'] = 0


    # --- Check if prev_df is empty ---
    if prev_df.empty:
        print("Warning: Previous season DataFrame is empty.  Returning current_df without last season stats.")
        return current_df  # Return early, already initialized

    # --- Handle players that do not exist in prev season
    prev_players = prev_df['player_name'].unique()
    current_players = current_df['player_name'].unique()
    new_players = set(current_players) - set(prev_players)
    missing_players_df = pd.DataFrame() # Initialize empty DataFrame

    if new_players:
        # Create a DataFrame for new players with last_season columns filled with 0
        new_players_data = []
        for player in new_players:
            player_data = {'player_name': player}
            for col in all_cats:
                player_data[f'last_season_avg_{col}'] = 0
            player_data['last_season_games_played'] = 0
            player_data['last_season_double_doubles'] = 0
            player_data['last_season_triple_doubles'] = 0
            for col in thresholds_for_exceptional_games.keys():
                player_data[f'last_season_{col}_exceptional_games'] = 0
            new_players_data.append(player_data)

        missing_players_df = pd.DataFrame(new_players_data)

    # Calculate aggregate stats for the previous season
    agg_cols = {f'last_season_avg_{cat}': (cat, 'mean') for cat in all_cats}
    agg_cols['last_season_games_played'] = ('game_id', 'count')
    agg_data = prev_df.groupby('player_name').agg(**agg_cols).reset_index()


    # Calculate exceptional games and doubles
    doubles_data = []
    for player_name, group in prev_df.groupby('player_name'):
        result = calculate_exceptional_games_and_doubles(group, thresholds=thresholds_for_exceptional_games)
        doubles_data.append({'player_name': player_name, **result})
    doubles_df = pd.DataFrame(doubles_data)

    # --- Handle empty doubles_df CORRECTLY ---
    if doubles_df.empty:
        # If there are NO double-doubles/triple-doubles, create those columns in agg_data and set to 0
        agg_data['last_season_double_doubles'] = 0
        agg_data['last_season_triple_doubles'] = 0
        for stat in thresholds_for_exceptional_games:
            agg_data[f'last_season_{stat}_exceptional_games'] = 0
    else:
        # --- Rename and Merge (ONLY if doubles_df is NOT empty) ---
        rename_cols = {
            col: f'last_season_{col}'
            for col in doubles_df.columns
            if col != 'player_name'
        }
        doubles_df = doubles_df.rename(columns=rename_cols)
        agg_data = agg_data.merge(doubles_df, on='player_name', how='left')

    # --- Combine with missing players (if any) ---
    if not missing_players_df.empty:
         agg_data = pd.concat([agg_data, missing_players_df], ignore_index=True).fillna(0)

    # --- Merge with current_df ---
    # Left merge is CRUCIAL here. We want to keep ALL rows from current_df,
    # and add data from agg_data where there's a match on 'player_name'.
    for col in agg_data.columns:
        if col != 'player_name':
            stat_mapping = agg_data.set_index('player_name')[col]
            current_df[col] = current_df['player_name'].map(stat_mapping).fillna(0)

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