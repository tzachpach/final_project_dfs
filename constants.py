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
