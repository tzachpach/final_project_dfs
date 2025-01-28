import pandas as pd

from config.dfs_categories import same_game_cols


def clean_numeric_columns(df, columns):
    """
    Convert columns to numeric, forcing errors to NaN, and handle specific non-numeric values.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert non-numeric to NaN
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