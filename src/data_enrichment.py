import pandas as pd

from config.constants import thresholds_for_exceptional_games
from config.dfs_categories import dfs_cats, same_game_cols
from config.fantasy_point_calculation import calculate_exceptional_games_and_doubles


def add_time_dependent_features_v2(df, rolling_window):
    # Sort the DataFrame to ensure correct order for rolling calculations
    df = df.sort_values(["player_name", "game_date"]).reset_index(drop=True)

    # Initialize a list to collect new features
    new_features_list = []

    # Group by 'player_name'
    grouped = df.groupby("player_name")

    # For each group (player), compute rolling features
    for name, group in grouped:
        # Ensure the group is sorted by 'game_date'
        group = group.sort_values("game_date").reset_index(drop=True)
        # Initialize a DataFrame to hold features for this group
        features = pd.DataFrame(index=group.index)

        # Rolling mean and std
        rolling = (
            group[same_game_cols].shift(1).rolling(window=rolling_window, min_periods=1)
        )
        rolling_mean = rolling.mean()
        rolling_std = rolling.std()

        # Rename columns
        rolling_mean.columns = [
            f"{col}_rolling_{rolling_window}_day_avg" for col in same_game_cols
        ]
        rolling_std.columns = [
            f"{col}_rolling_{rolling_window}_day_std" for col in same_game_cols
        ]

        # Collect rolling features
        features = pd.concat([features, rolling_mean, rolling_std], axis=1)

        # Lags and diffs
        for lag in [1, 2, 3]:
            lag_features = group[same_game_cols].shift(lag)
            lag_features.columns = [f"{col}_lag_{lag}" for col in same_game_cols]
            diff_features = group[same_game_cols].diff(lag)
            diff_features.columns = [f"{col}_diff_{lag}" for col in same_game_cols]

            # Concatenate lag and diff features
            features = pd.concat([features, lag_features, diff_features], axis=1)

        # Add 'player_name' and 'game_date' to features DataFrame
        features["player_name"] = name
        features["game_date"] = group["game_date"].values

        # Append the features DataFrame to the list
        new_features_list.append(features)

    # Concatenate all the features into a single DataFrame
    new_features_df = pd.concat(new_features_list, ignore_index=True)

    # Merge the new features DataFrame with the original DataFrame
    df = pd.merge(df, new_features_df, on=["player_name", "game_date"], how="left")

    return df


def add_last_season_data_with_extras(current_df, prev_df):
    """
    Adds last season aggregates and additional stats to current_df using prev_df (the previous season).
    """
    # Instead of a set, define bracket_quantiles as a LIST, so we have a fixed order:
    bracket_quantiles = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50]

    # Then compute the numeric thresholds from prev_df's salary distribution:
    bracket_thresholds = {}
    for q in bracket_quantiles:
        bracket_thresholds[q] = prev_df["salary-fanduel"].quantile(q)

    # Initialize the column in current_df
    current_df["salary_quantile"] = 0.0

    # Sort bracket_thresholds by the numeric cutoff ascending (lowest threshold first).
    # That way, the largest threshold is last, so it overwrites if the player's salary meets that bracket.
    for q, cutoff in sorted(bracket_thresholds.items(), key=lambda x: x[1]):
        # If the player's current salary >= this cutoff, set the bracket to q
        # This will overwrite smaller quantile assignments with bigger quantile ones,
        # ensuring the final bracket is the highest that the player meets.
        current_df.loc[current_df["salary-fanduel"] >= cutoff, "salary_quantile"] = q

    all_cats = dfs_cats + ["fp_fanduel"]  # , 'fp_yahoo', 'fp_draftkings']

    # Predefine new columns in current_df (important for consistent structure)
    for cat in all_cats:
        current_df[f"last_season_avg_{cat}"] = 0  # Initialize with 0
    current_df["last_season_games_played"] = 0
    current_df["last_season_double_doubles"] = 0
    current_df["last_season_triple_doubles"] = 0
    for col in thresholds_for_exceptional_games.keys():
        current_df[f"last_season_{col}_exceptional_games"] = 0

    # --- Check if prev_df is empty ---
    if prev_df.empty:
        print(
            "Warning: Previous season DataFrame is empty.  Returning current_df without last season stats."
        )
        return current_df  # Return early, already initialized

    # --- Handle players that do not exist in prev season
    prev_players = prev_df["player_name"].unique()
    current_players = current_df["player_name"].unique()
    new_players = set(current_players) - set(prev_players)
    missing_players_df = pd.DataFrame()  # Initialize empty DataFrame

    if new_players:
        # Create a DataFrame for new players with last_season columns filled with 0
        new_players_data = []
        for player in new_players:
            player_data = {"player_name": player}
            for col in all_cats:
                player_data[f"last_season_avg_{col}"] = 0
            player_data["last_season_games_played"] = 0
            player_data["last_season_double_doubles"] = 0
            player_data["last_season_triple_doubles"] = 0
            for col in thresholds_for_exceptional_games.keys():
                player_data[f"last_season_{col}_exceptional_games"] = 0
            new_players_data.append(player_data)

        missing_players_df = pd.DataFrame(new_players_data)

    # Calculate aggregate stats for the previous season
    agg_cols = {f"last_season_avg_{cat}": (cat, "mean") for cat in all_cats}
    agg_cols["last_season_games_played"] = ("game_id", "count")
    agg_data = prev_df.groupby("player_name").agg(**agg_cols).reset_index()

    # Calculate exceptional games and doubles
    doubles_data = []
    for player_name, group in prev_df.groupby("player_name"):
        result = calculate_exceptional_games_and_doubles(
            group, thresholds=thresholds_for_exceptional_games
        )
        doubles_data.append({"player_name": player_name, **result})
    doubles_df = pd.DataFrame(doubles_data)

    # --- Handle empty doubles_df CORRECTLY ---
    if doubles_df.empty:
        # If there are NO double-doubles/triple-doubles, create those columns in agg_data and set to 0
        agg_data["last_season_double_doubles"] = 0
        agg_data["last_season_triple_doubles"] = 0
        for stat in thresholds_for_exceptional_games:
            agg_data[f"last_season_{stat}_exceptional_games"] = 0
    else:
        # --- Rename and Merge (ONLY if doubles_df is NOT empty) ---
        rename_cols = {
            col: f"last_season_{col}"
            for col in doubles_df.columns
            if col != "player_name"
        }
        doubles_df = doubles_df.rename(columns=rename_cols)
        agg_data = agg_data.merge(doubles_df, on="player_name", how="left")

    # --- Combine with missing players (if any) ---
    if not missing_players_df.empty:
        agg_data = pd.concat([agg_data, missing_players_df], ignore_index=True).fillna(
            0
        )

    # --- Merge with current_df ---
    # Left merge is CRUCIAL here. We want to keep ALL rows from current_df,
    # and add data from agg_data where there's a match on 'player_name'.
    for col in agg_data.columns:
        if col != "player_name":
            stat_mapping = agg_data.set_index("player_name")[col]
            current_df[col] = current_df["player_name"].map(stat_mapping).fillna(0)

    # current_df = add_last_season_zscores(current_df)
    return current_df


def add_last_season_zscores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deals with the issue of identity leakage that flattens the learning problem.
    When last‑season stats are repeated verbatim across all rows, the trees discover trivial rules
    That adds illusory accuracy yet zero actionable edge (DFS salaries already price LeBron).
    A z‑score by position strips the identity but keeps the skill prior.
    """
    stat_cols = [c for c in df.columns if c.startswith("last_season_avg_")]

    for pos in df["pos-fanduel"].unique():
        mask = df["pos-fanduel"] == pos
        sub = df.loc[mask, stat_cols]
        mu, sigma = sub.mean(), sub.std(ddof=0).replace(0, 1)
        df.loc[mask, [f"{c}_z" for c in stat_cols]] = (sub - mu) / sigma
    df = df.drop(stat_cols, axis=1)
    return df


def add_running_season_stats(df):
    """Add running season aggregates and stats up to but not including the current game"""
    all_cats = dfs_cats + ["fp_fanduel", "fp_yahoo", "fp_draftkings"]

    # Predefine all new columns with zeros (more efficient than None)
    for cat in all_cats:
        df[f"running_season_avg_{cat}"] = 0.0
        df[f"running_season_total_{cat}"] = 0.0
    df["running_season_games_played"] = 0
    df["running_season_double_doubles"] = 0
    df["running_season_triple_doubles"] = 0

    for col in thresholds_for_exceptional_games.keys():
        df[f"running_season_{col}_exceptional_games"] = 0

    # Sort the entire dataframe once instead of multiple times
    df = df.sort_values(
        ["player_name", "team_abbreviation", "season_year", "game_date"]
    )

    # Process each group more efficiently
    for (player_name, team_abbreviation, season_year), group in df.groupby(
        ["player_name", "team_abbreviation", "season_year"], observed=True
    ):

        group_idx = group.index

        # Vectorized operations for all stats at once
        for cat in all_cats:
            # Calculate running averages and totals
            cumsum = group[cat].cumsum()
            cumcount = pd.Series(range(1, len(group) + 1), index=group.index)

            # Shift to exclude current game
            df.loc[group_idx, f"running_season_total_{cat}"] = cumsum.shift(1).fillna(0)
            df.loc[group_idx, f"running_season_avg_{cat}"] = (
                cumsum.shift(1) / cumcount.shift(1)
            ).fillna(0)

        # Calculate games played (vectorized)
        df.loc[group_idx, "running_season_games_played"] = (
            pd.Series(range(len(group)), index=group_idx).shift(1).fillna(0)
        )

        # Calculate double-doubles and triple-doubles more efficiently
        stats_matrix = group[["pts", "reb", "ast", "stl", "blk"]] >= 10
        double_doubles = (stats_matrix.sum(axis=1) >= 2).cumsum().shift(1)
        triple_doubles = (stats_matrix.sum(axis=1) >= 3).cumsum().shift(1)

        df.loc[group_idx, "running_season_double_doubles"] = double_doubles.fillna(0)
        df.loc[group_idx, "running_season_triple_doubles"] = triple_doubles.fillna(0)

        # Calculate exceptional games more efficiently
        for stat, threshold in thresholds_for_exceptional_games.items():
            exceptional_games = (group[stat] >= threshold).cumsum().shift(1)
            df.loc[group_idx, f"running_season_{stat}_exceptional_games"] = (
                exceptional_games.fillna(0)
            )

    return df


def add_anticipated_defense_features(df):
    """
    Calculates and adds features related to the opponent's historical defensive performance
    up to the date of each game.

    Args:
        df (pd.DataFrame): The DataFrame containing all preprocessed and enriched game logs,
                           including 'game_date', 'team_abbreviation', 'opponent_abbr',
                           and relevant statistical columns (like 'def_rating', 'pace',
                           and individual player stats like 'pts', 'reb', etc., and 'fp_fanduel').
                           Must have 'opponent_abbr' column created in preprocessing.

    Returns:
        pd.DataFrame: The DataFrame with new opponent defensive feature columns added.
    """
    # Ensure data is sorted correctly for time-series calculations
    # Sorting by team and then date is efficient for calculating team-specific rolling/expanding stats
    # Sort by the team whose stats we will calculate (which is opponent_abbr in the final merge)
    # but we need to calculate stats FOR EACH TEAM first. So sort by team_abbreviation.
    df = df.sort_values(["team_abbreviation", "game_date"]).reset_index(drop=True)

    # --- 1. Calculate Team-Level Stats (to be used as Opponent Stats) ---
    # These are metrics OF each team over time, which we will later merge as opponent stats.
    print("Calculating team-level defensive/pace stats for opponent features...")
    team_stats_for_opponent = df.copy()  # Work on a copy

    # Group by the team whose stats we are calculating
    team_groups = team_stats_for_opponent.groupby("team_abbreviation")

    # Calculate expanding (season-to-date) and rolling (last 10) averages for relevant stats
    # Use min_periods=1 to include early season data
    windows = [10]  # Define rolling windows, e.g., [10, 5]
    # Include relevant team-level stats that could describe their defensive/pace tendencies
    stats_to_roll_team = [
        "def_rating",
        "pace",
        "off_rating",
    ]  # Added off_rating as it influences pace/game flow

    # Define the names of the new columns being created for team stats
    new_team_stat_cols = []
    for stat in stats_to_roll_team:
        new_team_stat_cols.append(f"{stat}_season_avg")
        for window in windows:
            new_team_stat_cols.append(f"{stat}_last{window}_avg")
            # Consider adding std for team stats as well
            new_team_stat_cols.append(f"{stat}_last{window}_std")

    for stat in stats_to_roll_team:
        if stat in team_stats_for_opponent.columns:  # Ensure stat column exists
            # Calculate season average and rolling average, shifted
            team_stats_for_opponent[f"{stat}_season_avg"] = team_groups[stat].transform(
                lambda x: x.expanding().mean().shift(1)
            )
            for window in windows:
                # Shift(1) is CRITICAL here to exclude the current game's data from the calculation
                team_stats_for_opponent[f"{stat}_last{window}_avg"] = team_groups[
                    stat
                ].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
                )
                team_stats_for_opponent[f"{stat}_last{window}_std"] = team_groups[
                    stat
                ].transform(
                    lambda x: x.rolling(window=window, min_periods=1).std().shift(1)
                )
        else:
            print(
                f"Warning: Team stat column '{stat}' not found for rolling calculation."
            )

    # Select ONLY the newly calculated team stats and the columns needed for merging
    # Use the list of specifically created new column names
    team_stats_agg_for_merge = (
        team_stats_for_opponent[["game_date", "team_abbreviation"] + new_team_stat_cols]
        .drop_duplicates(subset=["game_date", "team_abbreviation"])
        .copy()
    )  # Use copy to avoid SettingWithCopyWarning

    # Now, merge these team stats back to the main df, matching team_abbreviation in team_stats_agg_for_merge
    # with the opponent_abbr in the main df. Rename columns to reflect they are opponent stats.
    rename_opponent_cols = {col: f"opp_{col}" for col in new_team_stat_cols}
    team_stats_agg_for_merge.rename(columns=rename_opponent_cols, inplace=True)

    df = pd.merge(
        df,
        team_stats_agg_for_merge.rename(
            columns={"team_abbreviation": "opponent_abbr"}
        ),  # Rename team_abbreviation to match opponent_abbr for merge
        on=["game_date", "opponent_abbr"],  # Use the correct merge key here
        how="left",
    )
    print("Finished calculating opponent team-level defensive features.")

    # --- 2. Calculate Opponent Defense vs. Position (DvP) Stats (Efficient) ---
    # These are metrics on how many points/rebounds/etc. an opponent allows BY POSITION FACED.
    # This is calculated by looking at the stats of players on the *opposing* team in past games
    # and aggregating them by the defending team and the position of the player they faced.
    print("Calculating opponent Defense vs. Position (DvP) features...")

    # Ensure opponent_abbr is available and pos-fanduel is cleaned
    # df should already have 'opponent_abbr' and 'pos-fanduel' by this stage
    df["pos-fanduel"] = df["pos-fanduel"].fillna(
        "Unknown"
    )  # Ensure no NaNs in position

    # Aggregate raw allowed stats per game, by defending team and position faced.
    # Group the main dataframe by the team that was defending (opponent_abbr),
    # the position they were defending (pos-fanduel of the players on the other team),
    # and the game date.
    # Sum the stats of the players on the *offensive* team for each group.

    # Select the columns needed for aggregation to improve performance
    dvp_agg_cols = (
        ["game_date", "opponent_abbr", "pos-fanduel"]
        + dfs_cats
        + ["fp_fanduel", "def_rating", "off_rating", "pace"]
    )  # Include relevant stats to sum
    # Ensure all columns exist before selecting
    dvp_agg_cols = [col for col in dvp_agg_cols if col in df.columns]

    dvp_game_totals = (
        df.groupby(["game_date", "opponent_abbr", "pos-fanduel"])[
            [
                col
                for col in dvp_agg_cols
                if col not in ["game_date", "opponent_abbr", "pos-fanduel"]
            ]  # Select only stat columns for sum
        ]
        .sum()
        .reset_index()
    )

    # Rename columns to indicate they are stats *allowed* in that game
    # Only rename the stat columns that were summed
    stats_summed = [
        col
        for col in dvp_agg_cols
        if col not in ["game_date", "opponent_abbr", "pos-fanduel"]
    ]
    rename_allowed_cols = {cat: f"{cat}_allowed_in_game" for cat in stats_summed}
    dvp_game_totals.rename(columns=rename_allowed_cols, inplace=True)

    # Now, calculate rolling/expanding averages on these game totals for DvP.
    # Group by the defending team ('opponent_abbr') and the position faced ('pos-fanduel').
    dvp_game_totals = dvp_game_totals.sort_values(
        ["opponent_abbr", "pos-fanduel", "game_date"]
    ).reset_index(drop=True)
    dvp_groups_agg = dvp_game_totals.groupby(["opponent_abbr", "pos-fanduel"])

    windows_dvp = [
        10
    ]  # Rolling window for DvP (games faced at this position vs this opponent)
    # Use the renamed columns as the base for rolling/expanding
    stats_to_roll_dvp = [
        col for col in dvp_game_totals.columns if "_allowed_in_game" in col
    ]

    # Define the names of the new DvP columns being created
    new_dvp_cols = []
    for stat_allowed in stats_to_roll_dvp:
        base_stat_name = stat_allowed.replace("_allowed_in_game", "")
        new_dvp_cols.append(f"opp_dvp_{base_stat_name}_season_avg")
        for window in windows_dvp:
            new_dvp_cols.append(f"opp_dvp_{base_stat_name}_last{window}_avg")
            # new_dvp_cols.append(f'opp_dvp_{base_stat_name}_last{window}_std') # Include std

    for stat_allowed in stats_to_roll_dvp:
        # Calculate expanding (season-to-date) average, shifted
        base_stat_name = stat_allowed.replace("_allowed_in_game", "")
        dvp_game_totals[f"opp_dvp_{base_stat_name}_season_avg"] = dvp_groups_agg[
            stat_allowed
        ].transform(lambda x: x.expanding().mean().shift(1))

        # Calculate rolling averages and stds, shifted
        for window in windows_dvp:
            dvp_game_totals[
                f"opp_dvp_{base_stat_name}_last{window}_avg"
            ] = dvp_groups_agg[stat_allowed].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            # dvp_game_totals[f'opp_dvp_{base_stat_name}_last{window}_std'] = dvp_groups_agg[stat_allowed].transform(lambda x: x.rolling(window=window, min_periods=1).std().shift(1))

    # Select the calculated DvP features and merge keys
    # Use the list of specifically created new DvP column names
    dvp_merge_cols = ["game_date", "opponent_abbr", "pos-fanduel"] + new_dvp_cols
    # Ensure selected columns exist in dvp_game_totals before selecting
    dvp_merge_cols = [col for col in dvp_merge_cols if col in dvp_game_totals.columns]
    dvp_for_merge = dvp_game_totals[dvp_merge_cols].copy()  # Use copy

    # Merge DvP stats back to the main df using the correct keys
    # This joins the calculated DvP for opponent O against position P on date D
    # to the main rows where a player with position P faced opponent O on date D.
    df = pd.merge(
        df, dvp_for_merge, on=["game_date", "opponent_abbr", "pos-fanduel"], how="left"
    )

    print("Finished calculating opponent Defense vs. Position (DvP) features.")

    # --- 3. Handle Missing Values for New Features ---
    # Players/teams/DvP groups might not have enough past data for rolling/expanding calculations.
    # Decide on a strategy: fill with 0, fill with league average, or use a specific indicator value.
    # Filling with 0 is a simple starting point but might not be ideal. League average is better.
    # Identify all columns that were newly created in this function
    all_new_feature_cols = list(team_stats_agg_for_merge.columns) + new_dvp_cols

    for col in all_new_feature_cols:
        if (
            col in df.columns
        ):  # Check if the column was successfully added during merges
            # Consider filling with league average up to that date, or a specific indicator
            # For now, sticking with fillna(0) as per previous code, but note this is basic.
            df[col] = df[col].fillna(0)
        else:
            print(
                f"Warning: New feature column '{col}' not found in final DataFrame after merges."
            )

    return df
