import pandas as pd
import os

from config.dfs_categories import same_game_cols
from config.fantasy_point_calculation import (
    calculate_fp_fanduel,
    calculate_fp_yahoo,
    calculate_fp_draftkings,
)
from config.feature_engineering import clean_numeric_columns


def preprocess_basic(df):
    """
    Perform basic preprocessing on the dataset, including feature engineering and cleaning.
    Applies row-level transformations and drops basic redundant columns.
    """
    df["starter"] = df["starter(y/n)"].apply(lambda x: 1 if x == "Y" else 0)
    df["venue"] = df["venue(r/h)"].apply(lambda x: 1 if x == "H" else 0)
    df["is_playoff"] = df["regular/playoffs"].apply(
        lambda x: 0 if x == "Regular" else 1
    )
    df["is_wl"] = df["wl"].apply(lambda x: 1 if x == "W" else 0)
    df["days_rest_int"] = df["days_rest"].astype(int)

    # Drop unnecessary columns that have been transformed or are redundant
    cols_to_drop = [
        "player_id",
        "team_id",
        "days_rest",
        "starter(y/n)",
        "venue(r/h)",
        "regular/playoffs",
        "wl",
    ]
    df = df.drop(cols_to_drop, axis=1)

    # Fill missing values for minutes played - essential for calculations
    df["minutes_played"] = df["minutes_played"].fillna(0)
    return df


def merge_all_seasons():
    """
    Merges all season files from the specified directory into a single DataFrame.
    Expects files matching 'merged_gamelogs_salaries_*.csv'.
    Performs initial concatenation and basic column cleanup.
    """
    # Get the path to the data directory (one level up from src)
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    dfs_to_merge = [
        pd.read_csv(os.path.join(data_path, f), low_memory=False)
        for f in os.listdir(data_path)
        if "merged_gamelogs_salaries_" in f and f.endswith(".csv")
    ]

    if not dfs_to_merge:
        raise FileNotFoundError(
            f"No 'merged_gamelogs_salaries_*.csv' files found in {data_path}"
        )

    # Combine all data into one DataFrame - handle potential inconsistencies across files
    df = pd.concat(dfs_to_merge, ignore_index=True)

    # Clean up unnecessary index column often generated during saving
    df = df.drop("Unnamed: 0", axis=1, errors="ignore")

    # Handle venue column inconsistencies - merge and drop older/redundant column
    if "venue(r/h/n)" in df.columns:  # Check existence before accessing
        df["venue(r/h)"] = df["venue(r/h)"].fillna(df["venue(r/h/n)"])
        df.drop(columns=["venue(r/h/n)"], inplace=True)
    elif "venue(r/h)" not in df.columns:  # Ensure venue column exists after checks
        raise KeyError("Neither 'venue(r/h)' nor 'venue(r/h/n)' found in dataframes.")

    # Drop rows with missing essential flags like availability
    df = df.dropna(subset=["available_flag"])

    # Apply basic preprocessing for row-level transformations
    df = preprocess_basic(df)
    df = standardize_opponent_names(df)
    return df


def standardize_opponent_names(df):
    """
    Standardize opponent names to 3-letter abbreviations.
    """
    team_name_to_abbr_map = {
        "Atlanta": "ATL",
        "Boston": "BOS",
        "Brooklyn": "BKN",
        "Charlotte": "CHA",
        "Chicago": "CHI",
        "Cleveland": "CLE",
        "Dallas": "DAL",
        "Denver": "DEN",
        "Detroit": "DET",
        "Golden State": "GSW",
        "Houston": "HOU",
        "Indiana": "IND",
        "LA Clippers": "LAC",
        "LA Lakers": "LAL",
        "Memphis": "MEM",
        "Miami": "MIA",
        "Milwaukee": "MIL",
        "Minnesota": "MIN",
        "New Orleans": "NOP",
        "New York": "NYK",
        "Oklahoma City": "OKC",
        "Orlando": "ORL",
        "Philadelphia": "PHI",
        "Phoenix": "PHX",
        "Portland": "POR",
        "Sacramento": "SAC",
        "San Antonio": "SAS",
        "Toronto": "TOR",
        "Utah": "UTA",
        "Washington": "WAS",
    }
    df["opponent_abbr"] = df["opponent"].replace(team_name_to_abbr_map)
    df = df.drop(columns=["opponent"])
    return df


def preprocess_all_seasons_data(all_seasons_df):
    all_seasons_df.drop(
        "Unnamed: 0", axis=1, errors="ignore", inplace=True
    )  # Handle potential missing column
    all_seasons_df["game_date"] = pd.to_datetime(all_seasons_df["game_date"])
    all_seasons_df.sort_values(["game_date"], ascending=True, inplace=True)

    # Clean numeric columns - handle NaNs and data types
    all_seasons_df = clean_numeric_columns(all_seasons_df, same_game_cols)

    # Calculate fantasy points for each platform - derive target variables
    all_seasons_df["fp_fanduel"] = all_seasons_df.apply(calculate_fp_fanduel, axis=1)
    all_seasons_df["fp_yahoo"] = all_seasons_df.apply(calculate_fp_yahoo, axis=1)
    all_seasons_df["fp_draftkings"] = all_seasons_df.apply(calculate_fp_draftkings, axis=1)
    return all_seasons_df
