import pandas as pd

from src.data_enrichment import (
    add_interaction_features,
    add_time_dependent_features_v2,
    add_running_season_stats,
    add_last_season_data_with_extras,
    add_anticipated_defense_features,
)
from src.preprocessing import merge_all_seasons, preprocess_all_seasons_data


def preprocess_pipeline():
    all_seasons_df = merge_all_seasons()
    return preprocess_all_seasons_data(all_seasons_df)


def enrich_pipeline(df):
    """
    Adds rolling/lags/diffs features (via add_time_dependent_features_v2)
    and merges previous-season data (via add_last_season_data_with_extras).
    Also adds running season stats (via add_running_season_stats).
    """
    # Rolling window stats
    df = add_time_dependent_features_v2(df, rolling_window=10)

    df = add_interaction_features(df)
    # Running season stats (all within the same season)
    df = add_running_season_stats(df)

    # Enrich across seasons
    enriched_seasons = []
    all_seasons = [s for s in df["season_year"].unique()]  # Store full season strings

    for current_season_year in df["season_year"].unique():
        print(f"Processing season year: {current_season_year}")
        season_df = df[df["season_year"] == current_season_year].copy()

        season_start_year = int(current_season_year.split("-")[0])  # Get as integer
        prev_season_start_year = str(season_start_year - 1)
        prev_season_year = f"{prev_season_start_year}-{str(season_start_year)[-2:]}"  # Full prev season

        if prev_season_year in all_seasons:
            print(f"Adding stats from previous season: {prev_season_year}")
            prev_season_df = df[df["season_year"] == prev_season_year]
            season_df = add_last_season_data_with_extras(season_df, prev_season_df)
        else:
            print(f"No data for previous season: {prev_season_year}")

        enriched_seasons.append(season_df)

    # Concatenate all enriched season DataFrames
    enriched_df = pd.concat(enriched_seasons, ignore_index=True)

    enriched_df = add_anticipated_defense_features(enriched_df)

    enriched_df = enriched_df.sort_values(["game_date"]).reset_index(drop=True)
    print("All seasons enriched successfully!")

    return enriched_df
