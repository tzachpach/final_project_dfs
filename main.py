import pandas as pd

from config.constants import best_params
from src.data_enrichment import add_last_season_data_with_extras, add_time_dependent_features_v2, \
    add_running_season_stats
from src.evaluate_results import evaluate_results
from src.lineup_genetic_optimizer import get_lineup
from src.predict_fp_rnn_weekly import *
from src.predict_fp_xgb_daily import predict_fp_xgb
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

    # Running season stats (all within same season)
    df = add_running_season_stats(df)

    # Enrich across seasons
    enriched_seasons = []
    all_seasons = [s for s in df['season_year'].unique()] # Store full season strings

    for current_season_year in df['season_year'].unique():
        print(f"Processing season year: {current_season_year}")
        season_df = df[df['season_year'] == current_season_year].copy()

        season_start_year = int(current_season_year.split("-")[0])  # Get as integer
        prev_season_start_year = str(season_start_year - 1)
        prev_season_year = f"{prev_season_start_year}-{str(season_start_year)[-2:]}" # Full prev season

        if prev_season_year in all_seasons: # Correct comparison
            print(f"Adding stats from previous season: {prev_season_year}")
            prev_season_df = df[df['season_year'] == prev_season_year]
            season_df = add_last_season_data_with_extras(season_df, prev_season_df)
        else:
            print(f"No data for previous season: {prev_season_year}")

        enriched_seasons.append(season_df)

    # Concatenate all enriched season DataFrames
    enriched_df = pd.concat(enriched_seasons, ignore_index=True)
    enriched_df = enriched_df.sort_values(['game_date']).reset_index(drop=True)
    print("All seasons enriched successfully!")

    return enriched_df



def main():
    # Step 1: Preprocess data
    print("Starting pipeline...")
    preprocessed_df = preprocess_pipeline()

    preprocessed_df = preprocessed_df[preprocessed_df['season_year'].isin(['2016-17', '2017-18'])]
    preprocessed_df = preprocessed_df.sort_values(['game_date']).reset_index(drop=True)
    print("Preprocessing completed successfully!")
    # Step 2: Enrich data
    enriched_df = enrich_pipeline(preprocessed_df)
    print("Enrichment completed successfully!")
    # Step 3: Train models and predict fantasy points

    predictions_df = predict_fp_xgb(enriched_df, mode="daily", train_window_days=10)
    # predictions_df = run_rnn_and_merge_results(
    #     df=enriched_df,
    #     platform="fanduel",
    #     group_by="week",
    #     multi_target_mode=True,
    #     **best_params
    # )
    print("Daily fantasy point predictions completed successfully!")
    # Step 4: Optimize lineups
    lineup_df = get_lineup(predictions_df)
    today = pd.Timestamp.now().strftime("%Y-%m-%d")
    lineup_df.to_csv(f"output/lineup_{today}.csv", index=False)
    # Save the results
    print("Lineup calculations completed successfully!")
    df_overall_results, df_percentile_data = evaluate_results(predictions_df, lineup_df)
    print("Evaluation results completed successfully!")

if __name__ == "__main__":
    main()
