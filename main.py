import pandas as pd

from src.data_enrichment import add_last_season_data_with_extras, add_time_dependent_features_v2, \
    add_running_season_stats
from src.lineup_genetic_optimizer import get_lineup
from src.predict_fp_xgb_daily import predict_fp
from src.preprocessing import merge_all_seasons, preprocess_all_seasons_data


def preprocess_pipeline():
    all_seasons_df = merge_all_seasons()
    return preprocess_all_seasons_data(all_seasons_df)

def enrich_pipeline(df):

    df = add_time_dependent_features_v2(df, rolling_window=10)
    df = add_running_season_stats(df)
    enriched_seasons = []  # List to hold enriched DataFrames for each season

    all_seasons_start_years = [s.split("-")[0] for s in df['season_year'].unique()]
    for current_season_year in df['season_year'].unique():
        print(f"Processing season year: {current_season_year}")

        # Filter the current season's data
        season_df = df[df['season_year'] == current_season_year].copy()

        # Get the previous season's start year
        season_start_year = current_season_year.split("-")[0]
        prev_season_start_year = str(int(season_start_year) - 1)

        if prev_season_start_year in all_seasons_start_years:
            print(f"Adding stats from previous season: {prev_season_start_year}-{season_start_year}")
            prev_season_df = df[df['season_year'] == f'{prev_season_start_year}-{season_start_year}']
            season_df = add_last_season_data_with_extras(season_df, prev_season_df)
        else:
            print(f"No data for previous season: {prev_season_start_year}-{season_start_year}")

        # Append the enriched DataFrame for this season to the list
        enriched_seasons.append(season_df)

    # Concatenate all enriched season DataFrames into a single DataFrame
    enriched_df = pd.concat(enriched_seasons, ignore_index=True)
    enriched_df = enriched_df.sort_values(['game_date']).reset_index(drop=True)
    print("All seasons enriched successfully!")
    return enriched_df



def main():
    # Step 1: Preprocess data
    preprocessed_df = preprocess_pipeline()

    preprocessed_df = preprocessed_df[preprocessed_df['season_year'].isin(['2019-20', '2020-21'])]
    preprocessed_df = preprocessed_df.sort_values(['game_date']).reset_index(drop=True)

    # Step 2: Enrich data
    enriched_df = enrich_pipeline(preprocessed_df)

    # Step 3: Train models and predict fantasy points
    daily_fp_predictions = predict_fp(enriched_df, mode="daily", train_window_days=10)

    # Step 4: Optimize lineups
    lineup_df = get_lineup(daily_fp_predictions)

    # Save the results
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    lineup_df.to_csv(f'output_csv/final_lineup_{today}.csv', index=False)
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
