import pandas as pd

from config.dfs_categories import same_game_cols, dfs_cats
from config.fantasy_point_calculation import calculate_fp_fanduel, calculate_fp_yahoo, calculate_fp_draftkings
from src.test_train_utils import rolling_train_test_for_xgb


def predict_fp(
    enriched_df,
    season_year=None,
    start_date=None,
    end_date=None,
    mode="daily",
    train_window_days=20,
    train_window_weeks=4,
    percentile_to_filter_over=0.6,
    save_model=True
):
    """
    Predict fantasy points using rolling training and testing for both daily and weekly training.
    Processes all available data at once, rather than per season.

    Args:
        enriched_df (pd.DataFrame): Enriched DataFrame containing player and game data with necessary features.
        season_year (str): Specific season year to filter (e.g., "2022-2023"). If None, includes all seasons.
        start_date (str): Start date to filter data (e.g., "2023-01-01"). If None, includes all start dates.
        end_date (str): End date to filter data (e.g., "2023-04-01"). If None, includes all end dates.
        mode (str): "daily" for daily rolling prediction, "weekly" for weekly rolling prediction.
        train_window_days (int): Number of previous days to use for daily training.
        train_window_weeks (int): Number of previous weeks to use for weekly training.
        save_model (bool): Whether to save trained models.

    Returns:
        pd.DataFrame: DataFrame containing predictions and results for the specified filters.
    """
    # Validate mode
    if mode not in ["daily", "weekly"]:
        raise ValueError("Invalid mode. Use 'daily' or 'weekly'.")

    # Determine grouping method and training window
    group_by = "date" if mode == "daily" else "week"
    train_window = train_window_days if mode == "daily" else train_window_weeks

    # Apply filters
    if season_year:
        enriched_df = enriched_df[enriched_df['season_year'] == season_year]
    if start_date:
        enriched_df = enriched_df[enriched_df['game_date'] >= pd.to_datetime(start_date)]
    if end_date:
        enriched_df = enriched_df[enriched_df['game_date'] <= pd.to_datetime(end_date)]
    if percentile_to_filter_over:
        enriched_df = enriched_df[enriched_df['salary-fanduel'] >= enriched_df['salary-fanduel'].quantile(percentile_to_filter_over)]

    if enriched_df.empty:
        raise ValueError("No data available after applying filters. Please adjust the filters.")

    # Prepare features and target variables
    features = enriched_df.columns.difference(same_game_cols).tolist()
    combined_df = pd.DataFrame()

    for cat in dfs_cats:
        target = cat
        X = enriched_df[features]
        y = enriched_df[target]

        print(f"Training models for {cat} (Mode: {mode})")
        cat_results = rolling_train_test_for_xgb(
            X, y, enriched_df, group_by=group_by, train_window=train_window, save_model=save_model
        )

        cat_results.rename(columns={'y': cat, 'y_pred': f'{cat}_pred'}, inplace=True)

        if combined_df.empty:
            combined_df = cat_results
        else:
            combined_df = pd.merge(
                combined_df,
                cat_results,
                on=['player_name', 'game_date', 'game_id', 'fanduel_salary', 'draftkings_salary', 'yahoo_salary',
                    'fanduel_position', 'draftkings_position', 'yahoo_position'],
                how='outer',
                suffixes=('', f'_{cat}'))

    # Compute fantasy points
    combined_df['fp_fanduel_pred'] = combined_df.apply(lambda row: calculate_fp_fanduel(row, pred_mode=True), axis=1)
    combined_df['fp_yahoo_pred'] = combined_df.apply(lambda row: calculate_fp_yahoo(row, pred_mode=True), axis=1)
    combined_df['fp_draftkings_pred'] = combined_df.apply(lambda row: calculate_fp_draftkings(row, pred_mode=True), axis=1)

    # Save results
    res_name = f'fp_xgb_{mode}_pred_{train_window_days}_days'
    if season_year:
        res_name += f'_{season_year}'
    if start_date:
        res_name += f'_from_{start_date}'
    if end_date:
        res_name += f'_to_{end_date}'
    if percentile_to_filter_over:
        res_name += f'_salary_over_{percentile_to_filter_over:.2f}_percentile'
    res_name += '.csv'

    combined_df['fp_fanduel'] = combined_df.apply(calculate_fp_fanduel, axis=1)
    combined_df['fp_yahoo'] = combined_df.apply(calculate_fp_yahoo, axis=1)
    combined_df['fp_draftkings'] = combined_df.apply(calculate_fp_draftkings, axis=1)

    combined_df.to_csv(f'output_csv/{res_name}', index=False)
    return combined_df
