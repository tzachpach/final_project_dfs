import pandas as pd
from config.dfs_categories import same_game_cols, dfs_cats
from config.fantasy_point_calculation import calculate_fp_fanduel, calculate_fp_yahoo, calculate_fp_draftkings
from src.test_train_utils import rolling_train_test_for_xgb

def predict_fp_xgb(
    enriched_df,
    season_year=None,
    start_date=None,
    end_date=None,
    mode="daily",
    # Original numeric defaults
    train_window_days=20,
    train_window_weeks=4,
    percentile_to_filter_over=0.6,
    save_model=True,
    # NEW: optional lists for hyperparam tuning
    train_window_days_list=None,   # e.g. [10, 20]
    xgb_param_list=None,          # e.g. [{"max_depth":3},{"max_depth":5,"learning_rate":0.1}]
    percentile_list=None          # e.g. [0.6, 0.7, 0.8]
):
    """
    Predict fantasy points using rolling training and testing for both daily and weekly training.
    Optionally do multiple runs (tuning) over:
      - train_window_days_list
      - xgb_param_list
      - percentile_list

    If only single values are provided (or no lists), behaves like the classic function
    returning a single result DataFrame. If multiple combos, returns a list of (param_info, df) pairs.

    Args:
        enriched_df (pd.DataFrame): DataFrame with player + game data + features
        season_year (str): e.g., "2022-23"
        start_date (str): e.g. "2023-01-01"
        end_date (str): e.g. "2023-04-01"
        mode (str): "daily" or "weekly"
        train_window_days (int): default daily train window
        train_window_weeks (int): default weekly train window
        percentile_to_filter_over (float): default salary quantile threshold
        save_model (bool): whether to save XGBoost models
        train_window_days_list (list[int]): If you want to try multiple daily windows
        xgb_param_list (list[dict]): If you want to try multiple XGBoost param sets
        percentile_list (list[float]): If you want multiple salary quantile thresholds

    Returns:
        * If single-run scenario (no multi combos): returns pd.DataFrame
        * If multi-run scenario: returns list of (param_dict, results_df)
    """
    # Validate mode
    if mode not in ["daily", "weekly"]:
        raise ValueError("Invalid mode. Use 'daily' or 'weekly'.")

    group_by = "date" if mode == "daily" else "week"
    default_train_window = train_window_days if mode == "daily" else train_window_weeks

    # If not doing multi-run, we default to single scenario
    if train_window_days_list is None:
        train_window_days_list = [train_window_days]
    if xgb_param_list is None:
        xgb_param_list = [{}]  # means no custom param override (just uses default inside rolling_train_test_for_xgb)
    if percentile_list is None:
        percentile_list = [percentile_to_filter_over]

    # Filter the data
    df_filtered = enriched_df.copy()

    if season_year:
        df_filtered = df_filtered[df_filtered['season_year'] == season_year]
    if start_date:
        df_filtered = df_filtered[df_filtered['game_date'] >= pd.to_datetime(start_date)]
    if end_date:
        df_filtered = df_filtered[df_filtered['game_date'] <= pd.to_datetime(end_date)]
    if df_filtered.empty:
        raise ValueError("No data after applying year/date filters. Please adjust filters.")

    # Check if we have multiple combos to run
    do_multi_run = (
        len(train_window_days_list) > 1 or
        len(xgb_param_list) > 1 or
        len(percentile_list) > 1
    )

    # We'll store results if multi-run
    all_runs = []

    def run_single_combo(tw, xgb_params):
        """
        1) Filter df by percentile p (if not None)
        2) Use train_window=tw
        3) For each category in dfs_cats, do rolling_train_test_for_xgb with xgb_params
        4) Merge partial results
        5) Compute fp_fanduel_pred, etc.
        6) Return final DataFrame
        """
        local_df = df_filtered.copy()

        # Prepare features (exclude same_game_cols)
        features = local_df.columns.difference(same_game_cols).tolist()

        combined_df = pd.DataFrame()

        for cat in dfs_cats:
            target = cat
            X = local_df[features]
            y = local_df[target]

            print(f"\n--- Rolling XGB for cat={cat}, train_window={tw}, percentile={p}, xgb_params={xgb_params} ---")
            cat_results = rolling_train_test_for_xgb(
                X, y, local_df,
                n_percent=percentile_to_filter_over,
                group_by=group_by,
                train_window=tw,
                save_model=save_model,
                model_dir="models",
            )
            cat_results.rename(columns={'y': cat, 'y_pred': f'{cat}_pred'}, inplace=True)

            if combined_df.empty:
                combined_df = cat_results
            else:
                combined_df = pd.merge(
                    combined_df,
                    cat_results,
                    on=[
                        'player_name', 'minutes_played', 'game_date', 'game_id',
                        'fanduel_salary','draftkings_salary','yahoo_salary',
                        'fanduel_position','draftkings_position','yahoo_position'
                    ],
                    how='outer',
                    suffixes=('', f'_{cat}')
                )

        # Now compute final fantasy point columns
        combined_df['fp_fanduel_pred'] = combined_df.apply(lambda row: calculate_fp_fanduel(row, pred_mode=True), axis=1)
        combined_df['fp_yahoo_pred'] = combined_df.apply(lambda row: calculate_fp_yahoo(row, pred_mode=True), axis=1)
        combined_df['fp_draftkings_pred'] = combined_df.apply(lambda row: calculate_fp_draftkings(row, pred_mode=True), axis=1)

        combined_df['fp_fanduel'] = combined_df.apply(calculate_fp_fanduel, axis=1)
        combined_df['fp_yahoo']   = combined_df.apply(calculate_fp_yahoo, axis=1)
        combined_df['fp_draftkings'] = combined_df.apply(calculate_fp_draftkings, axis=1)

        return combined_df

    # Loop over each combo
    for tw in train_window_days_list:
        final_tw = tw if mode == "daily" else train_window_weeks
        for p in percentile_list:
            for xgb_params_dict in xgb_param_list:
                result_df = run_single_combo(final_tw, p, xgb_params_dict)
                if result_df is None:
                    continue  # e.g. no data left
                # if not do_multi_run:
                    # Single-run scenario
                    # name + save the result CSV (like the old code) then return
                res_name = f'fp_xgb_{mode}_pred_{tw}_days'
                if season_year:
                    res_name += f'_{season_year}'
                if start_date:
                    res_name += f'_from_{start_date}'
                if end_date:
                    res_name += f'_to_{end_date}'
                if p:
                    res_name += f'_salary_over_{p:.2f}_percentile'
                # if xgb_params_dict has extra fields like "max_depth" we can add them in
                for k,v in xgb_params_dict.items():
                    res_name += f'_{k}{v}'
                res_name += '.csv'

                result_df.to_csv(f'output_csv/{res_name}', index=False)


                # multi-run scenario => store
                run_info = {
                    "train_window_days": tw,
                    "percentile_threshold": p,
                    **xgb_params_dict
                }
                all_runs.append((run_info, result_df))

    # If we did multi-run, return the list of results
    if do_multi_run:
        return all_runs
    else:
        # Single-run scenario, return the last result
        return result_df
