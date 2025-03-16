#!/usr/bin/env python3
"""
evaluate_lineup_results.py

Usage:
1) Adjust the file paths for:
   - final_lineup CSV (predicted lineups + actual performance).
   - contests CSV (actual DFS contests).
   - player_predictions CSV (predicted vs. actual stats for each player).

2) Run:
   python evaluate_lineup_results.py

3) Script will produce:
   - console outputs (win/cash rates, profits, etc.)
   - some plots (saved as PNG or shown interactively)
   - potential merged DataFrames for further usage
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

###############################################################################
# 1) Evaluate: Lineup vs. Actual Contest Data
###############################################################################
def evaluate_lineups_vs_contests(
    lineup_csv_path : str=None,
    output_csv_path: str = None
) -> pd.DataFrame:
    """
    Merges a final_lineup CSV (with predicted & actual lineup scores)
    against real FanDuel contests. Computes metrics like:
      - Win/cash rates
      - Profit
      - Cumulative profit over time
      - Distribution of difference to winning/cash lines

    Returns:
        merged_df (pd.DataFrame) with columns:
          - game_date
          - winning_score, mincash_score
          - fanduel_predicted_lineup_GT_points, fanduel_GT_points
          - ...
    """
    print("\n=== Evaluating Lineup vs. Contest Data ===")

    # 1) Load Data
    contests_df = pd.read_csv("../data/contests_data/dfs_contests_fanduel_2022-23.csv")
    lineup_pred_df = pd.read_csv(lineup_csv_path)

    # 2) Filter & Merge
    #   We'll rename & unify date columns so we can merge on "game_date"
    contests_df.rename(columns={'period': 'game_date'}, inplace=True)
    lineup_pred_df.rename(columns={'date': 'game_date'}, inplace=True)

    contests_df['game_date'] = pd.to_datetime(contests_df['game_date'])
    lineup_pred_df['game_date'] = pd.to_datetime(lineup_pred_df['game_date'])

    # Basic filtering: Keep only certain contest types
    # Adjust or remove if not needed
    valid_titles = ['Main', 'After Hours', 'Express']
    contests_df = contests_df[contests_df['Title'].isin(valid_titles)]
    contests_df = contests_df[contests_df['total_entrants'] > 50]
    contests_df = contests_df[contests_df['cost'] >= 1]
    contests_df = contests_df[contests_df['game_date'].isin(lineup_pred_df['game_date'])]

    # Merge
    merged_df = pd.merge(contests_df, lineup_pred_df, on='game_date', how='left')

    # 3) Compare lineup performance
    #   Example columns in contests_df: 'winning_score', 'mincash_score'
    #   Example columns in pred_df: 'fanduel_predicted_lineup_GT_points', 'fanduel_GT_points'
    merged_df['winning_score_vs_pred'] = merged_df['winning_score'] - merged_df['fanduel_predicted_lineup_GT_points']
    merged_df['winning_score_vs_gt']   = merged_df['winning_score'] - merged_df['fanduel_GT_points']

    merged_df['cash_line_vs_pred']     = merged_df['mincash_score'] - merged_df['fanduel_predicted_lineup_GT_points']
    merged_df['cash_line_vs_gt']       = merged_df['mincash_score'] - merged_df['fanduel_GT_points']

    # Win/cash booleans
    merged_df['pred_lineup_would_win']  = merged_df['fanduel_predicted_lineup_GT_points'] >= merged_df['winning_score']
    merged_df['actual_lineup_would_win'] = merged_df['fanduel_GT_points'] >= merged_df['winning_score']

    merged_df['pred_lineup_would_cash']  = merged_df['fanduel_predicted_lineup_GT_points'] >= merged_df['mincash_score']
    merged_df['actual_lineup_would_cash'] = merged_df['fanduel_GT_points'] >= merged_df['mincash_score']

    # Profit calculation
    def compute_profit(row):
        # Extremely simplified logic
        if row['pred_lineup_would_win']:
            return row['prizepool'] - row['cost']
        elif row['pred_lineup_would_cash']:
            return row['mincash_payout'] - row['cost']
        else:
            return -row['cost']

    merged_df['pred_lineup_profit'] = merged_df.apply(compute_profit, axis=1)

    # Basic stats
    num_contests = len(merged_df)
    pred_win_rate  = merged_df['pred_lineup_would_win'].mean()
    pred_cash_rate = merged_df['pred_lineup_would_cash'].mean()
    print(f"Number of contests: {num_contests}")
    print(f"Predicted Win Rate:  {pred_win_rate:.2%}")
    print(f"Predicted Cash Rate: {pred_cash_rate:.2%}")

    total_profit = merged_df['pred_lineup_profit'].sum()
    avg_profit   = merged_df['pred_lineup_profit'].mean()
    print(f"Total Profit: ${total_profit:.2f}, Avg Profit per Contest: ${avg_profit:.2f}")

    # Daily or monthly charts
    daily_profit = (merged_df
                    .groupby('game_date')['pred_lineup_profit']
                    .sum()
                    .reset_index(name='daily_profit'))
    if not daily_profit.empty:
        daily_profit['cumulative_profit'] = daily_profit['daily_profit'].cumsum()
        # Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=daily_profit, x='game_date', y='cumulative_profit', marker='o')
        plt.title('Cumulative Profit Over Time')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Profit ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Optional distribution of (score_diff_to_win, score_diff_to_cash)
    merged_df['score_diff_to_win']  = merged_df['fanduel_predicted_lineup_GT_points'] - merged_df['winning_score']
    merged_df['score_diff_to_cash'] = merged_df['fanduel_predicted_lineup_GT_points'] - merged_df['mincash_score']

    # Save or return
    if output_csv_path:
        merged_df.to_csv(output_csv_path, index=False)
        print(f"Saved merged contests data to {output_csv_path}")

    return merged_df


###############################################################################
# 2) Evaluate: Player Predictions vs. Actual Stats
###############################################################################
def evaluate_player_predictions(
    player_pred_csv_path: str,
    output_csv_path: str = None
) -> pd.DataFrame:
    """
    Loads a CSV with columns like:
      player_name, game_date, fp_fanduel, fp_fanduel_pred, etc.
    Calculates:
      - error columns
      - distribution plots, boxplots, group by team, etc.

    Returns a DataFrame with new columns for error metrics.
    """
    print("\n=== Evaluating Player Predictions ===")

    df = pd.read_csv(player_pred_csv_path)
    df['game_date'] = pd.to_datetime(df['game_date'])

    # Example: Freed columns "fp_fanduel", "fp_fanduel_pred"
    df['fp_fanduel_error'] = (df['fp_fanduel'] - df['fp_fanduel_pred']).abs()
    # If you want scaled or % error:
    df['fp_fanduel_pct_error'] = (df['fp_fanduel_error'] / df['fp_fanduel'].replace(0,1))

    # Print some quick stats
    mse_fanduel = mean_squared_error(df['fp_fanduel'], df['fp_fanduel_pred'], squared=False)
    print(f"RMSE (FanDuel) overall: {mse_fanduel:.4f}")

    # Possibly do team or date-based grouping
    if 'team' in df.columns:
        team_errors = df.groupby('team')['fp_fanduel_error'].mean().sort_values()
        print("\nMean Error by Team (FanDuel):")
        print(team_errors.head(10))

    # Quick distribution plot
    plt.figure(figsize=(10,6))
    sns.histplot(df['fp_fanduel_error'], kde=True)
    plt.title('Distribution of FanDuel FP Errors')
    plt.xlabel('Absolute Error')
    plt.tight_layout()
    plt.show()

    # Save results
    if output_csv_path:
        df.to_csv(output_csv_path, index=False)
        print(f"Saved player predictions with error columns to {output_csv_path}")

    return df


###############################################################################
# 3) Main aggregator or usage example
###############################################################################
def main():
    """
    Example usage: Evaluate lineups vs. contests + Evaluate player predictions
    """
    # 1) Evaluate lineups vs. actual contests
    # Adjust paths:
    lineup_csv_path   = "../output_csv/final_lineup_2025-03-14.csv"
    contests_csv_path = "../data/contests_data/fanduel_nba_contests.csv"
    lineup_merged_out = "../output_csv/merged_contests_pred_data_25-03-14.csv"

    if os.path.exists(lineup_csv_path) and os.path.exists(contests_csv_path):
        contests_merged_df = evaluate_lineups_vs_contests(
            lineup_csv_path=lineup_csv_path,
            output_csv_path=lineup_merged_out
        )
    else:
        print(f"Skipping lineup vs. contests. Paths not found:\n  {lineup_csv_path}\n  {contests_csv_path}")

    # 2) Evaluate player-level predictions
    # For example: "fp_xgb_daily_pred_three_months_only.csv"
    player_pred_csv = "../output_csv/fp_xgb_daily_pred_three_months_only.csv"
    if os.path.exists(player_pred_csv):
        evaluate_player_predictions(
            player_pred_csv_path=player_pred_csv,
            output_csv_path="../output_csv/fp_xgb_daily_with_errors.csv"
        )
    else:
        print(f"Skipping player prediction eval. File not found: {player_pred_csv}")


if __name__ == "__main__":
    main()



