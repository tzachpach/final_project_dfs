# Fantasy Basketball Lineup Optimizer

This repository provides a **single pipeline** for **NBA daily fantasy** tasks: data preprocessing, feature engineering, model training/prediction, lineup optimization, **and** post-analysis comparing predicted lineups to actual contest results. The **entry point** is **`main.py`**, which orchestrates all steps end to end.

---

## Table of Contents
1. [Overview](#overview)  
2. [Pipeline Flow](#pipeline-flow)  
3. [Key Scripts and Modules](#key-scripts-and-modules)  
4. [Usage](#usage)  
5. [Contest Outcome Analysis](#contest-outcome-analysis)  
6. [Future Enhancements](#future-enhancements)

---

## 1. Overview <a name="overview"></a>
**Goal**: Combine historical NBA player data (including fantasy salaries, game logs, rolling stats, etc.) to:

1. **Predict** each player’s fantasy points (daily or weekly) using **XGBoost** or **RNN**.  
2. **Optimize** a DFS lineup subject to salary and positional constraints via a **genetic algorithm**.  
3. **Evaluate** how these predicted lineups compare to real contest results, including potential profit.

**Features**:
- **Rolling/lag** features (5 or 10 previous games).  
- **Cross-season** enrichment (pulling last season’s stats for continuity).  
- **Per-player** scaling for RNN models (avoids data leakage).  
- **Comparison** of predicted lineup scores against real DFS contest winning scores (profit analysis).

---

## 2. Pipeline Flow <a name="pipeline-flow"></a>
Running **`main.py`** will:

1. **Preprocess** data  
   - Merges all relevant season logs into one DataFrame (via `merge_all_seasons()`).  
   - Cleans & standardizes columns (`preprocess_all_seasons_data()`).

2. **Enrich** data  
   - Adds rolling/lags/diffs features (`add_time_dependent_features_v2`).  
   - Accumulates running season stats (`add_running_season_stats`).  
   - Pulls last season’s data for cross-season continuity (`add_last_season_data_with_extras`).

3. **Predict** fantasy points  
   - By default, runs **XGBoost** in daily mode (`predict_fp_xgb_daily.predict_fp`), using a rolling 10-day window.  
   - Alternatively, can call **RNN**-based scripts or hyperparameter tuning. ##TODO: change when it's ready

4. **Optimize** lineups  
   - **`get_lineup()`** in `lineup_genetic_optimizer.py` uses a **genetic algorithm** to select a high-upside DFS lineup under a salary cap.

5. **Save** final lineup  
   - Writes to **`output_csv/final_lineup_<date>.csv`** with predicted fantasy points.

6. **(Optional) Analyze** real DFS contests vs. your predicted lineups (see [Contest Outcome Analysis](#contest-outcome-analysis)).

---

## 3. Key Scripts and Modules <a name="key-scripts-and-modules"></a>
- **`main.py`**  
  - The central script orchestrating **preprocessing**, **enrichment**, **prediction**, and **genetic lineup** generation.  

- **`src.preprocessing`**  
  - `merge_all_seasons()`: merges multiple season CSVs.  
  - `preprocess_all_seasons_data()`: standardizes columns & cleans data.

- **`src.data_enrichment`**  
  - `add_time_dependent_features_v2()`: rolling/diffs for the last N games.  
  - `add_running_season_stats()`: accumulative stats within the same season.  
  - `add_last_season_data_with_extras()`: merges prior season’s data.

- **`src.predict_fp_xgb_daily` / `src.predict_fp_rnn_*`**  
  - Implements daily or weekly rolling predictions with **XGBoost** or **RNN**.  
  - By default, `main.py` calls the daily XGBoost method.

- **`src.lineup_genetic_optimizer`**  
  - `get_lineup()`: runs a **genetic algorithm** to build an optimal DFS lineup.

---
## 4. Usage <a name="usage"></a>
1. **Clone & Install**  
   ```bash
   git clone https://github.com/tzachpach/final_project_dfs.git
   cd final_project_dfs
   pip install -r requirements.txt
    ```
   
* For Apple Silicon GPU acceleration with PyTorch, install PyTorch with MPS support.

2. **Run the Pipeline**  
   - **`main.py`** orchestrates the entire pipeline.  
   - By default, it runs XGBoost for daily predictions and genetic lineup optimization.  
   - To run the RNN model, use the appropriate script in **`src.predict_fp_rnn_*`**.

---

## 5. Contest Outcome Analysis <a name="contest-outcome-analysis"></a>
To see how your predicted lineups might fare in actual DFS contests, use the notebook that merges:

* Contest data (fanduel_nba_contests.csv)
* Your predicted lineup CSV (final_lineup_<date>.csv)

### Steps
1. Generate final lineup: After running main.py, you'll have a file like `output_csv/final_lineup_<date>.csv`.
Open the notebook (e.g., `analyze_contests_vs_pred.ipynb`):
* It loads contest results (winning score, min cash line, etc.). 
* Merges with your predicted lineups by date. 
* Calculates:
  * Score difference from winning/min-cash lines.
  * Whether you would’ve cashed or won. 
  * Potential profit given entry fee and min-cash/winning payouts. 
  * Cumulative or daily profit over time (visualized with Seaborn/Matplotlib).
  See `merged_df['pred_lineup_profit']` for final profit calculations.

Below is a sample snippet:

```
# Basic Win & Cash Consistency
num_contests = len(merged_df)
win_pred_count = merged_df['pred_lineup_would_win'].sum()
cash_pred_count = merged_df['pred_lineup_would_cash'].sum()

print("Number of contests:", num_contests)
print("Win rate (predicted):", win_pred_count / num_contests)
print("Cash rate (predicted):", cash_pred_count / num_contests)

# Profit Calculation
merged_df['pred_lineup_profit'] = merged_df.apply(compute_profit, axis=1)
total_profit = merged_df['pred_lineup_profit'].sum()
avg_profit   = merged_df['pred_lineup_profit'].mean()
print(f"Total profit: ${total_profit:.2f}, Average per contest: ${avg_profit:.2f}")
```

Use the visuals (line plots, histograms) to evaluate how close your lineups come to cashing or winning and track long-term performance.

---

## 6. Future Enhancements <a name="future-enhancements"></a>
* **Extended Feature Engineering**: advanced metrics (pace, usage, synergy, real-time injuries).
* **Enhanced Tuning**: Bayesian methods or neural architecture search for RNN layers. 
* **Live Integration**: Real-time DFS salaries & last-minute changes.
* **Alternative Optimizers**: Reinforcement learning or specialized dynamic programming for lineup selection.
* **Detailed Bankroll Management**: automatic tracking of ROI across contests, multiple lineup entries, etc.

---

Enjoy using this pipeline for NBA daily fantasy! Please open an issue or submit a pull request for suggestions or improvements.
