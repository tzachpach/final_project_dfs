# Fantasy Basketball Lineup Optimizer
This project aims to optimize the selection of basketball players for a fantasy sports lineup. It uses historical player data to predict individual player performance and then employs a genetic algorithm to select the best combination of players based on those predictions and other constraints, such as a salary cap.

# Key Components:
1. Data Loading (merge_salaries_and_gamelogs_v2.py):
While the data is being loaded and externally, the script merge_salaries_and_gamelogs_v2.py is responsible for 
merging the existing data into one, comprehensive dataset. This dataset is then used for training the prediction model and for the lineup optimization process.
The notebook should be run before running the other scripts.


2. Player Performance Prediction:
Currently, we use three different methods to predict player performance. 
In all cases, we use a running window of training and testing, where the moving average of the player's previous performance is used to predict his upcoming performance.
We currently use three types of models:
2.1 XGBoost, daily (predict_fp_xgb_daily.py) - trains every day on the last 10 days and predicts the next day's performance.
2.2 XGBoost, weekly (predict_fp_xgb_weekly.py) - trains over the last 4 weeks of player performance and predicts the next week's performance.
2.3 RNN, daily (predict_fp_rnn_daily.py) - trains every day on the last 10 days and predicts the next day's performance.

3. Lineup Optimization (genetic_selection.py):
Implements a genetic algorithm to select the optimal lineup of players.
Considers constraints like salary cap and positional requirements.
Optimizes based on predicted player performance and other relevant factors.

4. Data Files:
The repository contains several CSV files with historical data on players, their performances in past games, and other relevant statistics.
This data is used for training the prediction model and for the genetic algorithm's optimization process.

5. Evaluation::
The evaluation of the player performnace prediction model is done on analyze_fp_pred.ipynb notebook.
The notebook looks at the top 10 players per each position and evaluates the model's performance.

# Usage:
Run the merge_salaries_and_gamelogs_v2.py script to merge the data files.
Generate player performance predictions using either one of the predicting scripts.
Optimize your lineup using genetic_selection.py.

# Future Enhancements:
* Evaluate the performance of the Genetic Algorithm, by both using optimal data and by comparing overall result to contest data and see how much competition it would have won. 
* Integrate real-time player pricing from fantasy platforms
* Feature engineering - currently the features that are implemented are the averages from the 5 and 10 previous games. Other, and especially price-based features, should be implemented.
* Explore other prediction models for improved accuracy (DL?)
* Implement additional optimization algorithms for lineup selection - (RL. LLM?)