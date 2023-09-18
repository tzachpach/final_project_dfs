# Fantasy Basketball Lineup Optimizer
This project aims to optimize the selection of basketball players for a fantasy sports lineup. It uses historical player data to predict individual player performance and then employs a genetic algorithm to select the best combination of players based on those predictions and other constraints, such as a salary cap.

# Key Components:
1. Data Loading (data_loader.py):
Responsible for loading and processing basketball player data from various CSV files.
Returns a DataFrame with relevant player statistics and information.
2. Player Performance Prediction (fetch_y_pred.py):
Uses a model to predict a player's performance in terms of fantasy points.
Predictions are based on historical player data and other relevant features.
3. Lineup Optimization (genetic_selection.py):
Implements a genetic algorithm to select the optimal lineup of players.
Considers constraints like salary cap and positional requirements.
Optimizes based on predicted player performance and other relevant factors.
4. Data Files:
The repository contains several CSV files with historical data on players, their performances in past games, and other relevant statistics.
This data is used for training the prediction model and for the genetic algorithm's optimization process.

# Usage:
Ensure all dependencies are installed.
Load the data using data_loader.py.
Generate player performance predictions using fetch_y_pred.py.
Optimize your lineup using genetic_selection.py.

# Future Enhancements:
* Integrate real-time player pricing from fantasy platforms
* Feature engineering - currently the features that are implemented are the averages from the 5 and 10 previous games. Other, and especially price-based features, should be implemented.
* Explore other prediction models for improved accuracy (DL?)
* Implement additional optimization algorithms for lineup selection - (RL. LLM?)
