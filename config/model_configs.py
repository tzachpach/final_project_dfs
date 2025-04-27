model_configs = [
    {
    "model_type": "XGBoost",
    # We want to try multiple threshold sets
    "thresholds": [
        [0.9, 0.6, 0.0],
        [0.7, 0.0],
        # [0.95, 0.7, 0.0]
    ],
    "mode": ["daily"],
    "train_window_days": [20, 30],  # two possible day-window values
    "train_window_weeks": [4],  # single int for weekly
    "save_model": [True],
    # multiple possible XGB param combos
    "xgb_params": [
        {},  # the default
        {"max_depth": 3},
        # {"max_depth": 5, "learning_rate": 0.05},
        {"max_depth": 5, "learning_rate": 0.01},
    ],
    "model_dir": ["models"]
},
 {
  "model_type": "RNN",
  "mode": ["weekly"],
  "train_window_weeks": [3, 6],
  "train_window_days": [1],
  "hidden_size": [3, 5],
  "num_layers": [2, 4],
  "learning_rate": [0.001, 0.01],
  "dropout_rate": [0.2],
  "epochs": [10, 20],
  "batch_size": [32, 64],
  "rnn_type": ["LSTM"],
  "salary_thresholds": [[0.9, 0.6, 0.0],[0.7, 0.0]],
  "multi_target_mode": [False, True],
  "predict_ahead": [1],
 }
]
# Add more model configurations as needed