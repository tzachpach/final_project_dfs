model_configs = [
    # {
    #     "model_type": "XGBoost",
    #     # ── DFS salary‑bin strategies (2) ────────────────────────────────
    #     "salary_thresholds": [
    #         [0.9, 0.6, 0.0],  # top‑10% / mid‑30% / rest
    #         [0.95, 0.8, 0.0],  # top‑5% / mid-15% / rest
    #         [0.0],  # rest
    #     ],
    #     # ── Mode toggle (2) ──────────────────────────────────────────────
    #     "mode": ["weekly", "daily"],  # ← both evaluated
    #     # Look‑back windows (1 each → they do **not** multiply)
    #     "train_window_days": [20, 30, 60],  # used only when mode == "daily"
    #     "train_window_weeks": [4,6, 8],  # used only when mode == "weekly"
    #     "save_model": [True],
    #     # ── Booster depth / LR pairs (2) ─────────────────────────────────
    #     "xgb_params": [
    #         # {},  # baseline (depth‑6, η0.3)
    #         {"max_depth": 5, "eta": 0.05},  # deeper, slower LR
    #     ],
    #     "model_dir": ["models"],
    #     "reduce_features_flag": ["Kbest", "PCA", False],
    #     # ── Optuna specific configurations ────────────────────────────────
    #     "use_optuna": [True],
    #     "optuna_params": {
    #         "n_trials": 100,
    #         "timeout": 7200,  # 2 hour timeout
    #         "metric": "rmse",  # metric to optimize
    #         "direction": "minimize",  # minimize RMSE
    #         "param_ranges": {
    #             "max_depth": [3, 9],
    #             "learning_rate": [0.01, 0.3],
    #             "n_estimators": [50, 500],
    #             "subsample": [0.6, 1.0],
    #             "colsample_bytree": [0.6, 1.0],
    #             "min_child_weight": [1, 10],
    #             "gamma": [0, 1],
    #             "reg_alpha": [0, 1],
    #             "reg_lambda": [1, 10]
    #         }
    #     }
    # },
    {
        "model_type": "RNN",
        "rnn_type": ["LSTM", "GRU"],
        # Mode toggle (2)
        "mode": ["daily", "weekly"],
        # Look‑back / window (1 each)
        "lookback_daily": [7],      # full param grids in commented brackets [] below
        "lookback_weekly": [15],    # [10, 15, 20]
        "train_window_days": [30, 60],  # used in daily mode [30, 45, 60]
        "train_window_weeks": [6, 10],  # used in weekly mode [6, 8, 10]
        # Network capacity grid: hidden size (1) × layers (2) = 2
        "hidden_size": [32],            # [32, 64]
        "num_layers": [1, 3],              # [1, 3]
        # Salary‑bin strategies (2)
        "salary_thresholds": [
            [0.9, 0.6, 0.0],  # top‑10% / mid‑30% / rest
            # [0.95, 0.8, 0.0],  # top‑5% / mid-15% / rest
            # [0.0],  # rest
        ],
        # Fixed training hyper‑params to keep runtime low
        "learning_rate": [0.01],
        "dropout_rate": [0.3],          
        "epochs": [10, 25],
        "batch_size": [32],             
        "multi_target_mode": [False, True],
        "predict_ahead": [1],
        "reduce_features_flag": ["PCA", "Kbest"],
        # ── Optuna specific configurations ────────────────────────────────
    },
]
