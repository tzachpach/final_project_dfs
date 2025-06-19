
model_configs = [
    {
        "model_type": "XGBoost",
        # ── DFS salary‑bin strategies (2) ────────────────────────────────
        "salary_thresholds": [
            [0.9, 0.6, 0.0],  # top‑10% / mid‑30% / rest
            [0.95, 0.8, 0.0],  # top‑5% / mid-15% / rest
        ],
        "multi_target_mode": [False, True],
        # ── Mode toggle (2) ──────────────────────────────────────────────
        "mode": ["weekly", "daily"],  # ← both evaluated
        # Look‑back windows (1 each → they do **not** multiply)
        "train_window_days": [20, 60],  # used only when mode == "daily"
        "train_window_weeks": [4, 8],  # used only when mode == "weekly"
        "save_model": [True],
        # ── Booster depth / LR pairs (8) ─────────────────────────────────
        "xgb_params": [
            # Basic configurations
            {"max_depth": 4, "eta": 0.1, "subsample": 0.8},    # shallow, fast, moderate subsample
            {"max_depth": 5, "eta": 0.05, "subsample": 0.9},   # medium, medium, high subsample
            {"max_depth": 6, "eta": 0.01, "subsample": 1.0},   # deep, slow, full sample       
            # Column sampling variations
            {"max_depth": 4, "eta": 0.1, "colsample_bytree": 0.8},  # feature sampling
            {"max_depth": 5, "eta": 0.05, "colsample_bylevel": 0.8},  # per-level sampling         
            # Regularization variations
            {"max_depth": 5, "eta": 0.05, "reg_alpha": 0.1, "reg_lambda": 1.0},  # L1/L2
            {"max_depth": 4, "eta": 0.1, "min_child_weight": 3},   # conservative splits
            {"max_depth": 6, "eta": 0.01, "gamma": 0.1},            # pruning control
        ],
        "model_dir": ["models"],
        "reduce_features_flag": ["Kbest", "PCA"],
        # ── Optuna specific configurations ────────────────────────────────
    },
    {
        "model_type": "RNN",
        "rnn_type": ["LSTM", "GRU"],
        # Mode toggle (2)
        "mode": ["daily", "weekly"],
        # Look‑back / window (1 each)
        "lookback_daily": [7],      # full param grids in commented brackets [] below
        "lookback_weekly": [15],    # [10, 15, 20]
        "train_window_days": [30, 60],  # used in daily mode [30, 45, 60], then [30, 60]
        "train_window_weeks": [6, 10],  # used in weekly mode [6, 8, 10], then [6, 10]
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
        "learning_rate": [0.01], # [0.01]
        "dropout_rate": [0.3],
        "epochs": [10, 25], # [10, 25]
        "batch_size": [32],
        "multi_target_mode": [False, True], # [False, True]
        "predict_ahead": [1],
        "reduce_features_flag": ["PCA", "Kbest"],
    },
]
