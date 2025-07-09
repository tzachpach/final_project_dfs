model_configs = [
    # {
    #     "model_type": "XGBoost",
    #     # ── DFS salary-bin strategies (now 4) ───────────────────────────
    #     # Each list must be in descending order; a trailing 0.0 isn't needed
    #     # because the code automatically treats the last threshold as the lower bound.
    #     "salary_thresholds": [
    #         [0.95, 0.85, 0.7],  # elite 5 % | star 15 % | starter 30 %
    #         [0.85, 0.6],        # elite 15 % | rest
    #     ],
    #     "multi_target_mode": [True],
    #     # ── Mode toggle (2) ──────────────────────────────────────────────
    #     "mode": ["weekly", "daily"],  # both evaluated
    #     # Look-back windows (they do **not** multiply with each other, only with mode)
    #     "train_window_days": [60, 90],   # placeholder single value (not used)
    #     "train_window_weeks": [10, 15],  # two options
    #     "save_model": [True],
    #     # ── Booster hyper-parameter grid (6) ────────────────────────────
    #     # Each dict may also include colsample_bytree, min_child_weight, etc.
    #     "xgb_params": [
    #         # depth / lr sweeps
    #         {"max_depth": 6,  "eta": 0.03, "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 3, "gamma": 0.1, "lambda": 1.0},
    #         {"max_depth": 8,  "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 2, "gamma": 0.05, "lambda": 1.0},
    #         {"max_depth": 10, "eta": 0.02, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 4, "gamma": 0.0, "lambda": 1.0},
    #         {"max_depth": 12, "eta": 0.015, "subsample": 0.9, "colsample_bytree": 0.9, "min_child_weight": 3, "gamma": 0.0, "lambda": 0.5},
    #     ],
    #     "model_dir": ["models"],
    #     "reduce_features_flag": [False],
    # },
    # {
    #     "model_type": "RNN",
    #     "rnn_type": ["LSTM", "GRU"],
    #     # Mode toggle (2)
    #     "mode": ["weekly"],
    #     # Look-back / window (1 each)
    #     "lookback_daily": [7],      # full param grids in commented brackets [] below
    #     "lookback_weekly": [15],    # [10, 15, 20]
    #     "train_window_days": [90],  # used in daily mode [30, 45, 60], then [30, 60]
    #     "train_window_weeks": [15],  # used in weekly mode [6, 8, 10], then [6, 10]
    #     # Network capacity grid: hidden size (1) × layers (2) = 2
    #     "hidden_size": [32],            # [32, 64]
    #     "num_layers": [1],              # [1, 3]
    #     # Salary-bin strategies (2)
    #     "salary_thresholds": [
    #         # [0.9, 0.6, 0.0],  # top-10% / mid-30% / rest
    #         # [0.95, 0.8, 0.0],  # top-5% / mid-15% / rest
    #         [0.0],  # rest
    #     ],
    #     # Fixed training hyper-params to keep runtime low
    #     "learning_rate": [0.01], # [0.01]
    #     "dropout_rate": [0.3],
    #     "epochs": [25], # [10, 25]
    #     "batch_size": [32],
    #     "multi_target_mode": [False, True], # [False, True]
    #     "predict_ahead": [1],
    #     "reduce_features_flag": ["Kbest"],
    # },
    {
        "model_type": "TST",
        # Run both granular daily and coarser weekly modes
        "mode": ["daily", "weekly"],

        # Time windows – focused around the configs that already work well
        "lookback_daily": [10],   # sequence length for daily mode
        "lookback_weekly": [15],  # sequence length for weekly mode

        "train_window_days": [60, 90],     # daily rolling window
        "train_window_weeks": [10, 15],     # weekly rolling window

        # Salary-binning — baseline (all players) + top-slice
        "salary_thresholds": [
            [0.0],          # all players
            [0.8, 0.5, 0.0] # elite 20%, mid 30%, rest
        ],

        # Evaluate both single-target and multi-target modes
        "multi_target_mode": [False, True],

        # Feature reducer grid
        "reduce_features_flag": [False, "Kbest"],

        # Transformer hyper-parameters – 2×2×2 grid  (total 8)
        "tst_config": [
            {"model_dim": 64,  "num_heads": 4, "num_layers": 2, "dropout": 0.1, "epochs": 10, "batch_size": 32, "learning_rate": 0.001},
            {"model_dim": 64,  "num_heads": 4, "num_layers": 5, "dropout": 0.1, "epochs": 20, "batch_size": 32, "learning_rate": 0.0007},
            {"model_dim": 128, "num_heads": 8, "num_layers": 2, "dropout": 0.2, "epochs": 10, "batch_size": 32, "learning_rate": 0.001},
            {"model_dim": 128, "num_heads": 8, "num_layers": 5, "dropout": 0.2, "epochs": 20, "batch_size": 32, "learning_rate": 0.0007},
        ],
    },
]
