model_configs = [
    # -------------------------------------------------- 8 XGBoost variants
    {
        "model_type": "XGBoost",
        # ── DFS salary‑bin strategies (2) ────────────────────────────────
        "thresholds": [
            [0.90, 0.60, 0.00],  # top‑10% / mid‑30% / rest
            # [0.75, 0.00],  # top‑25% / rest
        ],
        # ── Mode toggle (2) ──────────────────────────────────────────────
        "mode": ["weekly", "daily"],  # ← both evaluated
        # Look‑back windows (1 each → they do **not** multiply)
        "train_window_days": [30],  # used only when mode == "daily"
        "train_window_weeks": [6],  # used only when mode == "weekly"
        "save_model": [True],
        # ── Booster depth / LR pairs (2) ─────────────────────────────────
        "xgb_params": [
            # {},  # baseline (depth‑6, η0.3)
            {"max_depth": 5, "eta": 0.05},  # deeper, slower LR
        ],
        "model_dir": ["models"],
        "reduce_features_flag": ["Kbest", "PCA", False],
    },
    # -------------------------------------------------- 8 RNN variants
    {
        "model_type": "RNN",
        "rnn_type": ["LSTM"],
        # Mode toggle (2)
        "mode": ["weekly"],
        # Look‑back / window (1 each)
        "train_window_days": [30],  # used in daily mode
        "train_window_weeks": [6],  # used in weekly mode
        # Network capacity grid: hidden size (1) × layers (2) = 2
        "hidden_size": [64],
        "num_layers": [1, 2],
        # Salary‑bin strategies (2)
        "salary_thresholds": [
            [0.90, 0.60, 0.00],
            # [0.75, 0.00],
        ],
        # Fixed training hyper‑params to keep runtime low
        "learning_rate": [0.001],
        "dropout_rate": [0.2],
        "epochs": [15],
        "batch_size": [32],
        "multi_target_mode": [False, True],
        "predict_ahead": [1],
        "reduce_features_flag": ["PCA", "Kbest", False],
    },
]
