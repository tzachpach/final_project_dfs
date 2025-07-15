model_configs = [
    # {
    #     "model_type": "TST",
    #     # Weekly mode only
    #     "mode": ["weekly"],

    #     # Sequence and rolling window lengths
    #     "lookback_weekly": [20],   # lb20 in run name
    #     "train_window_weeks": [15], # tw15 in run name

    #     # Salary-binning strategy: elite 20% & mid 30% thresholds
    #     "salary_thresholds": [[0.8, 0.5]],
    #     # Multi-target prediction enabled
    #     "multi_target_mode": [True],
    #     # Keep full feature set (no reduction)
    #     "reduce_features_flag": [False],

    #     # Transformer configuration – dim64 / heads4 / layers3 / drop0.1 / ep10 / bs32 / lr0.001
    #     "tst_config": [
    #         {
    #             "model_dim": 64,
    #             "num_heads": 4,
    #             "num_layers": 3,
    #             "dropout": 0.1,
    #             "epochs": 10,
    #             "batch_size": 32,
    #             "learning_rate": 0.001,
    #         }
    #     ],
    # },
    
        {
        "model_type": "TST",
        "mode": ["daily"],
        "lookback_daily": [15],  # lb15
        "train_window_days": [60],  # tw8
        "salary_thresholds": [[0.0]],  # sal0p (all players)
        "multi_target_mode": [True],
        "reduce_features_flag": [False],
        "tst_config": [
            {
                "model_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "dropout": 0.1,
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001,
            }
        ],
    },

    
]
