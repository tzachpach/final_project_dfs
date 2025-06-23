"""
Turn the Python list `model_configs` into a flat stream of atomic
configuration dicts (one per experiment).

Every value that *is already a list/tuple* is treated as a hyper‑parameter
dimension; scalars are kept as‑is.  The Cartesian product of all list‑values
is yielded one‑by‑one.
"""

from itertools import product
import json
from copy import deepcopy

from config.model_configs import model_configs


def _generate_run_name(cfg):
    """Generates an informative run name from the config dictionary."""
    parts = [cfg["model_type"].lower()]

    if cfg["model_type"] == "RNN":
        parts.append(cfg.get("rnn_type", "rnn").lower())

    parts.append(cfg.get("mode", "n/a"))

    # Add window and lookback sizes
    mode = cfg.get("mode")
    if mode == "daily":
        if "train_window_days" in cfg: parts.append(f"dwin{cfg['train_window_days']}")
        if "lookback_daily" in cfg: parts.append(f"dlb{cfg['lookback_daily']}")
    elif mode == "weekly":
        if "train_window_weeks" in cfg: parts.append(f"wwin{cfg['train_window_weeks']}")
        if "lookback_weekly" in cfg: parts.append(f"wlb{cfg['lookback_weekly']}")
    
    if cfg["model_type"] == "TST" and "lookback" in cfg:
        parts.append(f"lb{cfg['lookback']}")

    # Add model-specific hyperparameters
    if cfg["model_type"] == "XGBoost" and "xgb_params" in cfg:
        key_map = {"max_depth": "md", "eta": "lr", "subsample": "ss", 
                   "colsample_bytree": "cs_t", "colsample_bylevel": "cs_l",
                   "reg_alpha": "a", "reg_lambda": "l2", "min_child_weight": "mcw", "gamma": "g"}
        for k, v in cfg["xgb_params"].items():
            short_key = key_map.get(k, k)
            val_str = str(v).replace('.', 'p')
            parts.append(f"{short_key}{val_str}")
    
    elif cfg["model_type"] == "RNN":
        if "hidden_size" in cfg: parts.append(f"hs{cfg['hidden_size']}")
        if "num_layers" in cfg: parts.append(f"nl{cfg['num_layers']}")
        if "epochs" in cfg: parts.append(f"ep{cfg['epochs']}")

    elif cfg["model_type"] == "TST" and "tst_config" in cfg:
        tst_cfg = cfg["tst_config"]
        if "model_dim" in tst_cfg: parts.append(f"dim{tst_cfg['model_dim']}")
        if "num_heads" in tst_cfg: parts.append(f"nh{tst_cfg['num_heads']}")

    # Add common flags
    if cfg.get("multi_target_mode"):
        parts.append("mt")
    if "reduce_features_flag" in cfg:
        parts.append(cfg["reduce_features_flag"].lower())

    # Add a concise representation of salary thresholds
    thresholds = cfg.get("salary_thresholds")
    if thresholds:
        thresh_str = "sal-" + "-".join(map(lambda x: str(int(x * 100)), thresholds))
        parts.append(thresh_str)

    return "_".join(parts)


def iter_cfgs():
    """
    Iterate over all blocks, handling mode-specific (daily/weekly) parameters
    correctly to avoid unintended Cartesian products.
    """
    for block in model_configs:
        common_params = {}
        daily_params = {}
        weekly_params = {}
        modes = []

        # Separate parameters into common, daily, and weekly sets
        for key, value in block.items():
            if key == "mode":
                modes = value if isinstance(value, list) else [value]
                continue
            
            if "daily" in key:
                daily_params[key] = value
            elif "weekly" in key:
                weekly_params[key] = value
            else:
                common_params[key] = value

        def get_product(params: dict):
            """Helper to get the Cartesian product of a parameter dictionary."""
            if not params:
                return [{}]
            keys = params.keys()
            value_lists = [v if isinstance(v, list) else [v] for v in params.values()]
            return [dict(zip(keys, p)) for p in product(*value_lists)]

        common_products = get_product(common_params)
        daily_products = get_product(daily_params)
        weekly_products = get_product(weekly_params)

        if "daily" in modes:
            for common_cfg in common_products:
                for daily_cfg in daily_products:
                    cfg = {**common_cfg, **daily_cfg, "mode": "daily"}
                    cfg["run_name"] = _generate_run_name(cfg)
                    if cfg.get("model_type") == "TST" and "tst_config" in cfg:
                        cfg.update(cfg["tst_config"])
                    yield cfg

        if "weekly" in modes:
            for common_cfg in common_products:
                for weekly_cfg in weekly_products:
                    cfg = {**common_cfg, **weekly_cfg, "mode": "weekly"}
                    cfg["run_name"] = _generate_run_name(cfg)
                    if cfg.get("model_type") == "TST" and "tst_config" in cfg:
                        cfg.update(cfg["tst_config"])
                    yield cfg
