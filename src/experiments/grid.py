"""
Turn the Python list `model_configs` into a flat stream of atomic
configuration dicts (one per experiment).

Every value that *is already a list/tuple* is treated as a hyper‑parameter
dimension; scalars are kept as‑is.  The Cartesian product of all list‑values
is yielded one‑by‑one.
"""

from itertools import product
from datetime import datetime

from config.model_configs import model_configs


def _generate_run_name(cfg):
    """Generates an informative run name from the config dictionary."""
    parts = []
    
    # 1. Model type (required)
    model_type = cfg["model_type"].lower()
    parts.append(model_type)
    
    # 2. Group by (required)
    parts.append(cfg.get("mode", "unknown"))
    
    # 3. Training window and lookback
    if cfg.get("mode") == "daily":
        parts.append(f"tw{cfg.get('train_window_days', 0)}")
        parts.append(f"lb{cfg.get('lookback_daily', 0)}")
    else:  # weekly
        parts.append(f"tw{cfg.get('train_window_weeks', 0)}")
        parts.append(f"lb{cfg.get('lookback_weekly', 0)}")
    
    # 4. Model-specific parameters
    if model_type == "rnn":
        parts.append(f"drop{cfg.get('dropout_rate', 0)}")
        parts.append(f"ep{cfg.get('epochs', 0)}")
        parts.append(f"h{cfg.get('hidden_size', 0)}")
        parts.append(f"l{cfg.get('num_layers', 1)}")
        parts.append(f"lr{cfg.get('learning_rate', 0)}")
        parts.append(f"b{cfg.get('batch_size', 32)}")
        parts.append(cfg.get('rnn_type', 'lstm').lower())
    elif model_type == "tst":
        tst_cfg = cfg.get("tst_config", {})
        parts.append(f"dim{tst_cfg.get('model_dim', 0)}")
        parts.append(f"nh{tst_cfg.get('num_heads', 0)}")
        parts.append(f"nl{tst_cfg.get('num_layers', 1)}")
        parts.append(f"drop{tst_cfg.get('dropout', 0)}")
        parts.append(f"ep{tst_cfg.get('epochs', 0)}")
        parts.append(f"b{tst_cfg.get('batch_size', 32)}")
        parts.append(f"lr{tst_cfg.get('learning_rate', 0)}")
    elif model_type == "xgb":
        xgb_params = cfg.get('xgb_params', {})
        parts.append(f"md{xgb_params.get('max_depth', 0)}")
        parts.append(f"lr{xgb_params.get('eta', 0)}")
        parts.append(f"ss{xgb_params.get('subsample', 1.0)}")
        parts.append(f"cst{xgb_params.get('colsample_bytree', 1.0)}")
        parts.append(f"csl{xgb_params.get('colsample_bylevel', 1.0)}")
        parts.append(f"mcw{xgb_params.get('min_child_weight', 1)}")
        parts.append(f"g{xgb_params.get('gamma', 0)}")
    
    # 5. Feature engineering flags
    if cfg.get("reduce_features_flag"):
        parts.append(f"feat{cfg['reduce_features_flag'].lower()}")
    
    # 6. Multi-target mode
    if cfg.get("multi_target_mode"):
        parts.append("multi")
    
    # 7. Salary thresholds - include all thresholds
    thresholds = cfg.get("salary_thresholds", [])
    if thresholds:
        thresh_str = "sal" + "_".join(str(int(t * 100)) for t in thresholds) + "p"
        parts.append(thresh_str)
    
    # 8. Platform
    if "platform" in cfg:
        parts.append(cfg["platform"])
    
    # Clean and join parts
    parts = [str(p).replace(".", "p") for p in parts]  # Replace dots with 'p'
    
    # 9. Add timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts.append(timestamp)
    
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
