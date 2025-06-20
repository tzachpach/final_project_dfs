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


def _cartesianize(block: dict):
    """
    Yield fully‑specified config dicts for a single block in model_configs.
    """
    keys, value_lists = [], []

    for k, v in block.items():
        # treat *every* list/tuple as a sweep dimension
        if isinstance(v, (list, tuple)):
            value_lists.append(list(v))
        else:  # scalar → wrap so it doesn't multiply
            value_lists.append([v])
        keys.append(k)

    for combo in product(*value_lists):
        cfg = {k: deepcopy(v) for k, v in zip(keys, combo)}  # stop aliasing issues
        yield cfg


def iter_cfgs():
    """
    Iterate over **all** blocks and all Cartesian products inside each block.
    Adds a human‑readable 'run_name' field to every emitted cfg.
    """
    for block in model_configs:
        for cfg in _cartesianize(block):
            # -------- nice, short name for MLflow UI ----------
            thresh_str = (
                json.dumps(cfg.get("salary_thresholds", []))
                .replace("[", "")
                .replace("]", "")
                .replace(" ", "")
            )
            cfg["run_name"] = (
                f'{cfg["model_type"].lower()}_'
                f'{cfg.get("mode", "n/a")}_'
                f"{thresh_str[:12]}"  # trim for brevity
            )
            if cfg["model_type"] == "TST" and "tst_config" in cfg:
                cfg.update(cfg["tst_config"])  # flatten for ease-of-use
            yield cfg
