import os
import time
import mlflow
import pandas as pd

from config.helpers import get_predictions_df, get_lineup
from src.experiments.grid import iter_cfgs
from src.experiments.ml_utils import start_run, log_cfg
from src.evaluate_results import evaluate_results
from config.pipelines import preprocess_pipeline, enrich_pipeline

import warnings

warnings.filterwarnings("ignore", category=Warning, module="urllib3")

# Create artifacts directory if it doesn't exist
os.makedirs("artifacts", exist_ok=True)

# ── one‑time data prep ─────────────────────────────────────────
pre = preprocess_pipeline()
pre = pre[pre.season_year.isin(["2016-17", "2017-18"])].sort_values("game_date")
enriched = enrich_pipeline(pre)
contests = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data/contests_data/fanduel_nba_contests.csv",
    )
)

# ── iterate grid ───────────────────────────────────────────────
for cfg in iter_cfgs():
    try:
        with start_run(cfg):
            try:
                # Set the run name
                log_cfg(cfg)

                t0 = time.time()
                preds = get_predictions_df(cfg, enriched)

                if preds.empty:
                    mlflow.log_metric("skip_empty", 1)
                    continue

                lineup = get_lineup(preds)
                kpis, pct = evaluate_results(preds, lineup, contests)

                mlflow.log_metrics(kpis)
                mlflow.log_metric("runtime_sec", round(time.time() - t0, 2))

                out_csv = f'artifacts/{cfg["run_name"]}_percentiles.csv'
                pct.to_csv(out_csv, index=False)

                try:
                    mlflow.log_artifact(out_csv)
                except Exception as e:
                    print(f"Error logging artifact: {e}")
            except Exception as e:
                print(f"Error in run {cfg.get('run_name', 'unknown')}: {e}")
    except Exception as e:
        print(f"Error starting run for config {cfg.get('run_name', 'unknown')}: {e}")
