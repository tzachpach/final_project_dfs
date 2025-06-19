import os
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import mlflow
import pandas as pd

from config.helpers import get_predictions_df, get_lineup
from src.experiments.grid import iter_cfgs
from src.experiments.ml_utils import log_cfg
from src.evaluate_results import evaluate_results, format_metrics_for_logging
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
        "data/contests_data/fanduel_nba_contests_all_seasons.csv",
    )
)


def run_one_cfg(cfg):
    try:
        # Set the run name
        log_cfg(cfg)

        t0 = time.time()
        preds = get_predictions_df(cfg, enriched)
        if preds.empty:
            mlflow.log_metric("skip_empty", 1)
            print(f"Empty predictions for {cfg['run_name']}. Skipping.")
            return

        lineup = get_lineup(preds)

        kpis, pct = evaluate_results(preds, lineup, contests)

        clean_kpis = {k: (v if pd.notna(v) else -1.0) for k, v in kpis.items()}

        mlflow.log_metrics(clean_kpis)
        mlflow.log_metric("runtime_sec", round(time.time() - t0, 2))

        formatted_kpis = format_metrics_for_logging(clean_kpis)
        # Log the formatted metrics as tags so they appear as-is in the UI (MLflow metrics must be float)
        for key, value in formatted_kpis.items():
            mlflow.set_tag(key + "_formatted", value)

        run_dir = Path(urlparse(mlflow.get_artifact_uri()).path)
        run_dir.mkdir(parents=True, exist_ok=True)

        lineup_path = run_dir / f"{cfg['run_name']}_{uuid.uuid4().hex[:6]}_lineup.csv"
        lineup.to_csv(lineup_path, index=False)

        # save artefacts
        pct_path = run_dir / f"{cfg['run_name']}_{uuid.uuid4().hex[:6]}_percentiles.csv"
        pct.to_csv(pct_path, index=False)
        mlflow.log_artifact(pct_path)

        kpi_path = run_dir / f"{cfg['run_name']}_kpis.csv"
        pd.DataFrame([clean_kpis]).to_csv(kpi_path, index=False)
        mlflow.log_artifact(kpi_path)

    except Exception as e:
        print(f"Error in run {cfg.get('run_name', 'unknown')}: {e}")


for cfg in iter_cfgs():
    mlflow.start_run(run_name=cfg["run_name"])
    try:
        run_one_cfg(cfg)
    except Exception as e:
        print(f"Run {cfg['run_name']} crashed: {e}")
        mlflow.set_tag("run_status", "failed")
    finally:
        mlflow.end_run()
