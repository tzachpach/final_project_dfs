import logging
import os
import time
from pathlib import Path
import multiprocessing
import traceback
import sys
from multiprocessing import Manager, Process

import mlflow
import pandas as pd

from config.helpers import get_predictions_df, get_lineup
from src.experiments.grid import iter_cfgs
from src.experiments.ml_utils import log_cfg
from src.evaluate_results import evaluate_results
from config.pipelines import preprocess_pipeline, enrich_pipeline

import warnings

warnings.filterwarnings("ignore", category=Warning, module="urllib3")

# --- MLflow Tracking URI ---
# It's safer to set this as an absolute path, especially for multiprocessing.
MLRUNS_DIR = os.path.abspath("mlruns")
MLFLOW_TRACKING_URI = f"file://{MLRUNS_DIR}"

# Create artifacts directory if it doesn't exist
os.makedirs("artifacts", exist_ok=True)

# ── one‑time data prep will be moved into main() ──────────────────

# Global variables for worker processes
worker_enriched_df = None
worker_contests_df = None
progress_queue = None

def init_worker(enriched_df, contests_df, queue):
    """Initializer for each worker process."""
    global worker_enriched_df, worker_contests_df, progress_queue
    worker_enriched_df = enriched_df
    worker_contests_df = contests_df
    progress_queue = queue


def _log_progress(run_name, phase):
    """Safely sends progress messages to the queue if it exists."""
    if progress_queue:
        try:
            progress_queue.put((run_name, phase))
        except Exception:
            # Fails silently if queue is broken. The run will continue.
            pass


def progress_listener(queue):
    """Listens for progress messages and prints them to the console."""
    while True:
        try:
            message = queue.get()
            if message is None:  # Sentinel to stop the listener
                break
            run_name, phase = message
            if phase == 'Started':
                print(f"  PROGRESS: {run_name} -> {phase}.")
            else:
                print(f"  PROGRESS: {run_name} -> {phase} complete.")
        except (KeyboardInterrupt, SystemExit):
            break
        except Exception:
            traceback.print_exc()


def run_one_cfg(cfg):
    try:
        _log_progress(cfg['run_name'], 'Started')
        # Set the run name
        log_cfg(cfg)

        t0 = time.time()
        print(f"Running {cfg['run_name']}...")

        # ---------------------------------------------------------
        # 1) Generate predictions
        # ---------------------------------------------------------
        preds = get_predictions_df(cfg, worker_enriched_df)

        logging.info(f"Predictions {cfg['run_name']} complete")
        _log_progress(cfg['run_name'], 'Prediction')

        # Ensure run output directory exists once
        base_run_dir = Path("results") / cfg["run_name"]
        base_run_dir.mkdir(parents=True, exist_ok=True)

        def _save_and_log(df: pd.DataFrame, rel_name: str):
            """Save DataFrame under base_run_dir / rel_name and log to MLflow."""
            fpath = base_run_dir / rel_name
            # Ensure parent directories exist
            fpath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(fpath, index=False)
            mlflow.log_artifact(fpath)
            return fpath

        _save_and_log(preds, "predictions.csv")

        # ---------------------------------------------------------
        # 2) Build optimal lineup from predictions
        # ---------------------------------------------------------
        lineup = get_lineup(preds)
        logging.info(f"Optimization {cfg['run_name']} complete")
        _log_progress(cfg['run_name'], 'Lineup')

        _save_and_log(lineup, "lineup.csv")

        # ---------------------------------------------------------
        # 3) Evaluate results (overall KPIs & percentiles)
        # ---------------------------------------------------------
        kpis_df, pct_df, kpis_dict = evaluate_results(
            preds, lineup, worker_contests_df, group_by_season=False
        )

        # Additionally, compute per-season KPIs
        season_kpis_df, season_pct_df = evaluate_results(
            preds, lineup, worker_contests_df, group_by_season=True
        )
        logging.info(f"Evaluation {cfg['run_name']} complete")
        _log_progress(cfg['run_name'], 'Evaluation')

        # Save detailed KPI artefacts first (before logging to MLflow)
        _save_and_log(kpis_df,       "metrics/kpis_overall.csv")
        _save_and_log(pct_df,        "metrics/percentiles_overall.csv")
        _save_and_log(season_kpis_df, "metrics/kpis_by_season.csv")
        _save_and_log(season_pct_df,  "metrics/percentiles_by_season.csv")

        # ---------------------------------------------------------
        # 4) Log numeric metrics to MLflow (filter out non-numeric)
        # ---------------------------------------------------------
        numeric_metrics = {k: float(v) for k, v in kpis_dict.items() if isinstance(v, (int, float))}
        if numeric_metrics:
            mlflow.log_metrics(numeric_metrics)

        mlflow.log_metric("runtime_sec", round(time.time() - t0, 2))
        
        return "SUCCESS"

    except Exception as e:
        print(f"Error in run {cfg.get('run_name', 'unknown')}: {e}")
        traceback.print_exc()
        mlflow.set_tag("run_status", "failed")
        return "FAILED"


def run_cfg_with_mlflow(cfg):
    """
    A wrapper to handle the MLflow start/end run for each parallel process.
    It also redirects stdout/stderr for each run to a separate log file.
    """
    # Create run-specific log directory
    run_name = cfg['run_name']
    log_dir = os.path.join("results", run_name, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Set up log file path
    log_file_path = os.path.join(log_dir, "run.log")

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        with open(log_file_path, 'w') as log_file:
            sys.stdout = log_file
            sys.stderr = log_file

            # Set tracking URI and experiment for each worker process to ensure fork-safety
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment("dfs_capstone")
            with mlflow.start_run(run_name=cfg["run_name"]):
                status = run_one_cfg(cfg)
                # Ensure all output is written to the log file before uploading
                log_file.flush()
                os.fsync(log_file.fileno())
                # Upload the log file as an artifact
                mlflow.log_artifact(log_file_path, artifact_path="logs")

    except Exception as e:
        # If logging setup fails, print error to the original console
        print(f"Error setting up logger for {cfg['run_name']}: {e}", file=original_stderr)
        traceback.print_exc(file=original_stderr)
        status = "FAILED"
    finally:
        # Restore stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
    
    return status, cfg


def main():
    """
    Main function to set up and run experiments in parallel.
    """
    start_time = time.time()
    
    # --- Data preparation is now done once in the main process ---
    print("Loading & preparing data...")
    pre = preprocess_pipeline()
    pre = pre.sort_values("game_date")
    enriched = enrich_pipeline(pre)
    contests = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "data/contests_data/fanduel_nba_contests_all_seasons.csv",
        )
    )
    print("Data preparation complete.")

    # Set the MLflow experiment
    mlflow.set_experiment("dfs_capstone")

    configs = list(iter_cfgs())
    if not configs:
        print("No configurations to run.")
        return
        
    print(f"Found {len(configs)} configurations to run.")

    # Use all but 2 CPUs for safety, with a minimum of 1
    num_processes = 3 #max(1, multiprocessing.cpu_count() - 2)
    print(f"Running {len(configs)} configs on {num_processes} processes...")
    print("Each run's output is being saved to a log file in artifacts/run_logs/")

    results = []
    completed_count = 0
    total_configs = len(configs)

    if num_processes > 1:
        # --- Parallel Execution ---
        print(f"Running {total_configs} configs on {num_processes} processes in parallel...")
        print("Each run's output is being saved to a log file in artifacts/run_logs/")

        # Set up a queue and a listener for progress updates
        manager = Manager()
        queue = manager.Queue()
        listener = Process(target=progress_listener, args=(queue,))
        listener.start()

        with multiprocessing.Pool(
            processes=num_processes,
            initializer=init_worker,
            initargs=(enriched, contests, queue)
        ) as pool:
            # Use imap_unordered to get results as they complete for better progress tracking
            for status, cfg in pool.imap_unordered(run_cfg_with_mlflow, configs):
                completed_count += 1
                results.append((status, cfg))
                run_name = cfg.get('run_name', 'unknown')
                print(f"[{completed_count}/{total_configs}] FINISHED: {run_name} (Status: {status})")

        # Stop the listener
        queue.put(None)
        listener.join()
    else:
        # --- Sequential Execution ---
        print(f"Running {total_configs} configs sequentially in a single process...")
        # In sequential mode, data is already available in the main process
        global worker_enriched_df, worker_contests_df
        worker_enriched_df = enriched
        worker_contests_df = contests

        for cfg in configs:
            run_name = cfg.get('run_name', 'unknown')
            print(f"[{completed_count + 1}/{total_configs}] RUNNING: {run_name}")
            status, _ = run_cfg_with_mlflow(cfg)
            results.append((status, cfg))
            completed_count += 1
            print(f"[{completed_count}/{total_configs}] FINISHED: {run_name} (Status: {status})")


    end_time = time.time()

    # --- Summary ---
    successful_runs = [r for r, c in results if r == "SUCCESS"]
    failed_runs = [c for r, c in results if r == "FAILED"]
    skipped_runs = [c for r, c in results if r == "SKIPPED"]

    print("\n--- Experiment Summary ---")
    print(f"Total configurations run: {len(configs)}")
    print(f"  - Successful: {len(successful_runs)}")
    print(f"  - Failed: {len(failed_runs)}")
    print(f"  - Skipped: {len(skipped_runs)}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    # Optionally log status to a CSV
    summary_df = pd.DataFrame([{"status": r, **c} for r, c in results])
    summary_path = f"artifacts/summary_{int(time.time())}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved run summary to: {summary_path}")

    if failed_runs:
        print("\n--- Failed Configurations ---")
        for i, cfg in enumerate(failed_runs):
            print(f"{i+1}. {cfg['run_name']}")


if __name__ == "__main__":
    main()
