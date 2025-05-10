import mlflow
import git
import json
import time
from pathlib import Path

# from omegaconf import OmegaConf
from main import main  # your current main()

REPO_ROOT = Path(__file__).resolve().parents[2]
repo = git.Repo(REPO_ROOT)


def run_with_mlflow(cfg: dict):
    mlflow.set_experiment("dfs_capstone")

    with mlflow.start_run(run_name=cfg["run_name"]):
        # ---- meta ----
        mlflow.log_params(cfg)
        mlflow.set_tag("git_sha", repo.head.commit.hexsha)
        mlflow.set_tag("git_dirty", repo.is_dirty())

        t0 = time.time()
        results_path = main(cfg)  # call your existing pipeline
        runtime = round(time.time() - t0, 2)

        # ---- metrics & artefacts ----
        res = json.load(open(results_path))  # or read a CSV
        mlflow.log_metrics(res)
        mlflow.log_metric("runtime_sec", runtime)
        mlflow.log_artifact(results_path)


if __name__ == "__main__":
    cfg = {
        "run_name": "quick_debug",
        "model": "xgb",
        "train_window": 20,
        "hidden_size": 64,
        # …anything else you want to see in MLflow
    }
    run_with_mlflow(cfg)
