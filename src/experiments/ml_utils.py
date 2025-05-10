import json
import git
import mlflow
from pathlib import Path

REPO = git.Repo(Path(__file__).resolve().parents[2])


def flatten(obj, p=""):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(flatten(v, f"{p}{k}."))
        return out
    if isinstance(obj, (list, tuple)):
        return {p[:-1]: json.dumps(obj)}
    return {p[:-1]: obj}


def start_run(cfg):
    mlflow.set_experiment("dfs_capstone")
    return mlflow.start_run(run_name=cfg["run_name"])


def log_cfg(cfg):
    mlflow.log_params(flatten(cfg))
    mlflow.set_tag("git_sha", REPO.head.commit.hexsha)
    mlflow.set_tag("git_dirty", REPO.is_dirty())
