import time
from datetime import datetime
import pandas as pd

from config.helpers import get_predictions_df, get_lineup
from config.model_configs import model_configs
from config.pipelines import preprocess_pipeline, enrich_pipeline
from src.evaluate_results import evaluate_results


def cartesian_product(config):
    """
    Generate all combinations of parameters for a given configuration.
    Any parameter that is a list/tuple is treated as a dimension to iterate over.
    """
    from itertools import product
    from copy import deepcopy

    keys, value_lists = [], []

    for k, v in config.items():
        if isinstance(v, (list, tuple)):
            value_lists.append(list(v))
        else:
            value_lists.append([v])
        keys.append(k)

    for combo in product(*value_lists):
        yield {k: deepcopy(v) for k, v in zip(keys, combo)}


def iter_configs():
    """
    Iterate over all configurations and their parameter combinations.
    """
    for config in model_configs:
        for sub_config in cartesian_product(config):
            # Add a run name for better identification
            model_type = sub_config["model_type"].lower()
            mode = sub_config.get("mode", "n/a")
            thresholds = str(sub_config.get("salary_thresholds", []))[:15]

            sub_config["run_name"] = f"{model_type}_{mode}_{thresholds}"
            yield sub_config


def main():
    print("Starting pipeline...")

    # Step 1: Preprocess data (done once)
    preprocessed_df = preprocess_pipeline()
    preprocessed_df = (
        preprocessed_df[preprocessed_df["season_year"].isin(["2016-17", "2017-18"])]
        .sort_values(["game_date"])
        .reset_index(drop=True)
    )
    print("Preprocessing completed successfully!")

    # Step 2: Enrich data (done once)
    enriched_df = enrich_pipeline(preprocessed_df)
    print("Enrichment completed successfully!")

    # Load contests data (done once)
    contests_df = pd.read_csv("data/contests_data/fanduel_nba_contests.csv")

    # Prepare a list to store results across runs
    all_runs = []

    # Step 3: Iterate through all configurations
    for cfg in iter_configs():
        model_type = cfg["model_type"].lower()
        run_name = cfg["run_name"]

        print(f"\n=== Running {model_type} configuration: {run_name} ===")

        # Get predictions for this configuration
        start_time = time.time()
        predictions_df = get_predictions_df(cfg, enriched_df)

        # Skip if no predictions
        if not isinstance(predictions_df, pd.DataFrame) or predictions_df.empty:
            print(f"[WARN] No predictions for {run_name}. Skipping.")
            continue

        # Generate lineup and evaluate results
        lineup_df = get_lineup(predictions_df)
        res_dict, df_percentiles = evaluate_results(
            prediction_df=predictions_df,
            lineup_df=lineup_df,
            contests_df=contests_df,
        )

        # Add configuration parameters to results
        for key, value in cfg.items():
            if key not in ["model_type", "run_name"]:
                res_dict[f"cfg_{key}"] = (
                    str(value) if isinstance(value, (list, dict)) else value
                )

        # Add runtime information
        res_dict["runtime_sec"] = round(time.time() - start_time, 2)

        # Save percentiles data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_percentiles_filename = f"output_csv/percentiles_{model_type}_{timestamp}.csv"
        df_percentiles.to_csv(df_percentiles_filename, index=False)

        # Store results
        all_runs.append(res_dict)

    # Create and save master results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    df_master = pd.DataFrame(all_runs)
    out_path = f"output_csv/master_results_{timestamp}.csv"
    df_master.to_csv(out_path, index=False)
    print(f"\nAll runs complete. Master results saved to {out_path}.")


if __name__ == "__main__":
    main()
