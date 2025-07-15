import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import DMatrix
from collections import defaultdict
from config.dfs_categories import same_game_cols
import os
import random
from math import sqrt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import time

# Get the absolute path to the project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
df = pd.read_csv(os.path.join(project_root, 'enriched_df.csv'))
df = df[df['season_year'] == '2017-18']
df['game_date'] = pd.to_datetime(df['game_date'])
df["week"] = df["game_date"].dt.strftime("%Y_%U")

# Create timestamp for results directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = os.path.join("results", f"pts_{timestamp}")
os.makedirs(results_dir, exist_ok=True)

def build_points_feature_buckets(columns: pd.Series) -> dict:
    """
    Given df.columns, return categorized feature buckets relevant for predicting points.
    Categorical features are explicitly separated.
    """
    features = columns.tolist()

    # Define known categorical features
    known_categorical_features = {
        "starter", "venue", "rest_category", "salary_bucket", "position_salary", "pos-draftkings"
    }

    # Initialize buckets
    buckets = {
        "categorical": [],
        "volume_efficiency": [],
        "shooting_mix": [],
        "role_context": [],
        "ratings": [],
        "interactions": [],
        "running_stats": [],
        "last_season_stats": [],
        "opponent_context": [],
        "hot_streaks": [],
    }

    for feat in features:
        lower = feat.lower()

        if feat in known_categorical_features:
            buckets["categorical"].append(feat)
            continue

        if any(x in lower for x in ["fgm", "fga", "ft", "ts_pct", "efg", "fg_pct", "poss", "usg", "pie", "minutes", "plus_minus"]):
            buckets["volume_efficiency"].append(feat)
        if "pct_" in lower or "3pm" in lower or "2pm" in lower or "uast" in lower:
            buckets["shooting_mix"].append(feat)
        if any(x in lower for x in ["starter", "venue", "salary", "rest", "position_salary", "fp_", "nba_fantasy_pts"]):
            buckets["role_context"].append(feat)
        if any(x in lower for x in ["off_rating", "def_rating", "net_rating", "pace", "usage_rate", "ast_pct", "ast_to", "ast_ratio", "tov_pct"]):
            buckets["ratings"].append(feat)
        if "_x_" in lower:
            buckets["interactions"].append(feat)
        if "running_season" in lower:
            buckets["running_stats"].append(feat)
        if "last_season" in lower:
            buckets["last_season_stats"].append(feat)
        if "opp_" in lower:
            buckets["opponent_context"].append(feat)
        if "hot_streak" in lower:
            buckets["hot_streaks"].append(feat)

    return buckets

def remove_same_game_cols_from_buckets(buckets, same_game_cols):
    cleaned_buckets = {}
    for k, features in buckets.items():
        cleaned_buckets[k] = [f for f in features if f not in same_game_cols]
    return cleaned_buckets

def get_cleaned_numeric_buckets(buckets: dict, exclude: List[str] = ["categorical"]) -> dict:
    """Remove categorical bucket and return only numeric ones."""
    return {k: v for k, v in buckets.items() if k not in exclude}

def sample_feature_subset(
    feature_pool: Dict[str, List[str]],
    n_total: int,
    must_have: List[str] = [],
    seed: int = None,
) -> List[str]:
    """
    Sample a feature subset from the full feature pool.
    """
    if seed is not None:
        random.seed(seed)

    # Flatten the full pool and remove duplicates
    all_features = sorted(set(
        feat for bucket in feature_pool.values() for feat in bucket
    ))

    # Ensure all must_have features exist
    for mh in must_have:
        if mh not in all_features:
            raise ValueError(f"Must-have feature '{mh}' not found in feature pool.")

    # Remove must_have from pool to avoid duplication
    remaining_pool = list(set(all_features) - set(must_have))

    if len(remaining_pool) < (n_total - len(must_have)):
        raise ValueError("Not enough remaining features to sample from.")

    # Sample remaining features randomly
    sampled = random.sample(remaining_pool, n_total - len(must_have))

    return sorted(must_have + sampled)

def add_advanced_transformations(df):
    """Add advanced feature transformations to improve model performance."""
    df = df.copy()
    
    # Log transform for heavily skewed features
    skewed_features = ['minutes_played', 'pts', 'reb', 'ast', 'salary-fanduel']
    for col in skewed_features:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])
    
    # Polynomial features for key metrics
    key_features = ['minutes_played_rolling_10_day_avg', 'usg_pct_x_rolling_10_day_avg']
    for col in key_features:
        if col in df.columns:
            df[f'{col}_squared'] = df[col] ** 2
            
    # Moving averages with different windows - with shift(1) to prevent leakage
    windows = [5, 10, 15, 20]
    for window in windows:
        df[f'pts_ma_{window}'] = df.groupby('player_name')['pts'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            
    # Trend indicators - with shift(1) to prevent leakage
    df['pts_trend'] = df.groupby('player_name')['pts'].transform(
        lambda x: x.shift(1).rolling(window=5).mean() - x.shift(1).rolling(window=15).mean())
        
    return df

def run_rolling_xgb_training(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling XGBoost training and evaluation for one experimental run."""
    df = df.copy()
    
    print("Preprocessing data...")
    # Add advanced feature transformations
    df = add_advanced_transformations(df)
    
    # Filter for training, but keep all data for predictions
    df_for_training = df[df["salary_quantile"] >= config["salary_quantile_threshold"]].copy()
    
    target = config["target_col"]
    cat_cols = config["categorical_features"]
    num_cols = config["numeric_features"]
    
    # Validate features exist in dataframe
    cat_cols = [col for col in cat_cols if col in df.columns]
    num_cols = [col for col in num_cols if col in df.columns]
    
    if not num_cols:
        raise ValueError("No valid numeric features found in the dataframe")
    
    all_feats = list(set(cat_cols + num_cols))
    
    print(f"Using {len(num_cols)} numeric features and {len(cat_cols)} categorical features")
    
    # Cast categorical columns
    for col in cat_cols:
        df[col] = df[col].astype("category")
        df_for_training[col] = df_for_training[col].astype("category")

    # Handle outliers and scale numeric features
    print("\nPreprocessing numeric features...")
    for col in tqdm(num_cols, desc="Handling outliers"):
        Q1 = df_for_training[col].quantile(0.25)
        Q3 = df_for_training[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_for_training[col] = df_for_training[col].clip(lower=lower_bound, upper=upper_bound)
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Scale features
    print("Scaling numeric features...")
    scaler = StandardScaler()
    df_for_training[num_cols] = scaler.fit_transform(df_for_training[num_cols])
    df[num_cols] = scaler.transform(df[num_cols])

    weeks = sorted(df["week"].unique())
    lookback = config["train_weeks_lookback"]
    total_weeks = len(weeks) - lookback

    all_predictions = []
    all_metrics = []

    print(f"\nTraining and evaluating on {total_weeks} time windows...")
    for test_idx in tqdm(range(lookback, len(weeks)), desc="Processing time windows"):
        train_weeks = weeks[test_idx - lookback:test_idx]
        test_week = weeks[test_idx]

        df_train = df_for_training[df_for_training["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == test_week].copy()

        if df_train.empty or df_test.empty:
            continue

        X_train = df_train[all_feats]
        y_train = df_train[target]
        X_test = df_test[all_feats]
        y_test = df_test[target]

        # Add sample weights based on recency
        sample_weights = np.linspace(0.5, 1.0, len(df_train))
        
        dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        
        # Train with early stopping
        model = xgb.train(
            params=config["xgb_params"],
            dtrain=dtrain,
            num_boost_round=config["num_boost_round"],
            early_stopping_rounds=50,
            evals=[(dtrain, 'train'), (dtest, 'eval')],
            verbose_eval=False
        )

        y_pred = model.predict(dtest)
        
        # Calculate metrics only for high-salary players in test set
        high_salary_mask = df_test["salary_quantile"] >= config.get("metrics_salary_quantile", 0.9)
        y_test_high_salary = y_test[high_salary_mask]
        y_pred_high_salary = y_pred[high_salary_mask]
        
        rmse = sqrt(mean_squared_error(y_test_high_salary, y_pred_high_salary))
        mae = mean_absolute_error(y_test_high_salary, y_pred_high_salary)
        r2 = r2_score(y_test_high_salary, y_pred_high_salary)

        df_pred = df_test[["player_name", "team_abbreviation", "game_date", "salary_quantile"]].copy()
        df_pred[f"{target}"] = y_test
        df_pred[f"{target}_pred"] = y_pred
        df_pred["week"] = test_week
        df_pred["is_high_salary"] = high_salary_mask
        all_predictions.append(df_pred)

        all_metrics.append({
            "test_week": test_week,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_high_salary_players": sum(high_salary_mask)
        })

    pred_df = pd.concat(all_predictions, ignore_index=True)
    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df, pred_df

def run_single_feature_selection_trial(
    df: pd.DataFrame,
    feature_pool: Dict[str, List[str]],
    must_have: List[str],
    n_features: int,
    config_template: Dict,
    seed: int = None,
    importance_type: str = "gain"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    Run a single feature selection trial using a sampled subset of features.
    """
    # 1. Sample features
    sampled_features = sample_feature_subset(
        feature_pool=feature_pool,
        n_total=n_features,
        must_have=must_have,
        seed=seed,
    )

    # 2. Update config
    config = config_template.copy()  # Simple dict copy is sufficient here
    config["numeric_features"] = sampled_features
    config["categorical_features"] = config_template.get("categorical_features", [])
    config["seed"] = seed

    # 3. Run training
    metrics_df, pred_df = run_rolling_xgb_training(df, config)

    # 4. Fit model once more to full data to extract importance
    df_full = df[df["salary_quantile"] >= config["salary_quantile_threshold"]]
    df_full = df_full.dropna(subset=sampled_features + [config["target_col"]])
    X_full = df_full[sampled_features]
    y_full = df_full[config["target_col"]]

    # Cast categorical features for the final model
    for col in config["categorical_features"]:
        if col in X_full.columns:
            X_full[col] = X_full[col].astype("category")

    dmatrix_full = DMatrix(X_full, label=y_full, enable_categorical=True)
    final_model = xgb.train(
        params=config["xgb_params"],
        dtrain=dmatrix_full,
        num_boost_round=config["num_boost_round"]
    )

    importance_dict = final_model.get_score(importance_type=importance_type)
    return metrics_df, pred_df, importance_dict

def create_importance_tracker():
    return defaultdict(float)

def run_exploration_phase(
    df,
    feature_pool,
    config_template,
    must_have,
    n_features=80,
    n_trials=30,
    seed_start=42
):
    importance_tracker = create_importance_tracker()
    all_metrics, all_preds = [], []
    
    print("\n=== Starting Exploration Phase ===")
    print(f"Running {n_trials} trials with {n_features} features each")
    print(f"Must-have features: {must_have}")
    
    # Create directory for exploration results
    exploration_dir = os.path.join(results_dir, "exploration")
    os.makedirs(exploration_dir, exist_ok=True)
    
    for i in range(n_trials):
        seed = seed_start + i
        print(f"\nTrial {i+1}/{n_trials} (seed={seed})")
        
        metrics_df, pred_df, imp = run_single_feature_selection_trial(
            df, feature_pool, must_have, n_features, config_template, seed
        )
        
        # Log metrics for this trial (only numeric columns)
        mean_metrics = metrics_df[['rmse', 'mae', 'r2']].mean()
        print(f"Average metrics for trial {i+1}:")
        print(f"  RMSE: {mean_metrics['rmse']:.4f}")
        print(f"  MAE: {mean_metrics['mae']:.4f}")
        print(f"  R²: {mean_metrics['r2']:.4f}")
        
        # Save trial results
        metrics_df.to_csv(os.path.join(exploration_dir, f"trial_{i+1}_metrics.csv"), index=False)
        pred_df.to_csv(os.path.join(exploration_dir, f"trial_{i+1}_predictions.csv"), index=False)
        
        # Update importance tracker
        for feat, val in imp.items():
            importance_tracker[feat] += val
        
        all_metrics.append(metrics_df)
        all_preds.append(pred_df)
    
    # Save aggregated results
    final_metrics = pd.concat(all_metrics)
    final_preds = pd.concat(all_preds)
    
    # Calculate and display overall metrics
    overall_metrics = final_metrics[['rmse', 'mae', 'r2']].mean()
    print("\n=== Overall Metrics ===")
    print(f"Average RMSE: {overall_metrics['rmse']:.4f}")
    print(f"Average MAE: {overall_metrics['mae']:.4f}")
    print(f"Average R²: {overall_metrics['r2']:.4f}")
    
    final_metrics.to_csv(os.path.join(exploration_dir, "all_metrics.csv"), index=False)
    final_preds.to_csv(os.path.join(exploration_dir, "all_predictions.csv"), index=False)
    
    # Save and display feature importance summary
    importance_df = pd.DataFrame({
        'feature': list(importance_tracker.keys()),
        'importance': list(importance_tracker.values())
    })
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df.to_csv(os.path.join(exploration_dir, "feature_importance.csv"), index=False)
    
    print("\n=== Exploration Phase Complete ===")
    print("\nTop 10 Most Important Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    return final_metrics, final_preds, importance_tracker

def run_exploitation_phase(
    df: pd.DataFrame,
    importance_tracker: Dict[str, float],
    config_template: Dict,
    must_have: List[str],
    top_k: int = 100,
    features_per_trial: int = 50,
    n_trials: int = 25,
    seed_start: int = 1000,
    importance_type: str = "gain"
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float], Dict[str, int]]:
    """
    Run exploitation phase using top features discovered during exploration.
    """
    print("\n=== Starting Exploitation Phase ===")
    print(f"Selecting top {top_k} features")
    print(f"Using {features_per_trial} features per trial")
    print(f"Running {n_trials} trials")
    
    # Create directory for exploitation results
    exploitation_dir = os.path.join(results_dir, "exploitation")
    os.makedirs(exploitation_dir, exist_ok=True)
    
    # 1. Rank and select top features
    sorted_features = sorted(
        importance_tracker.items(),
        key=lambda x: x[1],
        reverse=True
    )
    top_k_features = [f for f, _ in sorted_features[:top_k]]
    
    # Save top features list
    pd.DataFrame({
        'feature': [f for f, _ in sorted_features[:top_k]],
        'importance': [v for _, v in sorted_features[:top_k]]
    }).to_csv(os.path.join(exploitation_dir, "top_k_features.csv"), index=False)
    
    # Remove must_have features from top_k to avoid duplication
    top_k_features = [f for f in top_k_features if f not in must_have]
    
    print(f"\nSelected {len(top_k_features)} features (excluding {len(must_have)} must-have features)")
    
    # 2. Initialize trackers
    new_importance_tracker = defaultdict(float)
    coverage_counter = defaultdict(int)
    all_metrics, all_preds = [], []
    
    # 3. Run trials
    for i in range(n_trials):
        seed = seed_start + i
        print(f"\nTrial {i+1}/{n_trials} (seed={seed})")
        
        # Sample features for this trial
        n_to_sample = features_per_trial - len(must_have)
        if n_to_sample > len(top_k_features):
            raise ValueError(f"Not enough features to sample {n_to_sample} from top {top_k}")
            
        trial_features = random.sample(top_k_features, n_to_sample)
        all_trial_features = sorted(must_have + trial_features)
        
        # Save trial feature set
        pd.DataFrame({'feature': all_trial_features}).to_csv(
            os.path.join(exploitation_dir, f"trial_{i+1}_features.csv"), index=False
        )
        
        # Update coverage counter
        for feat in trial_features:
            coverage_counter[feat] += 1
            
        # Create trial config
        trial_config = config_template.copy()
        trial_config["numeric_features"] = all_trial_features
        trial_config["seed"] = seed
        
        # Run trial
        metrics_df, pred_df, imp = run_single_feature_selection_trial(
            df=df,
            feature_pool={"selected": all_trial_features},
            must_have=must_have,
            n_features=len(all_trial_features),
            config_template=trial_config,
            seed=seed,
            importance_type=importance_type
        )
        
        # Log metrics for this trial (only numeric columns)
        mean_metrics = metrics_df[['rmse', 'mae', 'r2']].mean()
        print(f"Average metrics for trial {i+1}:")
        print(f"  RMSE: {mean_metrics['rmse']:.4f}")
        print(f"  MAE: {mean_metrics['mae']:.4f}")
        print(f"  R²: {mean_metrics['r2']:.4f}")
        
        # Save trial results
        metrics_df.to_csv(os.path.join(exploitation_dir, f"trial_{i+1}_metrics.csv"), index=False)
        pred_df.to_csv(os.path.join(exploitation_dir, f"trial_{i+1}_predictions.csv"), index=False)
        
        # Update trackers
        for feat, score in imp.items():
            new_importance_tracker[feat] += score
        all_metrics.append(metrics_df)
        all_preds.append(pred_df)
        
    # 4. Aggregate results
    final_metrics = pd.concat(all_metrics, ignore_index=True)
    final_preds = pd.concat(all_preds, ignore_index=True)
    
    # Calculate and display overall metrics
    overall_metrics = final_metrics[['rmse', 'mae', 'r2']].mean()
    print("\n=== Overall Metrics ===")
    print(f"Average RMSE: {overall_metrics['rmse']:.4f}")
    print(f"Average MAE: {overall_metrics['mae']:.4f}")
    print(f"Average R²: {overall_metrics['r2']:.4f}")
    
    # Save final results
    final_metrics.to_csv(os.path.join(exploitation_dir, "all_metrics.csv"), index=False)
    final_preds.to_csv(os.path.join(exploitation_dir, "all_predictions.csv"), index=False)
    
    # 5. Normalize importance scores
    if new_importance_tracker:
        total = sum(new_importance_tracker.values())
        new_importance_tracker = {k: v/total for k, v in new_importance_tracker.items()}
    
    # Save feature coverage and importance
    coverage_df = pd.DataFrame({
        'feature': list(coverage_counter.keys()),
        'times_used': list(coverage_counter.values())
    }).sort_values('times_used', ascending=False)
    
    importance_df = pd.DataFrame({
        'feature': list(new_importance_tracker.keys()),
        'importance': list(new_importance_tracker.values())
    }).sort_values('importance', ascending=False)
    
    coverage_df.to_csv(os.path.join(exploitation_dir, "feature_coverage.csv"), index=False)
    importance_df.to_csv(os.path.join(exploitation_dir, "feature_importance.csv"), index=False)
    
    print("\n=== Exploitation Phase Complete ===")
    print("\nTop 10 Most Used Features:")
    for _, row in coverage_df.head(10).iterrows():
        print(f"{row['feature']}: used in {row['times_used']} trials")
    
    print("\nTop 10 Most Important Features (Exploitation Phase):")
    for _, row in importance_df.head(10).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    return final_metrics, final_preds, new_importance_tracker, coverage_counter

def generate_xgb_configs():
    """Generate different XGBoost configurations to experiment with."""
    base_config = {
        "objective": "reg:squarederror",
        "verbosity": 0,
        "tree_method": "hist",
    }
    
    # Different configuration sets to try
    configs = []
    
    # Configuration 1: Current optimized setup
    configs.append({
        **base_config,
        "max_depth": 6,
        "eta": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "min_child_weight": 3,
        "gamma": 0.1,
        "alpha": 0.1,
        "lambda": 1.0,
        "name": "current_optimized"
    })
    
    # Configuration 2: Deeper trees with stronger regularization
    configs.append({
        **base_config,
        "max_depth": 8,
        "eta": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.2,
        "alpha": 0.2,
        "lambda": 2.0,
        "name": "deep_regularized"
    })
    
    # Configuration 3: Shallow trees with high learning rate
    configs.append({
        **base_config,
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "gamma": 0.05,
        "alpha": 0.05,
        "lambda": 0.5,
        "name": "shallow_fast"
    })
    
    # Configuration 4: Balanced approach with moderate depth
    configs.append({
        **base_config,
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 4,
        "gamma": 0.15,
        "alpha": 0.15,
        "lambda": 1.5,
        "name": "balanced"
    })
    
    # Configuration 5: Heavy regularization with moderate depth
    configs.append({
        **base_config,
        "max_depth": 5,
        "eta": 0.04,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "min_child_weight": 6,
        "gamma": 0.3,
        "alpha": 0.3,
        "lambda": 3.0,
        "name": "heavy_reg"
    })
    
    return configs

def run_xgb_experiment(df: pd.DataFrame, base_config: Dict, xgb_config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run a single XGBoost experiment with a specific configuration."""
    experiment_config = base_config.copy()
    experiment_config["xgb_params"] = xgb_config.copy()
    config_name = xgb_config.pop("name")  # Remove name from params but keep for reference
    
    print(f"\nRunning experiment with configuration: {config_name}")
    print("Parameters:")
    for k, v in xgb_config.items():
        print(f"  {k}: {v}")
    
    start_time = time.time()
    metrics_df, pred_df = run_rolling_xgb_training(df, experiment_config)
    end_time = time.time()
    
    # Calculate aggregate metrics
    agg_metrics = {
        "rmse_mean": metrics_df["rmse"].mean(),
        "rmse_std": metrics_df["rmse"].std(),
        "mae_mean": metrics_df["mae"].mean(),
        "mae_std": metrics_df["mae"].std(),
        "r2_mean": metrics_df["r2"].mean(),
        "r2_std": metrics_df["r2"].std(),
    }
    
    print(f"\nResults (completed in {(end_time - start_time)/60:.2f} minutes):")
    print(f"  Average RMSE: {agg_metrics['rmse_mean']:.4f} ± {agg_metrics['rmse_std']:.4f}")
    print(f"  Average MAE: {agg_metrics['mae_mean']:.4f} ± {agg_metrics['mae_std']:.4f}")
    print(f"  Average R²: {agg_metrics['r2_mean']:.4f} ± {agg_metrics['r2_std']:.4f}")
    
    return metrics_df, pred_df, config_name

def run_all_xgb_experiments(df: pd.DataFrame, base_config: Dict) -> Dict:
    """Run all XGBoost experiments and compare results."""
    results = {}
    xgb_configs = generate_xgb_configs()
    
    print(f"\nStarting experiments with {len(xgb_configs)} configurations")
    total_start_time = time.time()
    
    # Create directory for XGBoost experiments
    xgb_dir = os.path.join(results_dir, "xgb_experiments")
    os.makedirs(xgb_dir, exist_ok=True)
    
    for i, config in enumerate(xgb_configs, 1):
        print(f"\nExperiment {i}/{len(xgb_configs)} - Configuration: {config['name']}")
        start_time = time.time()
        
        metrics_df, pred_df, config_name = run_xgb_experiment(df, base_config, config)
        
        end_time = time.time()
        time_taken = (end_time - start_time) / 60
        time_remaining = time_taken * (len(xgb_configs) - i)
        
        print(f"Time taken: {time_taken:.2f} minutes")
        print(f"Estimated time remaining: {time_remaining:.2f} minutes")
        
        # Store results
        results[config_name] = {
            "metrics_df": metrics_df,
            "pred_df": pred_df,
            "config": config,
            "runtime_minutes": time_taken,
            "summary": {
                "rmse_mean": metrics_df["rmse"].mean(),
                "rmse_std": metrics_df["rmse"].std(),
                "mae_mean": metrics_df["mae"].mean(),
                "mae_std": metrics_df["mae"].std(),
                "r2_mean": metrics_df["r2"].mean(),
                "r2_std": metrics_df["r2"].std(),
            }
        }
        
        # Save results to files
        config_dir = os.path.join(xgb_dir, config_name)
        os.makedirs(config_dir, exist_ok=True)
        metrics_df.to_csv(os.path.join(config_dir, "metrics.csv"), index=False)
        pred_df.to_csv(os.path.join(config_dir, "predictions.csv"), index=False)
    
    total_time = (time.time() - total_start_time) / 60
    print(f"\nAll experiments completed in {total_time:.2f} minutes")
    
    # Create summary DataFrame
    summary_data = []
    for config_name, result in results.items():
        summary_data.append({
            "config_name": config_name,
            "runtime_minutes": result["runtime_minutes"],
            **result["summary"]
        })
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_df.to_csv(os.path.join(xgb_dir, "summary.csv"), index=False)
    
    # Print comparison
    print("\n=== XGBoost Experiments Summary ===")
    print("\nAverage Metrics by Configuration:")
    print(summary_df.to_string(index=False))
    
    # Find best configuration
    best_config = summary_df.loc[summary_df["rmse_mean"].idxmin(), "config_name"]
    print(f"\nBest configuration (by RMSE): {best_config}")
    print("Configuration details:")
    for k, v in results[best_config]["config"].items():
        if k != "name":
            print(f"  {k}: {v}")
    
    return results

# Initialize feature buckets and configuration
feature_buckets = build_points_feature_buckets(df.columns)

# Clean feature buckets to remove same-game stats
feature_buckets = remove_same_game_cols_from_buckets(feature_buckets, same_game_cols)
print("\n=== Feature Buckets After Removing Same-Game Stats ===")
for bucket, features in feature_buckets.items():
    print(f"\n{bucket}:")
    print(", ".join(features))

feature_pool = get_cleaned_numeric_buckets(feature_buckets)
categorical_features = feature_buckets["categorical"]

# Cast categorical features globally
for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].astype("category")

assert "week" in df.columns and "salary_quantile" in df.columns

# Create base configuration templates with different XGBoost parameters
base_configs = [
    {
        "name": "current_optimized",
        "target_col": "pts",
        "categorical_features": categorical_features,
        "numeric_features": [],  # Will be set by feature selection
        "salary_quantile_threshold": 0.75,
        "metrics_salary_quantile": 0.9,
        "train_weeks_lookback": 15,
        "num_boost_round": 500,
        "xgb_params": {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.03,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "min_child_weight": 3,
            "gamma": 0.1,
            "alpha": 0.1,
            "lambda": 1.0,
            "tree_method": "hist",
            "verbosity": 0
        }
    },
    {
        "name": "deep_regularized",
        "target_col": "pts",
        "categorical_features": categorical_features,
        "numeric_features": [],  # Will be set by feature selection
        "salary_quantile_threshold": 0.75,
        "metrics_salary_quantile": 0.9,
        "train_weeks_lookback": 15,
        "num_boost_round": 500,
        "xgb_params": {
            "objective": "reg:squarederror",
            "max_depth": 8,
            "eta": 0.02,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.2,
            "alpha": 0.2,
            "lambda": 2.0,
            "tree_method": "hist",
            "verbosity": 0
        }
    },
    {
        "name": "shallow_fast",
        "target_col": "pts",
        "categorical_features": categorical_features,
        "numeric_features": [],  # Will be set by feature selection
        "salary_quantile_threshold": 0.75,
        "metrics_salary_quantile": 0.9,
        "train_weeks_lookback": 15,
        "num_boost_round": 500,
        "xgb_params": {
            "objective": "reg:squarederror",
            "max_depth": 4,
            "eta": 0.1,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "min_child_weight": 2,
            "gamma": 0.05,
            "alpha": 0.05,
            "lambda": 0.5,
            "tree_method": "hist",
            "verbosity": 0
        }
    }
]

# Run exploration and exploitation phases for each configuration
all_results = {}
for config in base_configs:
    print(f"\n{'='*50}")
    print(f"Running experiments with configuration: {config['name']}")
    print(f"{'='*50}")
    
    # Run exploration phase
    metrics_df, preds_df, importance_tracker = run_exploration_phase(
        df=df,
        feature_pool=feature_pool,
        config_template=config,
        must_have=["minutes_played_rolling_10_day_avg", "usg_pct_x_rolling_10_day_avg"],
        n_features=80,
        n_trials=30,
        seed_start=42
    )
    
    # Run exploitation phase
    exploitation_metrics, exploitation_preds, new_importance, coverage = run_exploitation_phase(
        df=df,
        importance_tracker=importance_tracker,
        config_template=config,
        must_have=["minutes_played_rolling_10_day_avg", "usg_pct_x_rolling_10_day_avg"],
        top_k=100,
        features_per_trial=50,
        n_trials=25,
        seed_start=1000
    )
    
    # Store results
    all_results[config['name']] = {
        'exploration': {
            'metrics': metrics_df,
            'predictions': preds_df,
            'importance': importance_tracker
        },
        'exploitation': {
            'metrics': exploitation_metrics,
            'predictions': exploitation_preds,
            'importance': new_importance,
            'coverage': coverage
        }
    }

# Print summary of results
print("\n=== Summary of All Configurations ===")
for config_name, results in all_results.items():
    print(f"\nConfiguration: {config_name}")
    print("Exploration phase:")
    exp_metrics = results['exploration']['metrics']
    print(f"  Average RMSE: {exp_metrics['rmse'].mean():.4f} ± {exp_metrics['rmse'].std():.4f}")
    print(f"  Average R²: {exp_metrics['r2'].mean():.4f} ± {exp_metrics['r2'].std():.4f}")
    
    print("Exploitation phase:")
    exp_metrics = results['exploitation']['metrics']
    print(f"  Average RMSE: {exp_metrics['rmse'].mean():.4f} ± {exp_metrics['rmse'].std():.4f}")
    print(f"  Average R²: {exp_metrics['r2'].mean():.4f} ± {exp_metrics['r2'].std():.4f}")

# Save all results
final_comparison_dir = os.path.join(results_dir, "final_comparison")
os.makedirs(final_comparison_dir, exist_ok=True)
for config_name, results in all_results.items():
    config_dir = os.path.join(final_comparison_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    
    # Save exploration results
    results['exploration']['metrics'].to_csv(os.path.join(config_dir, "exploration_metrics.csv"), index=False)
    results['exploration']['predictions'].to_csv(os.path.join(config_dir, "exploration_predictions.csv"), index=False)
    pd.DataFrame(results['exploration']['importance'].items(), columns=['feature', 'importance']).to_csv(
        os.path.join(config_dir, "exploration_importance.csv"), index=False
    )
    
    # Save exploitation results
    results['exploitation']['metrics'].to_csv(os.path.join(config_dir, "exploitation_metrics.csv"), index=False)
    results['exploitation']['predictions'].to_csv(os.path.join(config_dir, "exploitation_predictions.csv"), index=False)
    pd.DataFrame(results['exploitation']['importance'].items(), columns=['feature', 'importance']).to_csv(
        os.path.join(config_dir, "exploitation_importance.csv"), index=False
    )
    pd.DataFrame(results['exploitation']['coverage'].items(), columns=['feature', 'times_used']).to_csv(
        os.path.join(config_dir, "exploitation_coverage.csv"), index=False
    )
