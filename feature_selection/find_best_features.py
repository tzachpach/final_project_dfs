import pandas as pd
import numpy as np
from datetime import datetime
import xgboost as xgb
from typing import Dict, List, Tuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from collections import defaultdict
import os
from math import sqrt
from tqdm import tqdm
import time
from config.dfs_categories import same_game_cols

def build_feature_buckets_for_stat(columns: pd.Series, stat_name: str) -> Dict[str, List[str]]:
    """Build optimized feature buckets for specific statistics."""
    features = columns.tolist()
    
    # Remove same-game columns to prevent leakage
    features = [f for f in features if f not in same_game_cols]
    
    # Define categorical features
    categorical_features = {"starter", "venue", "rest_category", "salary_bucket", "position_salary", "pos-draftkings"}
    
    buckets = {
        "categorical": [f for f in features if f in categorical_features],
        "rolling_averages": [f for f in features if "rolling" in f.lower() and "avg" in f.lower()],
        "percentages": [f for f in features if "_pct" in f.lower()],
        "efficiency": [f for f in features if any(x in f.lower() for x in ["ts_", "efg", "per", "pie"])],
        "salary_context": [f for f in features if "salary" in f.lower()],
        "usage_metrics": [f for f in features if any(x in f.lower() for x in ["usg", "usage", "minutes"])],
        "ratings": [f for f in features if "rating" in f.lower()],
        "pace_context": [f for f in features if any(x in f.lower() for x in ["pace", "poss"])],
        "opponent": [f for f in features if "opp_" in f.lower()],
        "interactions": [f for f in features if "_x_" in f.lower()],
        "transformations": [f for f in features if any(x in f.lower() for x in ["log", "squared", "trend"])],
    }
    
    # Stat-specific buckets
    if stat_name == "pts":
        buckets["scoring_specific"] = [f for f in features if any(x in f.lower() for x in ["pts", "fg", "3p", "ft", "shot"])]
    elif stat_name == "reb":
        buckets["rebounding_specific"] = [f for f in features if any(x in f.lower() for x in ["reb", "orb", "drb"])]
    elif stat_name == "ast":
        buckets["playmaking_specific"] = [f for f in features if any(x in f.lower() for x in ["ast", "assist", "to", "turnover"])]
    elif stat_name == "tov":
        buckets["turnover_specific"] = [f for f in features if any(x in f.lower() for x in ["tov", "turnover", "to", "ast"])]
    elif stat_name == "stl":
        buckets["defensive_specific"] = [f for f in features if any(x in f.lower() for x in ["stl", "steal", "def"])]
    elif stat_name == "blk":
        buckets["blocking_specific"] = [f for f in features if any(x in f.lower() for x in ["blk", "block"])]
    
    # Remove empty buckets
    buckets = {k: v for k, v in buckets.items() if v}
    
    # Ensure we only keep features that actually exist in the dataframe
    valid_cols = set(columns)
    buckets = {k: [f for f in v if f in valid_cols] for k, v in buckets.items()}
    
    return buckets

def run_simplified_xgb_training(df: pd.DataFrame, config: Dict) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Simplified XGBoost training that returns metrics and importance."""
    df = df.copy()
    
    # Filter for training data
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
    
    # Cast categorical columns
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
            df_for_training[col] = df_for_training[col].astype("category")

    # Simple preprocessing - handle missing values
    for col in num_cols:
        df_for_training[col] = df_for_training[col].fillna(df_for_training[col].median())
        df[col] = df[col].fillna(df[col].median())

    weeks = sorted(df["week"].unique())
    lookback = config["train_weeks_lookback"]
    
    all_metrics = []
    importance_accumulator = defaultdict(float)

    # --- moving rolling window: train on 'lookback' weeks, test on the next week ---
    # The first test week is index = lookback (0-based), continue until the end.
    test_weeks = weeks[lookback:]
    if not test_weeks:
        raise ValueError("Not enough weeks in dataset for the chosen lookback window")
    
    print("Numeric features being used:", num_cols)
    # print("first 20 week labels:", weeks[:20])
    # print("lookback window:", lookback)
    # print("first train set  :", weeks[0:lookback])
    # print("first test week  :", test_weeks[0])
    
    for test_week in test_weeks:
        test_idx = weeks.index(test_week)
        train_weeks = weeks[max(0, test_idx - lookback):test_idx]
        
        df_train = df_for_training[df_for_training["week"].isin(train_weeks)].copy()
        df_test = df[df["week"] == test_week].copy()

        if df_train.empty or df_test.empty or len(df_train) < 10:
            continue

        X_train = df_train[all_feats]
        y_train = df_train[target]
        X_test = df_test[all_feats]
        y_test = df_test[target]
        
        dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
        dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
        
        # Train model
        model = xgb.train(
            params=config["xgb_params"],
            dtrain=dtrain,
            num_boost_round=config["num_boost_round"],
            verbose_eval=False
        )

        y_pred = model.predict(dtest)
        
        # Calculate metrics for high-salary players
        high_salary_mask = df_test["salary_quantile"] >= config.get("metrics_salary_quantile", 0.9)
        if sum(high_salary_mask) > 0:
            y_test_high_salary = y_test[high_salary_mask]
            y_pred_high_salary = y_pred[high_salary_mask]
            
            rmse = sqrt(mean_squared_error(y_test_high_salary, y_pred_high_salary))
            mae = mean_absolute_error(y_test_high_salary, y_pred_high_salary)
            r2 = r2_score(y_test_high_salary, y_pred_high_salary)

            all_metrics.append({
                "test_week": test_week,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "n_high_salary_players": sum(high_salary_mask)
            })
            
            # Accumulate importance
            importance = model.get_score(importance_type='gain')
            for feat, score in importance.items():
                importance_accumulator[feat] += score

    if not all_metrics:
        raise ValueError("No valid test periods found")
        
    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df, dict(importance_accumulator)

def get_must_have_features(stat_name: str) -> List[str]:
    """Get essential features that must be included for each stat."""
    must_have_map = {
        "pts": ["minutes_played_rolling_10_day_avg", "usg_pct_x_rolling_10_day_avg"],
        "reb": ["minutes_played_rolling_10_day_avg", "reb_pct_rolling_10_day_avg"],
        "ast": ["minutes_played_rolling_10_day_avg", "ast_pct_rolling_10_day_avg"],
        "tov": ["minutes_played_rolling_10_day_avg", "usg_pct_x_rolling_10_day_avg"],
        "stl": ["minutes_played_rolling_10_day_avg", "def_rating_rolling_10_day_avg"],
        "blk": ["minutes_played_rolling_10_day_avg", "dreb_pct_rolling_10_day_avg"]
    }
    return must_have_map.get(stat_name, ["minutes_played_rolling_10_day_avg"])

def run_single_trial(df: pd.DataFrame, features: List[str], config: Dict) -> Dict:
    """Run a single experiment trial and return metrics."""
    trial_config = config.copy()
    trial_config["numeric_features"] = features
    
    try:
        # Use our own simplified training function
        metrics_df, importance = run_simplified_xgb_training(df, trial_config)
        return {
            'success': True,
            'rmse': metrics_df['rmse'].mean(),
            'mae': metrics_df['mae'].mean(),
            'r2': metrics_df['r2'].mean(),
            'rmse_std': metrics_df['rmse'].std(),
            'r2_std': metrics_df['r2'].std(),
            'importance': importance,
            'features': features
        }
    except Exception as e:
        print(f"Trial failed: {e}")
        return {'success': False, 'error': str(e)}

def find_best_features_systematic(df: pd.DataFrame, stat_name: str, 
                                 feature_buckets: Dict[str, List[str]],
                                 must_have: List[str],
                                 config: Dict,
                                 max_features: int = 100,
                                 n_trials: int = 40) -> Dict:
    """
    Systematically find the best features using a reliable methodology.
    """
    
    print(f"\n=== Finding Best Features for {stat_name.upper()} ===")
    
    # Combine all features
    all_features = []
    for bucket_features in feature_buckets.values():
        all_features.extend(bucket_features)
    all_features = sorted(set(all_features))
    
    print(f"Total available features: {len(all_features)}")
    print(f"Must-have features: {must_have}")
    
    # Phase 1: Test individual feature importance using correlation and univariate methods
    print("\nPhase 1: Individual feature scoring...")
    feature_scores = {}
    
    for feature in tqdm(all_features, desc="Scoring features"):
        if feature not in df.columns:
            continue
            
        try:
            # Calculate correlation with target
            corr = abs(df[feature].corr(df[stat_name]))
            if np.isnan(corr):
                corr = 0
            
            # Calculate mutual information (simplified)
            # Using a simple variance-based score as proxy
            feature_var = df[feature].var()
            target_var = df[stat_name].var()
            var_score = min(feature_var, target_var) / max(feature_var, target_var) if max(feature_var, target_var) > 0 else 0
            
            # Combined score
            combined_score = 0.7 * corr + 0.3 * var_score
            feature_scores[feature] = combined_score
            
        except Exception:
            feature_scores[feature] = 0
    
    # Phase 2: Progressive feature selection
    print("\nPhase 2: Progressive feature selection...")
    
    # Start with must-have features
    current_features = must_have.copy()
    available_features = [f for f in all_features if f not in must_have and feature_scores.get(f, 0) > 0]
    
    # Sort available features by score
    available_features.sort(key=lambda x: feature_scores.get(x, 0), reverse=True)
    
    best_performance = {'rmse': float('inf'), 'r2': -float('inf'), 'features': current_features.copy()}
    performance_history = []
    
    # Test adding features progressively
    step_size = 30  # Add 30 features at a time (increased for larger feature sets)
    for i in range(0, min(len(available_features), max_features - len(must_have)), step_size):
        # Add next batch of features
        new_features = available_features[i:i + step_size]
        test_features = current_features + new_features
        
        print(f"\nTesting {len(test_features)} features...")
        
        # Run multiple trials with this feature set
        trial_results = []
        for trial in range(min(3, n_trials // 15)):  # Fewer trials per step for efficiency with larger sets
            result = run_single_trial(df, test_features, config)
            if result['success']:
                trial_results.append(result)
        
        if trial_results:
            # Calculate average performance
            avg_rmse = np.mean([r['rmse'] for r in trial_results])
            avg_r2 = np.mean([r['r2'] for r in trial_results])
            std_rmse = np.std([r['rmse'] for r in trial_results])
            
            performance_history.append({
                'n_features': len(test_features),
                'rmse': avg_rmse,
                'r2': avg_r2,
                'rmse_std': std_rmse,
                'features': test_features.copy()
            })
            
            print(f"  {len(test_features)} features: RMSE={avg_rmse:.4f}±{std_rmse:.4f}, R²={avg_r2:.4f}")
            
            # Update best_performance if better
            if avg_r2 > best_performance['r2'] or (avg_r2 >= best_performance['r2'] * 0.99 and avg_rmse < best_performance['rmse']):
                best_performance = {
                    'rmse': avg_rmse,
                    'r2': avg_r2,
                    'rmse_std': std_rmse,
                    'features': test_features.copy()
                }
                current_features = test_features.copy()
                print("  New best performance – continuing exploration.")
            else:
                # Keep exploring even if no improvement to ensure broader search
                print("  No improvement this round, but continuing exploration for broader coverage.")
    
    # Phase 3: Fine-tune the best feature set with more trials
    print(f"\nPhase 3: Fine-tuning best feature set ({len(best_performance['features'])} features)...")
    
    final_results = []
    feature_importance_sum = defaultdict(float)
    
    for trial in tqdm(range(n_trials), desc="Final validation"):
        result = run_single_trial(df, best_performance['features'], config)
        if result['success']:
            final_results.append(result)
            # Accumulate feature importance
            for feat, imp in result['importance'].items():
                feature_importance_sum[feat] += imp
    
    if final_results:
        final_rmse = np.mean([r['rmse'] for r in final_results])
        final_r2 = np.mean([r['r2'] for r in final_results])
        final_rmse_std = np.std([r['rmse'] for r in final_results])
        final_r2_std = np.std([r['r2'] for r in final_results])
        
        # Normalize importance scores
        total_importance = sum(feature_importance_sum.values())
        if total_importance > 0:
            normalized_importance = {f: imp/total_importance for f, imp in feature_importance_sum.items()}
        else:
            normalized_importance = {}
        
        # Create feature ranking
        feature_ranking = sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True)
        
        results = {
            'stat': stat_name,
            'best_features': best_performance['features'],
            'n_features': len(best_performance['features']),
            'final_rmse': final_rmse,
            'final_r2': final_r2,
            'final_rmse_std': final_rmse_std,
            'final_r2_std': final_r2_std,
            'feature_importance': normalized_importance,
            'feature_ranking': feature_ranking,
            'performance_history': performance_history,
            'n_successful_trials': len(final_results)
        }
        
        print(f"\n=== RESULTS for {stat_name.upper()} ===")
        print(f"Best feature count: {results['n_features']}")
        print(f"Final RMSE: {final_rmse:.4f} ± {final_rmse_std:.4f}")
        print(f"Final R²: {final_r2:.4f} ± {final_r2_std:.4f}")
        print(f"Successful trials: {len(final_results)}/{n_trials}")
        
        print(f"\nTop 20 Features:")
        for i, (feat, imp) in enumerate(feature_ranking[:20], 1):
            print(f"{i:2d}. {feat:50s} (importance: {imp:.4f})")
        
        return results
    
    else:
        print(f"ERROR: No successful trials for {stat_name}")
        return None

def save_results(results: Dict, output_dir: str):
    """Save results to files."""
    if not results:
        return
    
    stat = results['stat']
    stat_dir = os.path.join(output_dir, f"{stat}_results")
    os.makedirs(stat_dir, exist_ok=True)
    
    # Save best features
    features_df = pd.DataFrame({
        'rank': range(1, len(results['feature_ranking']) + 1),
        'feature': [f for f, _ in results['feature_ranking']],
        'importance': [imp for _, imp in results['feature_ranking']]
    })
    features_df.to_csv(os.path.join(stat_dir, "best_features_ranked.csv"), index=False)
    
    # Save performance history
    if results['performance_history']:
        history_df = pd.DataFrame(results['performance_history'])
        history_df.to_csv(os.path.join(stat_dir, "performance_history.csv"), index=False)
    
    # Save summary
    with open(os.path.join(stat_dir, "summary.txt"), 'w') as f:
        f.write(f"BEST FEATURES FOR {stat.upper()} PREDICTION\n")
        f.write("="*50 + "\n\n")
        f.write(f"Number of features: {results['n_features']}\n")
        f.write(f"Final RMSE: {results['final_rmse']:.4f} ± {results['final_rmse_std']:.4f}\n")
        f.write(f"Final R²: {results['final_r2']:.4f} ± {results['final_r2_std']:.4f}\n")
        f.write(f"Successful trials: {results['n_successful_trials']}\n\n")
        
        f.write("OPTIMAL FEATURE SET:\n")
        f.write("-"*30 + "\n")
        for i, feat in enumerate(results['best_features'], 1):
            importance = results['feature_importance'].get(feat, 0)
            f.write(f"{i:2d}. {feat} (importance: {importance:.4f})\n")
        
        f.write(f"\nFEATURE RANKING (Top 50):\n")
        f.write("-"*30 + "\n")
        for i, (feat, imp) in enumerate(results['feature_ranking'][:50], 1):
            f.write(f"{i:2d}. {feat:50s} {imp:.4f}\n")

def main():
    """Main function to find best features for all statistics."""
    
    print("="*80)
    print("SYSTEMATIC FEATURE SELECTION FOR NBA DFS PREDICTIONS")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    df = pd.read_csv(os.path.join(project_root, 'enriched_df.csv'))
    df = df[df['season_year'] == '2017-18']
    df['game_date'] = pd.to_datetime(df['game_date'])
    df["week"] = df["game_date"].dt.strftime("%Y_%U")
    
    print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Base configuration
    base_config = {
        "target_col": "",  # Will be set per stat 
        "categorical_features": [],
        "salary_quantile_threshold": 0.75,
        "metrics_salary_quantile": 0.9,
        "train_weeks_lookback": 15,
        "num_boost_round": 300,
        "xgb_params": {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.6,
            "min_child_weight": 3,
            "gamma": 0.1,
            "alpha": 0.1,
            "lambda": 1.0,
            "tree_method": "hist",
            "verbosity": 0
        }
    }
    
    # Statistics to analyze
    stats_to_analyze = ["pts", "reb", "ast", "tov", "stl", "blk"]
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"best_features_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    all_results = {}
    total_start_time = time.time()
    
    for i, stat in enumerate(stats_to_analyze, 1):
        print(f"\n{'='*60}")
        print(f"ANALYZING STATISTIC {i}/{len(stats_to_analyze)}: {stat.upper()}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Build feature buckets for this stat
        feature_buckets = build_feature_buckets_for_stat(df.columns, stat)
        must_have = get_must_have_features(stat)
        
        # Setup config
        config = base_config.copy()
        config["target_col"] = stat
        config["categorical_features"] = feature_buckets.get("categorical", [])
        
        # Cast categorical features
        for col in config["categorical_features"]:
            if col in df.columns:
                df[col] = df[col].astype("category")
        
        print(f"Feature buckets:")
        for bucket, features in feature_buckets.items():
            print(f"  {bucket}: {len(features)} features")
        
        # Find best features
        try:
            results = find_best_features_systematic(
                df=df,
                stat_name=stat,
                feature_buckets=feature_buckets,
                must_have=must_have,
                config=config,
                max_features=100,
                n_trials=40
            )
            
            if results:
                all_results[stat] = results
                save_results(results, results_dir)
            
        except Exception as e:
            print(f"ERROR analyzing {stat}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        end_time = time.time()
        duration = (end_time - start_time) / 60
        remaining = (len(stats_to_analyze) - i) * duration
        
        print(f"\n{stat.upper()} completed in {duration:.1f} minutes")
        print(f"Estimated time remaining: {remaining:.1f} minutes")
    
    total_time = (time.time() - total_start_time) / 60
    
    # Create final summary
    print(f"\n{'='*80}")
    print("FEATURE SELECTION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f} minutes")
    print(f"Results saved to: {results_dir}")
    
    # Create comprehensive summary
    summary_data = []
    for stat, results in all_results.items():
        summary_data.append({
            'statistic': stat,
            'n_features': results['n_features'],
            'rmse': results['final_rmse'],
            'r2': results['final_r2'],
            'top_feature': results['feature_ranking'][0][0] if results['feature_ranking'] else 'None'
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(results_dir, "final_summary.csv"), index=False)
        
        print("\nFINAL SUMMARY:")
        print(summary_df.to_string(index=False))
    
    return all_results

if __name__ == "__main__":
    main() 