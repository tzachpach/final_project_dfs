import pandas as pd
import numpy as np
import subprocess
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import glob

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from config.fantasy_point_calculation import calculate_fp_fanduel

def select_device():
    """Select the best available device for computation."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
            return device
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS)")
            return device
        else:
            device = torch.device('cpu')
            print("Using CPU")
            return device
    except ImportError:
        print("PyTorch not available, using CPU for XGBoost")
        return "cpu"

def run_prediction_script(script_name: str, experiment_name: str) -> Tuple[bool, str]:
    """
    Run a prediction script and return success status and any error message.
    """
    script_path = os.path.join("notebooks", script_name)
    
    print(f"\n{'='*60}")
    print(f"Starting {experiment_name} Prediction Experiment")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the script using subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        end_time = time.time()
        duration = (end_time - start_time) / 60  # Convert to minutes
        
        if result.returncode == 0:
            print(f"✅ {experiment_name} completed successfully in {duration:.2f} minutes")
            print("Script output (last 10 lines):")
            output_lines = result.stdout.split('\n')
            for line in output_lines[-10:]:
                if line.strip():
                    print(f"  {line}")
            return True, ""
        else:
            print(f"❌ {experiment_name} failed after {duration:.2f} minutes")
            print("Error output:")
            print(result.stderr)
            return False, result.stderr
            
    except Exception as e:
        end_time = time.time()
        duration = (end_time - start_time) / 60
        error_msg = f"Exception running {experiment_name}: {str(e)}"
        print(f"❌ {error_msg} after {duration:.2f} minutes")
        return False, error_msg

def find_latest_results_dir(stat_prefix: str) -> str:
    """
    Find the most recent results directory for a given stat prefix.
    """
    results_pattern = os.path.join(project_root, "results", f"{stat_prefix}_*")
    matching_dirs = glob.glob(results_pattern)
    
    if not matching_dirs:
        raise ValueError(f"No results directories found for {stat_prefix}")
    
    # Sort by modification time and get the most recent
    latest_dir = max(matching_dirs, key=os.path.getmtime)
    return latest_dir

def load_predictions_from_results(stat: str, phase: str = "exploitation") -> pd.DataFrame:
    """
    Load predictions from the most recent results directory for a given stat.
    """
    try:
        results_dir = find_latest_results_dir(stat)
        predictions_path = os.path.join(results_dir, phase, "all_predictions.csv")
        
        if os.path.exists(predictions_path):
            df = pd.read_csv(predictions_path)
            print(f"✅ Loaded {len(df)} predictions for {stat} from {results_dir}")
            return df
        else:
            print(f"❌ Predictions file not found: {predictions_path}")
            return pd.DataFrame()
            
    except Exception as e:
        print(f"❌ Error loading predictions for {stat}: {str(e)}")
        return pd.DataFrame()

def aggregate_all_predictions() -> pd.DataFrame:
    """
    Aggregate predictions from all stat prediction experiments.
    """
    print("\n" + "="*60)
    print("AGGREGATING ALL PREDICTIONS")
    print("="*60)
    
    stats = ["pts", "reb", "ast", "tov", "stl", "blk"]
    all_predictions = {}
    
    # Load predictions for each stat
    for stat in stats:
        print(f"\nLoading predictions for {stat}...")
        predictions = load_predictions_from_results(stat)
        
        if not predictions.empty:
            # Ensure we have the required columns
            required_cols = ["player_name", "team_abbreviation", "game_date", 
                           f"{stat}", f"{stat}_pred", "week", "salary_quantile"]
            
            if all(col in predictions.columns for col in required_cols):
                all_predictions[stat] = predictions[required_cols]
                print(f"  ✅ {stat}: {len(predictions)} predictions loaded")
            else:
                missing_cols = [col for col in required_cols if col not in predictions.columns]
                print(f"  ❌ {stat}: Missing columns: {missing_cols}")
        else:
            print(f"  ❌ {stat}: No predictions loaded")
    
    if not all_predictions:
        print("❌ No predictions loaded from any experiment!")
        return pd.DataFrame()
    
    # Merge all predictions on common keys
    print(f"\nMerging predictions from {len(all_predictions)} experiments...")
    
    merge_keys = ["player_name", "team_abbreviation", "game_date", "week", "salary_quantile"]
    base_df = None
    
    for stat, pred_df in all_predictions.items():
        # Rename stat columns to avoid conflicts
        pred_df = pred_df.rename(columns={
            stat: f"{stat}_actual",
            f"{stat}_pred": f"{stat}_pred"
        })
        
        if base_df is None:
            base_df = pred_df
        else:
            base_df = pd.merge(
                base_df, 
                pred_df.drop([col for col in pred_df.columns if col in base_df.columns and col not in merge_keys], axis=1),
                on=merge_keys,
                how="inner"
            )
    
    print(f"✅ Merged dataset contains {len(base_df)} rows")
    return base_df

def calculate_fantasy_points_and_metrics(merged_df: pd.DataFrame) -> Dict:
    """
    Calculate fantasy points using FanDuel scoring and compute aggregated metrics.
    """
    print("\n" + "="*60)
    print("CALCULATING FANTASY POINTS AND FINAL METRICS")
    print("="*60)
    
    if merged_df.empty:
        print("❌ No data to calculate fantasy points")
        return {}
    
    stats = ["pts", "reb", "ast", "tov", "stl", "blk"]
    
    # Verify we have all required columns
    required_actual_cols = [f"{stat}_actual" for stat in stats]
    required_pred_cols = [f"{stat}_pred" for stat in stats]
    
    missing_actual = [col for col in required_actual_cols if col not in merged_df.columns]
    missing_pred = [col for col in required_pred_cols if col not in merged_df.columns]
    
    if missing_actual or missing_pred:
        print(f"❌ Missing columns - Actual: {missing_actual}, Predicted: {missing_pred}")
        return {}
    
    print(f"Calculating fantasy points for {len(merged_df)} player-games...")
    
    # Calculate actual fantasy points
    merged_df['fp_actual'] = merged_df.apply(
        lambda row: calculate_fp_fanduel({
            'pts': row['pts_actual'],
            'reb': row['reb_actual'], 
            'ast': row['ast_actual'],
            'tov': row['tov_actual'],
            'stl': row['stl_actual'],
            'blk': row['blk_actual']
        }), axis=1
    )
    
    # Calculate predicted fantasy points
    merged_df['fp_pred'] = merged_df.apply(
        lambda row: calculate_fp_fanduel({
            'pts': row['pts_pred'],
            'reb': row['reb_pred'],
            'ast': row['ast_pred'], 
            'tov': row['tov_pred'],
            'stl': row['stl_pred'],
            'blk': row['blk_pred']
        }), axis=1
    )
    
    # Calculate metrics for each individual stat
    individual_metrics = {}
    for stat in stats:
        actual_col = f"{stat}_actual"
        pred_col = f"{stat}_pred"
        
        if actual_col in merged_df.columns and pred_col in merged_df.columns:
            # Filter for high-salary players for final metrics
            high_salary_mask = merged_df["salary_quantile"] >= 0.9
            actual_high = merged_df[high_salary_mask][actual_col]
            pred_high = merged_df[high_salary_mask][pred_col]
            
            if len(actual_high) > 0:
                rmse = sqrt(mean_squared_error(actual_high, pred_high))
                mae = mean_absolute_error(actual_high, pred_high)
                r2 = r2_score(actual_high, pred_high)
                bias = (pred_high - actual_high).mean()
                
                individual_metrics[stat] = {
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'bias': bias,
                    'n_samples': len(actual_high)
                }
                
                print(f"✅ {stat.upper()}: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, Bias={bias:.4f}")
    
    # Calculate metrics for fantasy points
    high_salary_mask = merged_df["salary_quantile"] >= 0.9
    fp_actual_high = merged_df[high_salary_mask]['fp_actual']
    fp_pred_high = merged_df[high_salary_mask]['fp_pred']
    
    if len(fp_actual_high) > 0:
        fp_rmse = sqrt(mean_squared_error(fp_actual_high, fp_pred_high))
        fp_mae = mean_absolute_error(fp_actual_high, fp_pred_high)
        fp_r2 = r2_score(fp_actual_high, fp_pred_high)
        fp_bias = (fp_pred_high - fp_actual_high).mean()
        
        fantasy_metrics = {
            'rmse': fp_rmse,
            'mae': fp_mae,
            'r2': fp_r2,
            'bias': fp_bias,
            'n_samples': len(fp_actual_high)
        }
        
        print(f"\n🏆 FANTASY POINTS: RMSE={fp_rmse:.4f}, MAE={fp_mae:.4f}, R²={fp_r2:.4f}, Bias={fp_bias:.4f}")
    else:
        fantasy_metrics = {}
    
    return {
        'individual_stats': individual_metrics,
        'fantasy_points': fantasy_metrics,
        'data': merged_df
    }

def save_aggregated_results(results: Dict, timestamp: str):
    """
    Save aggregated results to files.
    """
    print("\n" + "="*60)
    print("SAVING AGGREGATED RESULTS")
    print("="*60)
    
    # Create aggregated results directory
    agg_results_dir = os.path.join(project_root, "results", f"aggregated_{timestamp}")
    os.makedirs(agg_results_dir, exist_ok=True)
    
    # Save the merged predictions dataset
    if 'data' in results and not results['data'].empty:
        data_path = os.path.join(agg_results_dir, "all_predictions_merged.csv")
        results['data'].to_csv(data_path, index=False)
        print(f"✅ Saved merged predictions: {data_path}")
    
    # Save individual stat metrics
    if 'individual_stats' in results:
        individual_metrics_df = pd.DataFrame(results['individual_stats']).T
        individual_path = os.path.join(agg_results_dir, "individual_stat_metrics.csv")
        individual_metrics_df.to_csv(individual_path)
        print(f"✅ Saved individual stat metrics: {individual_path}")
    
    # Save fantasy point metrics
    if 'fantasy_points' in results and results['fantasy_points']:
        fp_metrics_df = pd.DataFrame([results['fantasy_points']])
        fp_path = os.path.join(agg_results_dir, "fantasy_points_metrics.csv")
        fp_metrics_df.to_csv(fp_path, index=False)
        print(f"✅ Saved fantasy points metrics: {fp_path}")
    
    # Save summary report
    summary_path = os.path.join(agg_results_dir, "experiment_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("NBA DFS PREDICTION EXPERIMENTS - SUMMARY REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Experiment completed: {timestamp}\n\n")
        
        if 'individual_stats' in results:
            f.write("INDIVIDUAL STAT PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for stat, metrics in results['individual_stats'].items():
                f.write(f"{stat.upper()}:\n")
                f.write(f"  RMSE: {metrics['rmse']:.4f}\n")
                f.write(f"  MAE:  {metrics['mae']:.4f}\n")
                f.write(f"  R²:   {metrics['r2']:.4f}\n")
                f.write(f"  Bias: {metrics['bias']:.4f}\n")
                f.write(f"  N:    {metrics['n_samples']}\n\n")
        
        if 'fantasy_points' in results and results['fantasy_points']:
            f.write("FANTASY POINTS PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            fp_metrics = results['fantasy_points']
            f.write(f"RMSE: {fp_metrics['rmse']:.4f}\n")
            f.write(f"MAE:  {fp_metrics['mae']:.4f}\n")
            f.write(f"R²:   {fp_metrics['r2']:.4f}\n")
            f.write(f"Bias: {fp_metrics['bias']:.4f}\n")
            f.write(f"N:    {fp_metrics['n_samples']}\n")
    
    print(f"✅ Saved summary report: {summary_path}")
    print(f"\n📁 All results saved to: {agg_results_dir}")

def main():
    """
    Main function to run all experiments and aggregate results.
    """
    print("🏀 NBA DFS PREDICTION EXPERIMENTS - COMPREHENSIVE RUNNER")
    print("=" * 60)
    
    # Select best available device
    device = select_device()
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define experiments to run
    experiments = [
        ("predicting_pts.py", "Points"),
        ("predicting_reb.py", "Rebounds"), 
        ("predicting_ast.py", "Assists"),
        ("predicting_tov.py", "Turnovers"),
        ("predicting_stl.py", "Steals"),
        ("predicting_blk.py", "Blocks")
    ]
    
    # Track experiment results
    experiment_results = {}
    total_start_time = time.time()
    
    # Run each experiment
    for script_name, experiment_name in experiments:
        success, error_msg = run_prediction_script(script_name, experiment_name)
        experiment_results[experiment_name] = {
            'success': success,
            'error': error_msg
        }
        
        if not success:
            print(f"⚠️  {experiment_name} failed - continuing with other experiments")
    
    # Print experiment summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    successful_experiments = []
    failed_experiments = []
    
    for exp_name, result in experiment_results.items():
        if result['success']:
            successful_experiments.append(exp_name)
            print(f"✅ {exp_name}: SUCCESS")
        else:
            failed_experiments.append(exp_name)
            print(f"❌ {exp_name}: FAILED - {result['error'][:100]}...")
    
    print(f"\nSUCCESSFUL: {len(successful_experiments)}/{len(experiments)}")
    print(f"FAILED: {len(failed_experiments)}/{len(experiments)}")
    
    # Only proceed with aggregation if we have successful experiments
    if successful_experiments:
        print(f"\nProceeding with result aggregation for {len(successful_experiments)} successful experiments...")
        
        # Wait a moment for file system to sync
        time.sleep(2)
        
        # Aggregate all predictions
        merged_predictions = aggregate_all_predictions()
        
        if not merged_predictions.empty:
            # Calculate fantasy points and metrics
            results = calculate_fantasy_points_and_metrics(merged_predictions)
            
            # Save aggregated results
            save_aggregated_results(results, timestamp)
            
            # Print final summary
            print("\n" + "🎯" * 20)
            print("FINAL RESULTS SUMMARY")
            print("🎯" * 20)
            
            if 'fantasy_points' in results and results['fantasy_points']:
                fp_metrics = results['fantasy_points']
                print(f"\n🏆 FANTASY POINTS PREDICTION PERFORMANCE:")
                print(f"   RMSE: {fp_metrics['rmse']:.4f}")
                print(f"   MAE:  {fp_metrics['mae']:.4f}")
                print(f"   R²:   {fp_metrics['r2']:.4f}")
                print(f"   Bias: {fp_metrics['bias']:.4f}")
                print(f"   High-salary players: {fp_metrics['n_samples']}")
                
                # Interpretation
                print(f"\n📊 INTERPRETATION:")
                if fp_metrics['r2'] > 0.3:
                    print("   🟢 Strong predictive power")
                elif fp_metrics['r2'] > 0.15:
                    print("   🟡 Moderate predictive power")
                else:
                    print("   🔴 Limited predictive power")
                    
                if abs(fp_metrics['bias']) < 1.0:
                    print("   🟢 Low bias - well calibrated predictions")
                elif abs(fp_metrics['bias']) < 2.0:
                    print("   🟡 Moderate bias")
                else:
                    print("   🔴 High bias - systematic over/under prediction")
        else:
            print("❌ Failed to aggregate predictions")
    else:
        print("❌ No successful experiments to aggregate")
    
    total_end_time = time.time()
    total_duration = (total_end_time - total_start_time) / 60
    
    print(f"\n⏱️  Total experiment runtime: {total_duration:.1f} minutes")
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if successful_experiments:
        print("\n🎯 All experiments completed! Check the results directory for detailed outputs.")
    else:
        print("\n💥 All experiments failed! Check error messages and logs.")

if __name__ == "__main__":
    main() 