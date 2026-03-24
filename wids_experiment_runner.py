"""
WiDS Experiment Orchestrator

Manages autonomous iteration:
1. Runs wids_train.py (captures output)
2. Extracts metrics (hybrid_score, c_index, brier)
3. Compares to baseline
4. Logs results to results.tsv
5. Suggests next experiment
6. Commits to git if improved
"""

import os
import sys
import subprocess
import re
import time
import pandas as pd
from datetime import datetime
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

# Project structure: files are at root level
PROJECT_ROOT = Path(__file__).parent  # Root of project
RESULTS_FILE = PROJECT_ROOT / "results.tsv"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRAINING_SCRIPT = PROJECT_ROOT / "wids_train_enhanced.py"

BASELINE_SCORE = 0.990362  # Known from first run

# Autonomous mode configuration
AUTONOMOUS_CONFIG = {
    'max_experiments': 30,
    'min_improvement': 0.0010,  # Must improve by this much to keep
    'patience': 5,  # Stop if no improvement after N experiments
    'overnight_mode': False,  # Set True for unattended runs
}

# ============================================================================
# Utilities
# ============================================================================

def extract_metrics_from_output(output: str) -> dict:
    """Parse metrics from wids_train.py output."""
    
    metrics = {
        'hybrid_score': None,
        'c_index': None,
        'brier_score': None,
        'time_sec': 0,
        'error': None
    }
    
    try:
        # Extract from "Final Metrics:" section
        final_match = re.search(r'Hybrid Score:\s*([\d.]+)', output)
        if final_match:
            metrics['hybrid_score'] = float(final_match.group(1))
        
        c_idx_match = re.search(r'C-Index:\s*([\d.]+)', output)
        if c_idx_match:
            metrics['c_index'] = float(c_idx_match.group(1))
        
        brier_match = re.search(r'Brier Score:\s*([\d.]+)', output)
        if brier_match:
            metrics['brier_score'] = float(brier_match.group(1))
        
        time_match = re.search(r'Time:\s*([\d.]+)s', output)
        if time_match:
            metrics['time_sec'] = float(time_match.group(1))
        
        if metrics['hybrid_score'] is None:
            metrics['error'] = "Could not parse metrics from output"
        
    except Exception as e:
        metrics['error'] = str(e)
    
    return metrics


def run_experiment() -> dict:
    """Execute wids_train_enhanced.py and capture metrics."""
    
    print(f"\n{'='*70}")
    print(f"Running WiDS Training Experiment")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")
    
    start_time = time.time()
    
    try:
        # Run training script
        result = subprocess.run(
            ['python', str(TRAINING_SCRIPT)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        elapsed = time.time() - start_time
        
        # Capture output
        output = result.stdout + "\n" + result.stderr
        
        # Log to file
        log_file = LOG_DIR / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.write_text(output)
        print(f"Logged to: {log_file}")
        
        # Extract metrics
        metrics = extract_metrics_from_output(output)
        metrics['elapsed_total'] = elapsed
        metrics['returncode'] = result.returncode
        
        if result.returncode != 0:
            metrics['error'] = f"Script failed with code {result.returncode}"
            print(f"⚠️ Script error: {metrics['error']}")
        
        return metrics
        
    except subprocess.TimeoutExpired:
        return {
            'hybrid_score': None,
            'c_index': None,
            'brier_score': None,
            'elapsed_total': 600,
            'error': "Timeout (>10 min)"
        }
    except Exception as e:
        return {
            'hybrid_score': None,
            'c_index': None,
            'brier_score': None,
            'elapsed_total': time.time() - start_time,
            'error': str(e)
        }


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout.strip()
    except:
        return "unknown"


def load_results() -> pd.DataFrame:
    """Load existing results.tsv or create new."""
    if RESULTS_FILE.exists():
        try:
            return pd.read_csv(RESULTS_FILE, sep='\t')
        except:
            pass
    
    # Create new
    return pd.DataFrame(columns=[
        'commit', 'hybrid_score', 'c_index', 'brier_score',
        'status', 'time_sec', 'description'
    ])


def save_results(df: pd.DataFrame):
    """Save results.tsv."""
    df.to_csv(RESULTS_FILE, sep='\t', index=False)
    print(f"✓ Saved results to: {RESULTS_FILE}")


def log_experiment(metrics: dict, description: str = ""):
    """Log experiment results."""
    
    results_df = load_results()
    commit = get_git_commit_hash()
    
    hybrid_score = metrics.get('hybrid_score')
    c_index = metrics.get('c_index')
    brier_score = metrics.get('brier_score')
    time_sec = metrics.get('time_sec', metrics.get('elapsed_total', 0))
    error = metrics.get('error')
    
    # Determine status
    if error:
        status = 'crash'
        print(f"❌ CRASH: {error}")
    elif hybrid_score is None:
        status = 'crash'
        print(f"❌ CRASH: No metrics extracted")
    else:
        # Get best known score
        best_known = BASELINE_SCORE
        if len(results_df) > 0 and results_df['status'].str.contains('keep').any():
            best_known = results_df[results_df['status'] == 'keep']['hybrid_score'].max()
        
        improvement = hybrid_score - best_known
        improvement_pct = (improvement / best_known * 100) if best_known > 0 else 0
        
        if improvement >= 0.0010:
            status = 'keep'
            print(f"✅ KEEP: +{improvement:.6f} (+{improvement_pct:.3f}%)")
        elif improvement >= 0.0005:
            status = 'marginal'
            print(f"⚠️ MARGINAL: +{improvement:.6f} (consider code complexity)")
        else:
            status = 'discard'
            print(f"❌ DISCARD: {improvement:.6f} (no improvement)")
    
    # Add row
    new_row = pd.DataFrame([{
        'commit': commit,
        'hybrid_score': hybrid_score if hybrid_score else 0.0,
        'c_index': c_index if c_index else 0.0,
        'brier_score': brier_score if brier_score else 0.0,
        'status': status,
        'time_sec': time_sec,
        'description': description
    }])
    
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    save_results(results_df)
    
    return status


def print_summary(results_df: pd.DataFrame):
    """Print summary of experiments."""
    
    print(f"\n{'='*70}")
    print("Results Summary")
    print(f"{'='*70}\n")
    
    if len(results_df) == 0:
        print("No experiments yet.")
        return
    
    # Overall stats
    kept = results_df[results_df['status'] == 'keep']
    discard = results_df[results_df['status'] == 'discard']
    crash = results_df[results_df['status'] == 'crash']
    
    print(f"Total experiments: {len(results_df)}")
    print(f"  ✅ Kept:     {len(kept)}")
    print(f"  ❌ Discarded: {len(discard)}")
    print(f"  💥 Crashed:   {len(crash)}")
    
    # Best result
    if len(kept) > 0:
        best = kept.loc[kept['hybrid_score'].idxmax()]
        best_score = best['hybrid_score']
        improvement = best_score - BASELINE_SCORE
        print(f"\nBest hybrid score: {best_score:.6f} ({improvement:+.6f} from baseline)")
        print(f"  C-Index:    {best['c_index']:.6f}")
        print(f"  Brier:      {best['brier_score']:.6f}")
        print(f"  Commit:     {best['commit']}")
    
    # Explore trends
    print(f"\nLatest 5 experiments:")
    latest = results_df.tail(5)[['commit', 'hybrid_score', 'status', 'description']]
    for idx, row in latest.iterrows():
        print(f"  {row['status']:8s} | {row['hybrid_score']:.6f} | {row['commit']} | {row['description'][:40]}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    
    # Optional: description passed as argument
    description = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "auto-experiment"
    
    # Run experiment
    print(f"Description: {description}")
    metrics = run_experiment()
    
    # Log results
    status = log_experiment(metrics, description)
    
    # Print summary
    results_df = load_results()
    print_summary(results_df)
    
    # Exit code based on status
    sys.exit(0 if status in ['keep', 'marginal'] else 1)
