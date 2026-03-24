"""
WiDS Autonomous Optimization Runner
===================================

Fully autonomous experiment runner that:
1. Proposes experiments based on strategy phases
2. Modifies wids_train_enhanced.py
3. Runs experiments and evaluates results
4. Keeps improvements (>= 0.0010 improvement)
5. Tracks progress and adapts strategy
6. Stops after max experiments or convergence

Usage:
    python wids_autonomous_runner.py "30 experiments"
    python wids_autonomous_runner.py "night batch"
"""

import os
import sys
import subprocess
import re
import time
import random
import shutil
import copy
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

try:
    import yaml
except Exception:
    yaml = None

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
RESULTS_FILE = PROJECT_ROOT / "results.tsv"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

TRAINING_SCRIPT = PROJECT_ROOT / "wids_train_enhanced.py"
CONFIG_FILE = PROJECT_ROOT / "wids_config.yaml"
EXPERIMENT_RUNNER = PROJECT_ROOT / "wids_experiment_runner.py"

# Default autonomous config
MAX_EXPERIMENTS = 30
MIN_IMPROVEMENT = 0.0003
PATIENCE = 10  # Stop if no improvement after N experiments (increased for more exploration)
OVERNIGHT_MODE = False

# ============================================================================
# Strategy Definitions
# ============================================================================

class ExperimentStrategy:
    """Defines experiment phases and parameter suggestions."""

    # Phase 1: Ensemble weights (experiments 1-10)
    PHASE1_WEIGHTS = [
        {'W_PHYS': 0.70, 'W_XGB': 0.20, 'W_RF': 0.05, 'W_GB': 0.05, 'desc': 'More XGBoost for C-index'},
        {'W_PHYS': 0.75, 'W_XGB': 0.15, 'W_RF': 0.05, 'W_GB': 0.05, 'desc': 'Balanced baseline'},
        {'W_PHYS': 0.65, 'W_XGB': 0.25, 'W_RF': 0.05, 'W_GB': 0.05, 'desc': 'Aggressive XGBoost'},
        {'W_PHYS': 0.70, 'W_XGB': 0.15, 'W_RF': 0.05, 'W_GB': 0.10, 'desc': 'More GB for Brier'},
        {'W_PHYS': 0.75, 'W_XGB': 0.10, 'W_RF': 0.05, 'W_GB': 0.10, 'desc': 'Physics + calibration'},
        {'W_PHYS': 0.68, 'W_XGB': 0.22, 'W_RF': 0.05, 'W_GB': 0.05, 'desc': 'Sweet spot XGBoost'},
        {'W_PHYS': 0.72, 'W_XGB': 0.18, 'W_RF': 0.05, 'W_GB': 0.05, 'desc': 'Conservative ML boost'},
        {'W_PHYS': 0.70, 'W_XGB': 0.17, 'W_RF': 0.08, 'W_GB': 0.05, 'desc': 'More RF diversity'},
        {'W_PHYS': 0.68, 'W_XGB': 0.20, 'W_RF': 0.07, 'W_GB': 0.05, 'desc': 'ML-forward ensemble'},
        {'W_PHYS': 0.73, 'W_XGB': 0.17, 'W_RF': 0.05, 'W_GB': 0.05, 'desc': 'Slight ML nudge'},
    ]

    # Phase 2: Hyperparameters (experiments 11-20)
    PHASE2_HYPERPARAMS = [
        {'xgb_max_depth': 2, 'desc': 'Shallower XGBoost (depth=2)'},
        {'xgb_max_depth': 4, 'desc': 'Deeper XGBoost (depth=4)'},
        {'xgb_min_child_weight': 10, 'desc': 'Higher min_child_weight (10)'},
        {'xgb_learning_rate': 0.03, 'desc': 'Slower learning rate (0.03)'},
        {'xgb_learning_rate': 0.10, 'desc': 'Faster learning rate (0.10)'},
        {'rf_max_depth': 3, 'desc': 'Shallower RF (depth=3)'},
        {'rf_min_samples_leaf': 10, 'desc': 'Higher RF min_samples (10)'},
        {'gb_max_depth': 1, 'desc': 'Very shallow GB (depth=1)'},
        {'gb_learning_rate': 0.02, 'desc': 'Slower GB learning (0.02)'},
        {'xgb_subsample': 0.6, 'desc': 'More aggressive subsampling (0.6)'},
    ]

    # Phase 3: Calibration (experiments 21-25)
    PHASE3_CALIBRATION = [
        {'squish_power': 1.0, 'desc': 'No calibration (power=1.0)'},
        {'squish_power': 1.3, 'desc': 'Stronger calibration (power=1.3)'},
        {'squish_power': 1.1, 'desc': 'Mild calibration (power=1.1)'},
        {'squish_power': 1.4, 'desc': 'Aggressive calibration (power=1.4)'},
        {'squish_power': 0.9, 'desc': 'Inverse calibration (power=0.9)'},
    ]

    def __init__(self):
        self.current_phase = 1
        self.experiment_count = 0
        self.best_score = 0.0
        self.no_improvement_count = 0
        self.random_pool = self.PHASE1_WEIGHTS + self.PHASE2_HYPERPARAMS + self.PHASE3_CALIBRATION
        self.used_random_configs = set()

    @staticmethod
    def _config_key(config: dict) -> tuple:
        """Create a stable hashable key for a config dict."""
        return tuple(sorted((k, str(v)) for k, v in config.items()))

    def get_next_experiment(self, results_df: pd.DataFrame) -> dict:
        """Get next experiment configuration based on current progress."""

        self.experiment_count = len(results_df)

        # Get best score so far from kept experiments
        if len(results_df) > 0 and results_df['status'].str.contains('keep').any():
            self.best_score = results_df[results_df['status'] == 'keep']['hybrid_score'].max()
        else:
            # Use highest score available as temporary baseline
            valid_scores = results_df[results_df['hybrid_score'] > 0]['hybrid_score']
            if len(valid_scores) > 0:
                self.best_score = valid_scores.max()
            else:
                self.best_score = 0.0

        # Determine phase based on experiment count
        if self.experiment_count < 10:
            self.current_phase = 1
            configs = self.PHASE1_WEIGHTS[self.experiment_count % len(self.PHASE1_WEIGHTS)]
            config_type = 'weights'
        elif self.experiment_count < 20:
            self.current_phase = 2
            configs = self.PHASE2_HYPERPARAMS[(self.experiment_count - 10) % len(self.PHASE2_HYPERPARAMS)]
            config_type = 'hyperparams'
        elif self.experiment_count < 25:
            self.current_phase = 3
            configs = self.PHASE3_CALIBRATION[(self.experiment_count - 20) % len(self.PHASE3_CALIBRATION)]
            config_type = 'calibration'
        else:
            # Random exploration for remaining experiments
            self.current_phase = 4
            config_type = 'random'
            available = [
                cfg for cfg in self.random_pool
                if self._config_key(cfg) not in self.used_random_configs
            ]

            if not available:
                self.used_random_configs.clear()
                available = self.random_pool.copy()

            configs = random.choice(available)
            self.used_random_configs.add(self._config_key(configs))

        return {
            'phase': self.current_phase,
            'config': configs,
            'config_type': config_type,
            'experiment_num': self.experiment_count + 1
        }


# ============================================================================
# Code Modification Utilities
# ============================================================================

def read_training_script() -> str:
    """Read current editable configuration source (YAML preferred)."""
    if CONFIG_FILE.exists() and yaml is not None:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as handle:
            return yaml.safe_load(handle) or {}
    return TRAINING_SCRIPT.read_text()


def write_training_script(content: str):
    """Write modified configuration source (YAML preferred)."""
    if isinstance(content, dict) and CONFIG_FILE.exists() and yaml is not None:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as handle:
            yaml.safe_dump(content, handle, sort_keys=False)
        return
    TRAINING_SCRIPT.write_text(content)


def modify_weights(content: str, weights: dict) -> tuple[str, str]:
    """Modify ensemble weights in the training script."""

    if isinstance(content, dict):
        modified = copy.deepcopy(content)
        ensemble = modified.setdefault('ensemble', {})
        if 'W_PHYS' in weights:
            ensemble['w_phys'] = float(weights['W_PHYS'])
        if 'W_XGB' in weights:
            ensemble['w_xgb'] = float(weights['W_XGB'])
        if 'W_RF' in weights:
            ensemble['w_rf'] = float(weights['W_RF'])
        if 'W_GB' in weights:
            ensemble['w_gb'] = float(weights['W_GB'])

        if 'desc' in weights:
            description = f"Phase 1: {weights['desc']}"
        else:
            description = "Phase 1 weights (yaml)"
        return modified, description

    modified = content

    # Only process actual weight keys (W_*)
    weight_keys = ['W_PHYS', 'W_XGB', 'W_RF', 'W_GB']

    for key in weight_keys:
        if key in weights:
            value = weights[key]
            pattern = rf'^({key}\s*=\s*)[\d.]+'
            replacement = f'\\g<1>{float(value):.2f}'
            modified = re.sub(pattern, replacement, modified, flags=re.MULTILINE)

    # Create description from the config's description field or build from weights
    if 'desc' in weights:
        description = f"Phase 1: {weights['desc']}"
    else:
        desc_parts = [f"{k}={float(weights[k]):.2f}" for k in weight_keys if k in weights]
        description = f"Phase 1 weights: {', '.join(desc_parts)}"

    return modified, description


def modify_hyperparams(content: str, hyperparams: dict) -> tuple[str, str]:
    """Modify hyperparameters in the training script."""

    if isinstance(content, dict):
        modified = copy.deepcopy(content)
        mp = modified.setdefault('model_params', {})
        description_parts = []
        for key, value in hyperparams.items():
            if key == 'desc':
                continue
            mp[key] = value
            description_parts.append(f"{key}={value}")
        description = f"Phase 2 hyperparams: {', '.join(description_parts)}"
        return modified, description

    modified = content
    description_parts = []

    for key, value in hyperparams.items():
        if key.startswith('xgb_'):
            pattern = rf'^(    xgb_{key[4:]}\s*=\s*)[\d.]+'
            replacement = f'\\g<1>{value}'
            modified = re.sub(pattern, replacement, modified, flags=re.MULTILINE)
            description_parts.append(f"xgb_{key[4:]}={value}")
        elif key.startswith('rf_'):
            pattern = rf'^(    rf_{key[3:]}\s*=\s*)[\d\'.]+'
            replacement = f'\\g<1>{value}'
            modified = re.sub(pattern, replacement, modified, flags=re.MULTILINE)
            description_parts.append(f"rf_{key[3:]}={value}")
        elif key.startswith('gb_'):
            pattern = rf'^(    gb_{key[3:]}\s*=\s*)[\d.]+'
            replacement = f'\\g<1>{value}'
            modified = re.sub(pattern, replacement, modified, flags=re.MULTILINE)
            description_parts.append(f"gb_{key[3:]}={value}")

    description = f"Phase 2 hyperparams: {', '.join(description_parts)}"
    return modified, description


def modify_calibration(content: str, power: float) -> tuple[str, str]:
    """Modify calibration power in the training script."""

    if isinstance(content, dict):
        modified = copy.deepcopy(content)
        calibration = modified.setdefault('calibration', {})
        calibration['power_squish'] = float(power)
        description = f"Phase 3 calibration: power={power:.1f}"
        return modified, description

    pattern = rf'(probs_final\s*=\s*np\.power\(probs_blend,\s*)[\d.]+'
    replacement = f'\\g<1>{power:.1f}'
    modified = re.sub(pattern, replacement, content)

    description = f"Phase 3 calibration: power={power:.1f}"
    return modified, description


def apply_experiment_config(content: str, experiment: dict) -> tuple[str, str]:
    """Apply experiment configuration to training script."""

    config = experiment['config']
    config_type = experiment['config_type']

    if config_type == 'weights':
        return modify_weights(content, config)
    elif config_type == 'hyperparams':
        return modify_hyperparams(content, config)
    elif config_type == 'calibration':
        return modify_calibration(content, config.get('squish_power', 1.2))
    else:
        # Random - route to the correct modifier based on keys
        if any(k.startswith('W_') for k in config.keys()):
            modified, desc = modify_weights(content, config)
            return modified, f"Random exploration ({desc})"
        if 'squish_power' in config:
            modified, desc = modify_calibration(content, config.get('squish_power', 1.2))
            return modified, f"Random exploration ({desc})"
        if any(k.startswith(('xgb_', 'rf_', 'gb_')) for k in config.keys()):
            modified, desc = modify_hyperparams(content, config)
            return modified, f"Random exploration ({desc})"

        # Fallback safety
        return content, f"Random exploration (no-op): {config}"


# ============================================================================
# Experiment Execution
# ============================================================================

def run_experiment(description: str) -> dict:
    """Run a single experiment and return results."""

    print(f"\n{'='*70}")
    print(f"Experiment: {description}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    start_time = time.time()

    try:
        # Run training script directly
        result = subprocess.run(
            ['python', str(TRAINING_SCRIPT)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        elapsed = time.time() - start_time
        output = result.stdout + "\n" + result.stderr

        # Log output
        log_file = LOG_DIR / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        log_file.write_text(output)

        # Extract metrics
        metrics = extract_metrics(output)
        metrics['elapsed_total'] = elapsed
        metrics['returncode'] = result.returncode

        if result.returncode != 0:
            metrics['error'] = f"Script failed with code {result.returncode}"

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


def extract_metrics(output: str) -> dict:
    """Parse metrics from training output."""

    metrics = {
        'hybrid_score': None,
        'c_index': None,
        'brier_score': None,
        'time_sec': 0,
        'error': None
    }

    try:
        # Extract from "FINAL METRICS:" section
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


def load_results() -> pd.DataFrame:
    """Load existing results."""
    if RESULTS_FILE.exists():
        try:
            return pd.read_csv(RESULTS_FILE, sep='\t')
        except:
            pass

    return pd.DataFrame(columns=[
        'commit', 'hybrid_score', 'c_index', 'brier_score',
        'status', 'time_sec', 'description'
    ])


def save_results(df: pd.DataFrame):
    """Save results."""
    df.to_csv(RESULTS_FILE, sep='\t', index=False)


def log_and_evaluate(metrics: dict, description: str, current_content: str, backup_content: str) -> str:
    """Log experiment and decide keep/discard."""

    results_df = load_results()

    hybrid_score = metrics.get('hybrid_score')
    c_index = metrics.get('c_index')
    brier_score = metrics.get('brier_score')
    time_sec = metrics.get('time_sec', metrics.get('elapsed_total', 0))
    error = metrics.get('error')

    # Get best known score from actual results (not hardcoded)
    best_known = 0.0
    if len(results_df) > 0 and results_df['status'].str.contains('keep').any():
        best_known = results_df[results_df['status'] == 'keep']['hybrid_score'].max()
    else:
        # No kept results yet - use first valid score as baseline
        if len(results_df) > 0:
            valid_scores = results_df[results_df['hybrid_score'] > 0]['hybrid_score']
            if len(valid_scores) > 0:
                best_known = valid_scores.max()

    # Determine status
    if error or hybrid_score is None:
        status = 'crash'
        print(f"\n CRASH: {error or 'No metrics'}")
        # Revert on crash
        write_training_script(backup_content)
        print("   Reverted to previous version")
    else:
        improvement = hybrid_score - best_known

        if improvement >= MIN_IMPROVEMENT:
            status = 'keep'
            print(f"\n KEEP: {hybrid_score:.6f} (+{improvement:.6f} vs best)")
            # Keep the changes (already in file)
        elif improvement >= 0.0005:
            status = 'marginal'
            print(f"\n MARGINAL: {hybrid_score:.6f} (+{improvement:.6f})")
            # Keep marginal for now
            write_training_script(backup_content)
        else:
            status = 'discard'
            print(f"\n DISCARD: {hybrid_score:.6f} ({improvement:+.6f})")
            # Revert on discard
            write_training_script(backup_content)

    # Log to results
    exp_num = len(results_df) + 1
    new_row = pd.DataFrame([{
        'commit': f'exp_{exp_num:03d}',
        'hybrid_score': hybrid_score if hybrid_score else 0.0,
        'c_index': c_index if c_index else 0.0,
        'brier_score': brier_score if brier_score else 0.0,
        'status': status,
        'time_sec': time_sec,
        'description': description
    }])

    if len(results_df) == 0:
        results_df = new_row
    else:
        results_df = pd.concat([results_df, new_row], ignore_index=True)
    save_results(results_df)

    return status


def print_summary(results_df: pd.DataFrame):
    """Print progress summary."""

    print(f"\n{'='*70}")
    print("PROGRESS SUMMARY")
    print(f"{'='*70}")

    if len(results_df) == 0:
        print("No experiments yet.")
        return

    kept = results_df[results_df['status'] == 'keep']

    print(f"Total experiments: {len(results_df)}")
    print(f"Kept: {len(kept)}")

    if len(kept) > 0:
        best = kept.loc[kept['hybrid_score'].idxmax()]
        print(f"\n Best score: {best['hybrid_score']:.6f}")
        print(f"   C-Index: {best['c_index']:.6f}")
        print(f"   Brier: {best['brier_score']:.6f}")
        print(f"   Config: {best['description'][:50]}")

    print(f"\n Latest 5 experiments:")
    latest = results_df.tail(5)[['hybrid_score', 'status', 'description']]
    for idx, row in latest.iterrows():
        status_icon = 'OK' if row['status'] == 'keep' else 'X '
        print(f"   [{status_icon}] {row['hybrid_score']:.4f} | {row['description'][:45]}")

    print(f"{'='*70}\n")


# ============================================================================
# Main Autonomous Loop
# ============================================================================

def run_autonomous_loop(max_experiments: int = MAX_EXPERIMENTS):
    """Main autonomous optimization loop."""

    print(f"\n{'='*70}")
    print("WiDS AUTONOMOUS OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Max experiments: {max_experiments}")
    print(f"Min improvement: {MIN_IMPROVEMENT}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}\n")

    strategy = ExperimentStrategy()
    no_improvement_count = 0

    for exp_iter in range(max_experiments):
        # Load current results
        results_df = load_results()

        # Check for early stopping
        if no_improvement_count >= PATIENCE:
            print(f"\n EARLY STOP: No improvement after {PATIENCE} experiments")
            break

        # Get next experiment configuration
        experiment = strategy.get_next_experiment(results_df)

        print(f"\n{'='*50}")
        print(f"EXPERIMENT {experiment['experiment_num']}/{max_experiments}")
        print(f"Phase: {experiment['phase']} | Type: {experiment['config_type']}")
        print(f"{'='*50}\n")

        # Backup current script
        backup_content = read_training_script()
        if isinstance(backup_content, dict):
            backup_content = copy.deepcopy(backup_content)

        # Apply experiment configuration
        modified_content, description = apply_experiment_config(backup_content, experiment)
        write_training_script(modified_content)

        print(f"Configuration: {description}")

        # Run experiment
        metrics = run_experiment(description)

        # Evaluate and log
        status = log_and_evaluate(metrics, description, modified_content, backup_content)

        # Track improvement streak
        if status == 'keep':
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        # Print progress
        print_summary(load_results())

        # Small delay between experiments
        time.sleep(2)

    # Final summary
    print(f"\n{'='*70}")
    print("AUTONOMOUS OPTIMIZATION COMPLETE")
    print(f"{'='*70}")
    print_summary(load_results())


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':

    # Parse command line arguments
    max_exp = MAX_EXPERIMENTS

    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg.isdigit():
            max_exp = int(arg)
        elif 'night' in arg or 'batch' in arg:
            OVERNIGHT_MODE = True
            max_exp = 50  # More experiments for overnight
        elif 'quick' in arg:
            max_exp = 5
        elif 'test' in arg:
            max_exp = 2

    print(f"Autonomous mode: {OVERNIGHT_MODE}")
    print(f"Max experiments: {max_exp}\n")

    run_autonomous_loop(max_exp)
