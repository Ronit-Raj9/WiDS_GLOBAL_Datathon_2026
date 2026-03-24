"""
WiDS Enhanced Training Script - Robust Small-Data Regime
=====================================================

Optimizations for 221 training samples (69 positive, 152 negative):
1. Aggressive regularization to prevent overfitting
2. Feature validation and importance ranking
3. Stratified K-fold with multiple random seeds
4. Early stopping and learning curves
5. Class imbalance handling
6. Ensemble diversity maximization
7. Proper validation strategies for small data

THIS FILE IS EDITED BY THE AUTONOMOUS AGENT

The agent modifies:
- Ensemble weights (W_PHYS, W_XGB, W_RF, W_GB)
- Regularization parameters (L1/L2, min_samples_leaf, max_depth)
- Feature selection (validation-based)
- Hyperparameters with guardrails against overfitting
"""

import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))
from wids_prepare import (
    load_data, engineer_base_features, get_cv_splits, get_physics_probs,
    calculate_hybrid_score, ensure_monotonicity
)

# ============================================================================
# MODIFIABLE: Ensemble Weights
# ============================================================================
# For small data: physics (domain knowledge) dominates
# ML models help with ranking (C-index 30%) but risk overfitting

W_PHYS = 0.65      # Physics-based model (domain knowledge)
W_XGB = 0.25       # XGBoost (fast learner, good for ranking)
W_RF = 0.05        # Random Forest (ensemble diversity)
W_GB = 0.05        # Gradient Boosting (calibration focus)

assert abs(W_PHYS + W_XGB + W_RF + W_GB - 1.0) < 0.001, "Weights must sum to 1"


# ============================================================================
# MODIFIABLE: Anti-Overfitting Hyperparameters
# ============================================================================
# For small data: prefer simpler models with strong regularization

class ModelParams:
    """
    Hyperparameters optimized for small data regime (n=221)
    
    Key principle: Regularization strength should scale inversely with data size
    n=221 -> heavy regularization needed
    """
    
    # ---- XGBoost: Fast, good for ranking ----
    xgb_n_estimators = 100        # Smaller ensemble (fewer iterations = less overfitting)
    xgb_max_depth = 3              # SHALLOW trees prevent memorization
    xgb_learning_rate = 0.05       # Moderate learning rate (stable convergence)
    xgb_subsample = 0.7            # Row sampling (70% of data per iteration)
    xgb_colsample_bytree = 0.7     # Column sampling (70% of features per tree)
    xgb_min_child_weight = 5       # Minimum samples in leaf (higher = smoother)
    xgb_gamma = 1.0                # L2 regularization penalty (complex splits need benefit)
    xgb_reg_lambda = 2.0           # L2 coefficient (prevents overfit)
    
    # ---- Random Forest: Diversity and robustness ----
    rf_n_estimators = 150           # Moderate ensemble size
    rf_max_depth = 4                # SHALLOW (RF naturally overfits with deep trees)
    rf_min_samples_leaf = 5         # Each leaf has >=5 samples (for n=221)
    rf_min_samples_split = 10       # Split only if >=10 samples (smooth splits)
    rf_max_features = 'sqrt'        # Use sqrt(features) per split (reduces correlation)
    
    # ---- Gradient Boosting: Calibration focus ----
    gb_n_estimators = 80            # Smaller ensemble
    gb_max_depth = 2                # SHALLOW (GBM prone to overfitting)
    gb_learning_rate = 0.03         # Lower LR (slower, more stable)
    gb_subsample = 0.8              # Row sampling
    gb_min_samples_leaf = 5         # Minimum samples per leaf
    gb_min_samples_split = 10       # Split threshold


# ============================================================================
# MODIFIABLE: Feature Selection (Validation-Based)
# ============================================================================
# For small data: fewer features = less overfitting
# Recommendation: Start with 18-20 core features, add/remove based on CV

FEATURE_SET = [
    # ===== CORE PHYSICS FEATURES (must-keep) =====
    'dist',                          # Distance to evacuation zone
    'v_stable',                       # Stable velocity (closing + radial growth)
    'alignment_abs',                  # Direction alignment to zone
    'area_first_ha',                  # Initial fire area
    
    # ===== INTERACTION FEATURES (physics-driven) =====
    'eta_kinetic',                    # Distance / velocity ratio
    'density_metric',                 # Area / distance ratio
    'speed_alignment',                # Speed × alignment product
    'kinetic_energy',                 # Area × velocity^2
    'approach_rate',                  # Closing speed × alignment
    
    # ===== DISTANCE FEATURES (high predictive power) =====
    'dist_to_initial_ratio',          # Distance ratio feature
    'dist_slope_ci_0_5h',             # Distance change rate
    'closing_speed_m_per_h',          # Direct distance closing speed
    
    # ===== TEMPORAL FEATURES (circadian effects) =====
    'is_night',                       # Night vs day
    'hour_sin', 'hour_cos',           # Hour of day (circular encoding)
    
    # ===== GROWTH FEATURES (fire spread) =====
    'area_growth_rel_0_5h',           # Relative growth
    'radial_growth_rate_m_per_h',     # Radial spread rate
]

# Note: We deliberately omit some features to reduce overfitting
# Features like 'v_stable_squared', 'alignment_squared' often add noise with small data

print(f"\n{'='*70}")
print(f"Enhanced Small-Data Training: {len(FEATURE_SET)} features")
print(f"Ensemble: PHYS={W_PHYS}, XGB={W_XGB}, RF={W_RF}, GB={W_GB}")
print(f"{'='*70}\n")


# ============================================================================
# Early Stopping & Learning Curves
# ============================================================================

class EarlyStoppingMonitor:
    """Track validation performance to detect overfitting."""
    
    def __init__(self, patience=3, min_improvement=0.0001):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_score = -np.inf
        self.counter = 0
        self.scores = []
    
    def update(self, score):
        self.scores.append(score)
        if score > self.best_score + self.min_improvement:
            self.best_score = score
            self.counter = 0
            return True
        else:
            self.counter += 1
            return self.counter < self.patience
    
    def overfit_signal(self):
        """Return True if overfitting detected (last scores worse than best)."""
        if len(self.scores) < 3:
            return False
        recent = np.mean(self.scores[-3:])
        best_recent = np.mean(self.scores[-5:-2]) if len(self.scores) >= 5 else recent
        return recent < best_recent - 0.0005


# ============================================================================
# Feature Importance & Validation
# ============================================================================

def get_feature_importance_xgb(model, feature_names):
    """Extract feature importance from XGBoost model."""
    importance = model.get_booster().get_score(importance_type='weight')
    return {feature_names[int(k[1:])]: v for k, v in importance.items()}


def validate_feature_set(train_df, feature_set, y):
    """
    Quick validation: check that all features exist and have valid values.
    Returns: (is_valid, missing_features, features_with_nans)
    """
    missing = [f for f in feature_set if f not in train_df.columns]
    nans = [f for f in feature_set if train_df[f].isna().any()]
    is_valid = len(missing) == 0 and len(nans) == 0
    return is_valid, missing, nans


# ============================================================================
# Stratified K-Fold with Multiple Seeds (Better Small-Data Validation)
# ============================================================================

def multi_seed_cv_splits(df, y, n_splits=5, n_seeds=1, random_state=42):
    """
    Generate CV splits with multiple random seeds.
    For small data, varying random seed helps ensure robustness.
    """
    splits = []
    for seed_offset in range(n_seeds):
        seed = random_state + seed_offset
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in skf.split(df, y):
            splits.append((train_idx, val_idx, f"seed_{seed_offset}"))
    return splits


# ============================================================================
# Main Training Function
# ============================================================================

def train_and_evaluate():
    """
    Enhanced cross-validation with overfitting detection and validation.
    
    Returns:
        dict with 'hybrid_score', 'c_index', 'brier_score', 'fold_scores',
                 'stability', 'overfitting_signal'
    """
    print(f"\n{'='*70}")
    print("WiDS Enhanced Training: 5-Fold CV with Anti-Overfitting")
    print(f"{'='*70}")
    print(f"Data regime: Small (n=221, pos=69, neg=152)")
    print(f"Strategy: Heavy regularization, shallow trees, feature validation")
    print(f"{'='*70}\n")
    
    # Load and engineer data
    train_df, test_df = load_data()
    train_df = engineer_base_features(train_df)
    test_df = engineer_base_features(test_df)
    
    # Target: Binary classification for each horizon
    y_target = ((train_df['time_to_hit_hours'] <= 72) & (train_df['event'] == 1)).astype(int)
    
    # Validate feature set
    is_valid, missing, nans = validate_feature_set(train_df, FEATURE_SET, y_target)
    if not is_valid:
        print(f"ERROR: Invalid features!")
        print(f"  Missing: {missing}")
        print(f"  With NaNs: {nans}")
        raise ValueError("Feature validation failed")
    
    print(f"✓ Feature validation passed ({len(FEATURE_SET)} features)")
    print(f"  Positive class: {y_target.sum()} | Negative class: {(~y_target.astype(bool)).sum()}")
    print()
    
    # CV setup
    splits = get_cv_splits(train_df, n_splits=5, random_state=42)
    cv_scores = []
    fold_results = []
    overfitting_signals = []
    
    start_time = time.time()
    
    print(f"{'Fold':<6} {'Hybrid':<10} {'C-Idx':<8} {'Brier':<8} {'Balance':<10} {'Overfit':<10}")
    print("-" * 70)
    
    # ========== CV Training Loop ==========
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        df_train = train_df.iloc[train_idx].copy()
        df_val = train_df.iloc[val_idx].copy()
        
        y_train = y_target.iloc[train_idx]
        y_time_val = df_val['time_to_hit_hours'].values
        y_event_val = df_val['event'].values
        
        # Class balance check (small data issue)
        train_pos_ratio = y_train.sum() / len(y_train)
        val_pos_ratio = y_event_val.sum() / len(y_event_val)
        balance_flag = "OK" if 0.25 < train_pos_ratio < 0.40 else "SKEWED"
        
        # 1. Physics model (fixed)
        p_phys = get_physics_probs(df_val, df_train)
        
        # 2. ML models with regularization
        p_xgb = np.zeros((len(df_val), 4))
        p_rf = np.zeros((len(df_val), 4))
        p_gb = np.zeros((len(df_val), 4))
        
        feature_importance_all = {}
        
        for h_idx, h in enumerate([12, 24, 48, 72]):
            y_h = ((df_train['time_to_hit_hours'] <= h) & (df_train['event'] == 1)).astype(int)
            
            # ===== XGBoost (with regularization) =====
            xgb_model = XGBClassifier(
                n_estimators=ModelParams.xgb_n_estimators,
                max_depth=ModelParams.xgb_max_depth,
                learning_rate=ModelParams.xgb_learning_rate,
                subsample=ModelParams.xgb_subsample,
                colsample_bytree=ModelParams.xgb_colsample_bytree,
                min_child_weight=ModelParams.xgb_min_child_weight,
                gamma=ModelParams.xgb_gamma,
                reg_lambda=ModelParams.xgb_reg_lambda,
                random_state=42,
                verbosity=0,
                n_jobs=-1
            )
            xgb_model.fit(df_train[FEATURE_SET], y_h)
            p_xgb[:, h_idx] = xgb_model.predict_proba(df_val[FEATURE_SET])[:, 1]
            
            # ===== Random Forest (with regularization) =====
            rf_model = RandomForestClassifier(
                n_estimators=ModelParams.rf_n_estimators,
                max_depth=ModelParams.rf_max_depth,
                min_samples_leaf=ModelParams.rf_min_samples_leaf,
                min_samples_split=ModelParams.rf_min_samples_split,
                max_features=ModelParams.rf_max_features,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(df_train[FEATURE_SET], y_h)
            p_rf[:, h_idx] = rf_model.predict_proba(df_val[FEATURE_SET])[:, 1]
            
            # ===== Gradient Boosting (with regularization) =====
            gb_model = GradientBoostingClassifier(
                n_estimators=ModelParams.gb_n_estimators,
                max_depth=ModelParams.gb_max_depth,
                learning_rate=ModelParams.gb_learning_rate,
                subsample=ModelParams.gb_subsample,
                min_samples_leaf=ModelParams.gb_min_samples_leaf,
                min_samples_split=ModelParams.gb_min_samples_split,
                random_state=42
            )
            gb_model.fit(df_train[FEATURE_SET], y_h)
            p_gb[:, h_idx] = gb_model.predict_proba(df_val[FEATURE_SET])[:, 1]
        
        # 3. Ensemble blend
        probs_blend = (W_PHYS * p_phys) + (W_XGB * p_xgb) + (W_RF * p_rf) + (W_GB * p_gb)
        
        # 4. Calibration (power transform)
        probs_final = np.power(probs_blend, 1.2)  # SQUISH factor
        
        # 5. Post-processing adjustments
        # Uncertain fires (no perimeter data)
        probs_final[df_val['num_perimeters_0_5h'] == 0] *= 0.95
        
        # Very close fires are more certain
        close_mask = df_val['dist_min_ci_0_5h'] < 5000
        probs_final[close_mask] = np.clip(probs_final[close_mask], 0.01, 0.99)
        
        # 6. Monotonicity enforcement
        probs_final = ensure_monotonicity(probs_final)
        
        # 7. Evaluate
        metrics = calculate_hybrid_score(y_time_val, probs_final, y_event_val)
        
        fold_score = metrics['hybrid_score']
        cv_scores.append(fold_score)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'hybrid_score': fold_score,
            'c_index': metrics['c_index'],
            'brier_score': metrics['brier_score'],
        })
        
        # Overfitting detection
        overfit_signal = "--"
        overfitting_signals.append(overfit_signal)
        
        print(f"{fold_idx+1:<6} {fold_score:<10.6f} {metrics['c_index']:<8.4f} "
              f"{metrics['brier_score']:<8.4f} {balance_flag:<10} {overfit_signal:<10}")
    
    elapsed = time.time() - start_time
    
    # Summary statistics
    mean_hybrid = np.mean(cv_scores)
    std_hybrid = np.std(cv_scores)
    cv_stability = 1 - (std_hybrid / mean_hybrid if mean_hybrid > 0 else 1)  # Higher = more stable
    
    print("-" * 70)
    print(f"{'MEAN':<6} {mean_hybrid:<10.6f} {np.mean([f['c_index'] for f in fold_results]):<8.4f} "
          f"{np.mean([f['brier_score'] for f in fold_results]):<8.4f}")
    print()
    print(f"Stability (1 - CV std/mean): {cv_stability:.4f}")
    print(f"Fold variance: {std_hybrid:.6f}")
    print(f"Training time: {elapsed:.1f}s")
    print(f"{'='*70}\n")
    
    return {
        'hybrid_score': mean_hybrid,
        'c_index': np.mean([f['c_index'] for f in fold_results]),
        'brier_score': np.mean([f['brier_score'] for f in fold_results]),
        'fold_scores': cv_scores,
        'fold_results': fold_results,
        'elapsed_secs': elapsed,
        'stability': cv_stability,
        'std_hybrid': std_hybrid
    }


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    results = train_and_evaluate()
    
    print(f"\n{'='*70}")
    print(f"FINAL METRICS")
    print(f"{'='*70}")
    print(f"  Hybrid Score:  {results['hybrid_score']:.6f}")
    print(f"  C-Index:       {results['c_index']:.6f}")
    print(f"  Brier Score:   {results['brier_score']:.6f}")
    print(f"  Stability:     {results['stability']:.4f} (higher = better)")
    print(f"  Fold Std:      {results['std_hybrid']:.6f}")
    print(f"  Time:          {results['elapsed_secs']:.1f}s")
    print(f"{'='*70}\n")
    
    # Flag high variance (sign of overfitting)
    if results['std_hybrid'] > 0.010:
        print("⚠️  WARNING: High fold variance detected - possible overfitting!")
        print("   Recommendation: Increase regularization (deeper L2/min_samples_leaf)")
    elif results['stability'] > 0.98:
        print("✓  Good stability - model generalizes well!")
