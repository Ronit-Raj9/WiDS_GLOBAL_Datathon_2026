"""
WiDS Datathon 2026 - Evaluation Harness (READ-ONLY)

Handles:
- Data loading from CSV
- Base feature engineering
- Evaluation metrics (C-index, Brier, Hybrid Score)
- CV fold creation

DO NOT MODIFY - This is the ground truth evaluation.
"""

import pandas as pd
import numpy as np
from scipy.special import erf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss, roc_auc_score
import os
import sys

# ============================================================================
# Constants
# ============================================================================

# Get the parent directory (root of project)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
HORIZONS = [12, 24, 48, 72]


# ============================================================================
# Data Loading
# ============================================================================

def load_data():
    """Load train and test data."""
    train = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
    test = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
    return train, test


# ============================================================================
# Base Feature Engineering (Fixed Foundation)
# ============================================================================

def engineer_base_features(df):
    """
    Core feature engineering applied to all experiments.
    
    Do NOT modify this function - use it as the foundation in wids_train.py.
    You can add NEW features in wids_train.py on top of these.
    """
    df = df.copy()
    
    # Handle missing values
    df['closing_speed_m_per_h'] = df['closing_speed_m_per_h'].fillna(
        df['closing_speed_m_per_h'].median()
    )
    df['radial_growth_rate_m_per_h'] = df['radial_growth_rate_m_per_h'].fillna(
        df['radial_growth_rate_m_per_h'].median()
    )
    
    # Clip outliers
    df['area_growth_rel_0_5h'] = df['area_growth_rel_0_5h'].clip(
        upper=df['area_growth_rel_0_5h'].quantile(0.99)
    )
    
    # Core physics features
    df['persistence'] = np.log1p(df['num_perimeters_0_5h'])
    df['dist'] = (df['dist_min_ci_0_5h'] - 5000).clip(lower=10)
    df['v_stable'] = (
        df['closing_speed_m_per_h'] + df['radial_growth_rate_m_per_h']
    ).clip(lower=1.0) * (df['alignment_abs']**1.5) * df['persistence']
    
    # Interaction features
    df['eta_kinetic'] = df['dist'] / df['v_stable'].clip(lower=0.1)
    df['density_metric'] = df['area_first_ha'] / (df['dist'] + 1)
    df['speed_alignment'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    
    # Enhanced features
    df['kinetic_energy'] = df['area_first_ha'] * df['v_stable']**2
    df['approach_rate'] = df['closing_speed_m_per_h'] * df['alignment_abs']
    df['lateral_motion'] = df['centroid_speed_m_per_h'] * (1 - df['alignment_abs'])
    
    # Distance features
    df['dist_to_initial_ratio'] = df['dist_min_ci_0_5h'] / (df['dist_min_ci_0_5h'].mean() + 1)
    df['dist_std_normalized'] = df['dist_std_ci_0_5h'] / (df['dist_min_ci_0_5h'] + 1)
    
    # Growth features
    df['growth_efficiency'] = df['area_growth_rate_ha_per_h'] / (df['area_first_ha'] + 1)
    df['radial_to_area_ratio'] = df['radial_growth_rate_m_per_h'] / (np.sqrt(df['area_first_ha']) + 1)
    
    # Temporal features
    df['is_night'] = ((df['event_start_hour'] >= 20) | (df['event_start_hour'] <= 5)).astype(int)
    df['is_weekend'] = (df['event_start_dayofweek'] >= 5).astype(int)
    df['hour_sin'] = np.sin(2 * np.pi * df['event_start_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['event_start_hour'] / 24)
    
    # Polynomial features
    df['dist_squared'] = df['dist'] ** 2 / 1e10
    df['v_stable_squared'] = df['v_stable'] ** 2
    df['alignment_squared'] = df['alignment_abs'] ** 2
    
    # Ratio features
    df['closing_to_centroid_ratio'] = df['closing_speed_m_per_h'] / (
        df['centroid_speed_m_per_h'].clip(lower=0.1) + 1
    )
    
    return df


# ============================================================================
# Physics-Based Model (Fixed)
# ============================================================================

def get_physics_probs(df, train_ref):
    """
    Calculate physics-based probabilities.
    
    Parameters calibrated from domain knowledge.
    DO NOT MODIFY - use as ensemble component with weight W_PHYS.
    """
    S_C, P_W, SIGMA, BIAS = 1.83455702, 0.00975066, -1.11494169, -1.17324153
    LK_FAST, LK_SLOW = [12.0, 15.0, 16.0, 16.0], [8.0, 10.0, 8.0, 12.0]
    QUANTILES = [0.85, 0.65, 0.65, 0.65]
    
    probs = np.zeros((len(df), 4))
    t_kinetic = df['dist'] / df['v_stable'].clip(lower=1.0)
    mu_log = np.log((t_kinetic * S_C).clip(0.1, 1000)) - P_W * np.log1p(df['area_growth_rel_0_5h'])
    sig = np.exp(SIGMA).clip(0.1, 5)
    
    for i, t in enumerate(HORIZONS):
        v_thresh = train_ref['v_stable'].quantile(QUANTILES[i])
        is_fast = (df['v_stable'] > v_thresh).values
        z = (np.log(t) - mu_log) / (sig * np.sqrt(2))
        p_arr = 0.5 * (1 + erf(z))
        lk_vec = np.where(is_fast, LK_FAST[i], LK_SLOW[i])
        probs[:, i] = 1.0 / (1.0 + np.exp(-(lk_vec * (p_arr - 0.5) + BIAS)))
    
    return probs


# ============================================================================
# Evaluation Metrics (Fixed - Ground Truth)
# ============================================================================

def calculate_c_index(times, probabilities, events):
    """C-index: measures ranking quality (30% of final score)."""
    try:
        risk_scores = -probabilities[:, 3]  # 72h horizon
        c_idx = roc_auc_score(((times <= 72) & (events == 1)).astype(int), -risk_scores)
        return c_idx
    except:
        return 0.5


def calculate_weighted_brier_score(times, probabilities, events):
    """Weighted Brier: measures calibration (70% of final score)."""
    brier_weights = [0.3, 0.4, 0.3]  # 24h, 48h, 72h
    b_total = 0
    
    for i, h in enumerate([24, 48, 72]):
        mask = (events == 1) | (times >= h)
        y_true = ((times <= h) & (events == 1)).astype(int)[mask]
        
        if len(y_true) > 0:
            bs = brier_score_loss(y_true, probabilities[mask, i])
            b_total += brier_weights[i - 1] * bs
    
    return b_total


def calculate_hybrid_score(times, probabilities, events):
    """
    Primary evaluation metric.
    Hybrid Score = 0.3×C-index + 0.7×(1 - Weighted Brier)
    """
    c_idx = calculate_c_index(times, probabilities, events)
    brier = calculate_weighted_brier_score(times, probabilities, events)
    hybrid = 0.3 * c_idx + 0.7 * (1.0 - brier)
    
    return {
        'hybrid_score': hybrid,
        'c_index': c_idx,
        'brier_score': brier
    }


# ============================================================================
# Cross-Validation Setup
# ============================================================================

def get_cv_splits(train_df, n_splits=5, random_state=42):
    """Generate 5-fold stratified splits."""
    y = ((train_df['time_to_hit_hours'] <= 72) & (train_df['event'] == 1)).astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, val_idx in skf.split(train_df, y):
        splits.append((train_idx, val_idx))
    
    return splits


# ============================================================================
# Utilities
# ============================================================================

def ensure_monotonicity(probs):
    """Enforce: prob_12h <= prob_24h <= prob_48h <= prob_72h."""
    return np.sort(probs, axis=1)


if __name__ == '__main__':
    # Quick check
    train, test = load_data()
    print(f"✓ Train shape: {train.shape}")
    print(f"✓ Test shape: {test.shape}")
    
    train = engineer_base_features(train)
    print(f"✓ Features engineered: {train.shape[1]} columns")
    
    splits = get_cv_splits(train)
    print(f"✓ CV splits created: {len(splits)} folds")
    
    print("\nWiDS evaluation harness ready for import.")
