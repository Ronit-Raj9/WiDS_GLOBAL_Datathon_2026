import pandas as pd
import numpy as np
from scipy.special import erf
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import CalibratedClassifierCV
from lifelines import CoxPHFitter
import warnings
warnings.filterwarnings('ignore')

# --- 1. Load and Advanced Preprocessing ---
def load_and_preprocess():
     # Use local data path
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    def transform(df):
        df = df.copy()
        # Fill missing values
        df['closing_speed_m_per_h'] = df['closing_speed_m_per_h'].fillna(df['closing_speed_m_per_h'].median())
        df['radial_growth_rate_m_per_h'] = df['radial_growth_rate_m_per_h'].fillna(df['radial_growth_rate_m_per_h'].median())

        # Clip outliers
        df['area_growth_rel_0_5h'] = df['area_growth_rel_0_5h'].clip(upper=df['area_growth_rel_0_5h'].quantile(0.99))

        # Core physics features (from original)
        df['persistence'] = np.log1p(df['num_perimeters_0_5h'])
        df['dist'] = (df['dist_min_ci_0_5h'] - 5000).clip(lower=10)

        # Core Physics Meta-Feature
        df['v_stable'] = (df['closing_speed_m_per_h'] + df['radial_growth_rate_m_per_h']).clip(lower=1.0) * \
                         (df['alignment_abs']**1.5) * df['persistence']

        # Interaction Features (original)
        df['eta_kinetic'] = df['dist'] / df['v_stable'].clip(lower=0.1)
        df['density_metric'] = df['area_first_ha'] / (df['dist'] + 1)
        df['speed_alignment'] = df['closing_speed_m_per_h'] * df['alignment_abs']

        # ===== ENHANCED FEATURE ENGINEERING =====
        # Additional physics-based features
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

        # Polynomial features for key predictors
        df['dist_squared'] = df['dist'] ** 2 / 1e10
        df['v_stable_squared'] = df['v_stable'] ** 2
        df['alignment_squared'] = df['alignment_abs'] ** 2

        # Ratio features
        df['closing_to_centroid_ratio'] = df['closing_speed_m_per_h'] / (df['centroid_speed_m_per_h'].clip(lower=0.1) + 1)

        return df

    return transform(train), transform(test)

train_df, test_df = load_and_preprocess()

# --- 2. Physics & Metric Engines ---
def get_physics_probs(df, train_ref):
    S_C, P_W, SIGMA, BIAS = 1.83455702, 0.00975066, -1.11494169, -1.17324153
    HORIZONS = [12, 24, 48, 72]
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

def calculate_metric(probs, times, events):
    """Calculate the competition metric (Hybrid Score)"""
    b_weights = [0.20, 0.40, 0.40]
    b_total = 0
    for i, h in enumerate([12, 24, 48, 72]):
        if i == 0: continue
        mask = (events == 1) | (times >= h)
        y_true = ((times <= h) & (events == 1)).astype(int)[mask]
        b_total += b_weights[i-1] * brier_score_loss(y_true, probs[mask, i])
    c_score = roc_auc_score(((times <= 72) & (events == 1)).astype(int), probs[:, 3])
    return 0.10 * c_score + 0.90 * (1 - b_total)

    # --- 4. Final Submission Generation ---
print("\n=== Generating Enhanced Submission ===")

# Physics predictions on test
p_phys_test = get_physics_probs(test_df, train_df)

# Train all models on full training data
p_xgb_test, p_rf_test, p_gb_test = np.zeros_like(p_phys_test), np.zeros_like(p_phys_test), np.zeros_like(p_phys_test)

for i, h in enumerate([12, 24, 48, 72]):
    y_tr_h = ((train_df['time_to_hit_hours'] <= h) & (train_df['event'] == 1)).astype(int)

    # XGBoost
    xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42, verbosity=0,
                       subsample=0.8, colsample_bytree=0.8)
    xgb.fit(train_df[X_features], y_tr_h)
    p_xgb_test[:, i] = xgb.predict_proba(test_df[X_features])[:, 1]

    # Random Forest
    rf = RandomForestClassifier(n_estimators=250, max_depth=6, random_state=42, min_samples_leaf=3)
    rf.fit(train_df[X_features], y_tr_h)
    p_rf_test[:, i] = rf.predict_proba(test_df[X_features])[:, 1]

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
    gb.fit(train_df[X_features], y_tr_h)
    p_gb_test[:, i] = gb.predict_proba(test_df[X_features])[:, 1]

# Final ensemble
final_probs = (W_PHYS * p_phys_test) + (W_XGB * p_xgb_test) + (W_RF * p_rf_test) + (W_GB * p_gb_test)
final_probs = np.power(final_probs, SQUISH)

# Apply adjustments
final_probs[test_df['num_perimeters_0_5h'] == 0] *= 0.95

# Close fires adjustment
close_mask = test_df['dist_min_ci_0_5h'] < 5000
final_probs[close_mask] = np.clip(final_probs[close_mask], 0.01, 0.99)

# Create submission
submission = pd.DataFrame({
    'event_id': test_df['event_id'],
    'prob_12h': final_probs[:, 0],
    'prob_24h': final_probs[:, 1],
    'prob_48h': final_probs[:, 2],
    'prob_72h': final_probs[:, 3]
})

# Ensure monotonicity (prob_12h <= prob_24h <= prob_48h <= prob_72h)
submission[['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']] = np.sort(
    submission[['prob_12h', 'prob_24h', 'prob_48h', 'prob_72h']].values, axis=1
)

# Save submission
submission.to_csv('submission_enhanced.csv', index=False)
print("Saved: submission_enhanced.csv")
print(f"\nSubmission shape: {submission.shape}")
print(f"CV Score: {np.mean(cv_scores):.6f}")
print(f"\nSample predictions:\n{submission.head(10)}")

# --- 3. Enhanced Feature Set & Triple Ensemble CV with Additional Models ---
# Updated features with enhanced engineering
X_features = [
    # Original features
    'dist', 'v_stable', 'eta_kinetic', 'alignment_abs', 'area_first_ha', 'density_metric', 'speed_alignment',
    # New features
    'kinetic_energy', 'approach_rate', 'lateral_motion', 'dist_to_initial_ratio', 'dist_std_normalized',
    'growth_efficiency', 'radial_to_area_ratio', 'is_night', 'is_weekend', 'hour_sin', 'hour_cos',
    'dist_squared', 'v_stable_squared', 'alignment_squared', 'closing_to_centroid_ratio',
    # Keep some original features that are important
    'num_perimeters_0_5h', 'closing_speed_m_per_h', 'radial_growth_rate_m_per_h', 'centroid_speed_m_per_h',
    'area_growth_rel_0_5h', 'dist_slope_ci_0_5h', 'dist_change_ci_0_5h', 'along_track_speed'
]
y_target = ((train_df['time_to_hit_hours'] <= 72) & (train_df['event'] == 1)).astype(int)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []

# ===== OPTIMIZED ENSEMBLE WEIGHTS =====
W_PHYS = 0.80  # Slightly reduced to make room for GB
W_XGB = 0.10
W_RF = 0.05
W_GB = 0.05   # New: Gradient Boosting
SQUISH = 1.20

print(f"=== Running Enhanced Triple-Threat CV ===")
print(f"Weights - Phys: {W_PHYS}, XGB: {W_XGB}, RF: {W_RF}, GB: {W_GB}")

for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_target)):
    df_tr, df_val = train_df.iloc[train_idx], train_df.iloc[val_idx]
    y_time_val, y_event_val = df_val['time_to_hit_hours'].values, df_val['event'].values

    # 1. Physics Model
    p_phys = get_physics_probs(df_val, df_tr)

    # 2. ML Predictions (XGB, RF, GB)
    p_xgb, p_rf, p_gb = np.zeros_like(p_phys), np.zeros_like(p_phys), np.zeros_like(p_phys)

    for i, h in enumerate([12, 24, 48, 72]):
        y_tr_h = ((df_tr['time_to_hit_hours'] <= h) & (df_tr['event'] == 1)).astype(int)

        # XGBoost
        xgb = XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.03, random_state=42, verbosity=0,
                           subsample=0.8, colsample_bytree=0.8)
        xgb.fit(df_tr[X_features], y_tr_h)
        p_xgb[:, i] = xgb.predict_proba(df_val[X_features])[:, 1]

        # Random Forest
        rf = RandomForestClassifier(n_estimators=250, max_depth=6, random_state=42, min_samples_leaf=3)
        rf.fit(df_tr[X_features], y_tr_h)
        p_rf[:, i] = rf.predict_proba(df_val[X_features])[:, 1]

        # Gradient Boosting (new)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42)
        gb.fit(df_tr[X_features], y_tr_h)
        p_gb[:, i] = gb.predict_proba(df_val[X_features])[:, 1]

    # 3. Enhanced Ensemble Blend
    blend_val = (W_PHYS * p_phys) + (W_XGB * p_xgb) + (W_RF * p_rf) + (W_GB * p_gb)

    # 4. Calibration & Penalty
    final_val = np.power(blend_val, SQUISH)

    # Apply perimeters penalty (fires with no perimeter growth are less predictable)
    final_val[df_val['num_perimeters_0_5h'] == 0] *= 0.95

    # Apply distance-based adjustment (very close fires are more certain)
    close_mask = df_val['dist_min_ci_0_5h'] < 5000
    final_val[close_mask] = np.clip(final_val[close_mask], 0.01, 0.99)

    fold_score = calculate_metric(final_val, y_time_val, y_event_val)
    cv_scores.append(fold_score)
    print(f"Fold {fold+1}: {fold_score:.6f}")

print(f"\n=== Enhanced Mean CV: {np.mean(cv_scores):.6f} ===")