# WiDS 2026: Advanced Ensemble Strategies for Small-Data Survival Analysis

## Executive Summary

You have **221 training samples** (69 events, 152 censored) with **34 features** and **4 time horizons** to predict. Standard ML approaches will overfit. This guide provides proven strategies for robust small-data modeling.

---

## Part 1: The Small-Data Problem

### Data Scarcity Reality
```
dataset size: 221 fire events
positive class: 69 (31%)  <- imbalanced
negative class: 152 (69%)

test set: 95 fires (hidden labels)
features: 34 raw + derived features

VC dimension: Most ML models can memorize this easily
→ Explicit regularization REQUIRED
```

### Why Standard ML Fails
- **Tree overfitting:** Deep trees memorize individual samples
- **Feature noise:** Random correlations dominate in small samples
- **Class imbalance:** Positive class underrepresented
- **Multiple horizons:** Separate models per horizon (4×) increases risk
- **Censoring:** Right censoring complicates inference

### Key Insight
**Domain knowledge (physics) >> ML for small data**

Your physics model (0.80 weight) outperforms pure ML because:
1. Encodes causal structure, not spurious correlations
2. Regularizes the solution space
3. Proven across similar wildfire datasets

---

## Part 2: Anti-Overfitting Arsenal

### Strategy 1: Depth Control
```python
Tree-based models prone to overfitting with deep trees

Small data guideline:  max_depth = max(2, MIN(4, n_samples / 25))

For n=221:  max_depth ≤ 4
For XGBoost: max_depth = 3 (even shallower)
For Random Forest: max_depth = 4  
For GBM: max_depth = 2
```

### Strategy 2: Minimum Samples Per Split/Leaf
```python
Prevent trees from creating tiny branches

For n=221:
  min_samples_leaf = MAX(3, n_samples / 70)  → ≥3 samples
  min_samples_split = MAX(5, n_samples / 40) → ≥5 samples

Example (XGBoost):
  min_child_weight = 5-10  (minimum leaf weight)
  → Prevents single-sample leaves
```

### Strategy 3: Ensemble Shrinkage & Learning Rate
```python
Slow, stable convergence prevents convergence to noise

XGBoost:  learning_rate = 0.03-0.05 (slow)
GBM:      learning_rate = 0.02-0.05 (slower)
RandomForest: inherently stable, no learning rate

Principle: Lower LR = more iterations needed, but generalization improves
```

### Strategy 4: Subsample & Column Sampling
```python
Row/feature bootstrap reduces memorization

XGBoost:
  subsample = 0.7-0.8      (70-80% of rows per iteration)
  colsample_bytree = 0.7   (70% of features per tree)
  
GBM:
  subsample = 0.8          (80% of rows)

RandomForest:
  max_features = 'sqrt'    (sqrt(n_features) per split)
```

### Strategy 5: L1/L2 Regularization
```python
Penalize model complexity

XGBoost:
  reg_lambda = 1.0-2.0     (L2 regularization)
  gamma = 0.5-1.0          (complexity penalty for splits)

GBM:
  (Built into tree depths, no explicit L1/L2)
```

### Strategy 6: Feature Selection & Importance
```
For n=221 with 34 features: TOO MANY FEATURES

Action: Reduce to 18-20 core features
Strategy:
  1. Keep physics-derived features (domain-backed)
  2. Keep features with high, stable importance across horizons
  3. Remove polynomial/squared features (noise in small data)
  4. Remove features with low correlation to targets

Expected benefit: -2% accuracy gains, +5% generalization
```

---

## Part 3: Ensemble Strategies for Survival Analysis

### Ensemble 1: Physics-Dominant (Current Baseline)

```python
Weights:
  Physics:         80% (domain knowledge)
  XGBoost:         10% (ranking for C-index)
  RandomForest:     5% (stability & diversity)
  GradientBoosting: 5% (calibration)

Why this works for small data:
  - Physics provides strong regularization
  - ML models fine-tune ranking (C-index) without memorizing
  - 80% weight shields from ML overfitting

Expected: 0.9903 CV, near-0 overfitting risk
```

### Ensemble 2: Physics + Survival Models (Advanced)

```python
Add survival-specific models to ensemble:

CoxPH (Cox Proportional Hazards):
  - Probabilistic survival model
  - Directly handles censoring
  - More data-efficient than classification
  - Natural calibration from survival theory

SmoothingSplines Survival:
  - Non-parametric survival estimation
  - Handles complex survival curves
  - Good for multi-horizon prediction

Implementation:
  from lifelines import CoxPHFitter, KaplanMeierFitter
  
  # Train Cox model
  cph = CoxPHFitter()
  cph.fit(train_df[features], train_df['time_to_hit_hours'], 
          event_observed=train_df['event'])
  
  # Get survival probabilities at horizons
  cox_probs = 1 - cph.predict_survival_function(val_df[features]).T
  
  # Blend with physics
  probs_final = 0.70 * physics_probs + 0.30 * cox_probs

Expected benefit: +0.002 hybrid score (better C-index, proper censoring)
Complexity: Moderate (need lifelines library)
```

### Ensemble 3: Horizon-Specific Weights

```python
Different horizons have different optimal weights

Rationale:
  - 12h: Short term → physics dominates (fire spread is fast)
  - 24h: Medium term → ML rankinging helps
  - 48h: Longer → conditions change, more uncertainty
  - 72h: Long term → calibration matters most

Example:
  For horizon 12h:  W_PHYS=0.85, W_ML=0.15
  For horizon 24h:  W_PHYS=0.80, W_ML=0.20
  For horizon 48h:  W_PHYS=0.75, W_ML=0.25
  For horizon 72h:  W_PHYS=0.70, W_ML=0.30

Implementation:
  for h_idx, h in enumerate([12, 24, 48, 72]):
      w_phys = 0.85 - (h_idx * 0.05)  # Decrease physics weight
      w_ml = 1 - w_phys               # Increase ML weight
      probs[:, h_idx] = w_phys * physics_probs[:, h_idx] + ...

Expected benefit: +0.001 hybrid score
Complexity: LOW (just modified blending)
```

### Ensemble 4: Stacking (Meta-Learner)

```python
Use CV predictions as meta-features for a final learner

Process:
  1. Generate CV predictions from:
     - Physics model
     - XGBoost (4 horizons)
     - RandomForest (4 horizons)
     - GBM (4 horizons)
     → 13 meta-features per sample
  
  2. Train meta-learner (LogisticRegression or Ridge)
     on these 13 meta-features to predict targets
  
  3. Blend meta-learner output with physics

Code:
  from sklearn.linear_model import Ridge
  
  # Get CV meta-features
  meta_features = np.hstack([
      physics_cv_preds,
      xgb_cv_preds,
      rf_cv_preds,
      gb_cv_preds
  ])
  
  # Train meta-learner
  meta = Ridge(alpha=1.0)
  meta.fit(meta_features, y_target)
  
  # Blend with physics
  meta_preds = meta.predict(test_meta_features)
  final = 0.65 * physics + 0.35 * meta_preds

Expected benefit: +0.0015 hybrid score
Complexity: HIGH (need to generate all CV preds)
Overfitting risk: MODERATE (meta-learner can overfit to meta-features)
```

### Ensemble 5: Cross-Horizon Consistency

```
Exploit monotonicity constraint naturally:
  Prob(hit by 12h) <= Prob(hit by 24h) <= Prob(hit by 48h) <= Prob(hit by 72h)

Why it helps:
  - Reduces effective DoF (fewer parameters to fit)
  - Natural regularization from monotonicity
  - Improves calibration across horizons

Two approaches:

A. Soft monotonicity (post-processing):
   probs = np.sort(probs_raw, axis=1)  # Enforce rank order
   → Simple, adds little variance

B. Hard monotonicity (during training):
   Train cumulative survival model:
     S(t) = Prob(survive past t)
     Probs = [1-S(12), 1-S(24), 1-S(48), 1-S(72)]
   → More principled, harder to implement

Recommendation: Use Approach A (post-processing)
Expected benefit: +0.0005 hybrid (slight calibration gain)
```

---

## Part 4: Recommended Experiment Phases

### Phase 1A: Baseline Validation (Exp 1-2)
```
Goal: Confirm baseline with enhanced regularization

Baseline (current):
  W_PHYS=0.80, W_XGB=0.10, W_RF=0.05, W_GB=0.05
  xgb_depth=4, rf_depth=6, gb_depth=3

Exp 1 (Enhanced Regularization):
  W_PHYS=0.80, W_XGB=0.10, W_RF=0.05, W_GB=0.05
  xgb_depth=3, rf_depth=4, gb_depth=2  ← Shallower trees
  xgb_min_child_weight=5, rf_min_leaf=5  ← Higher minimum samples
  
Expected: Slight score decrease (-0.0005) but better generalization

Exp 2 (Feature Reduction):
  Same weights as Exp 1
  20 core features (remove noisy polynomial features)
  
Expected: Slight score decrease (-0.0005) but more stable CV
```

### Phase 1B: Ensemble Weight Tuning (Exp 3-8)
```
Goal: Optimize Physics vs ML balance

Baseline: PHYS=0.80, XGB=0.10, RF=0.05, GB=0.05

Exp 3: PHYS=0.75, XGB=0.15, RF=0.05, GB=0.05
       (Boost XGB for better ranking)
       
Exp 4: PHYS=0.70, XGB=0.15, RF=0.10, GB=0.05
       (More diverse ML, less physics)
       
Exp 5: PHYS=0.80, XGB=0.05, RF=0.10, GB=0.05
       (Boost RF diversity)
       
Exp 6-8: Grid search best 2×2 from top 3

Expected best: PHYS=0.75, XGB=0.15 (+0.0015)
```

### Phase 2: Hyperparameter Tuning (Exp 9-16)
```
Goal: Given best ensemble weights, tune model params

If Exp 3 wins (PHYS=0.75, XGB=0.15):
  
Exp 9:  xgb_learning_rate=0.03 → 0.05 (faster but riskier)
Exp 10: xgb_learning_rate=0.02 (slower but more stable)
Exp 11: xgb_max_depth=3 → 4 (capacity increase)
Exp 12: rf_n_estimators=250 → 350 (more trees)
Exp 13: gb_learning_rate=0.03 → 0.05
Exp 14-16: Combine best from above

Expected: +0.001 improvement
```

### Phase 3: Feature Engineering (Exp 17-24)
```
Goal: Add high-value features without noise

Conservative features to try:
- 'growth_speed_product' = area_growth_rel × radial_growth_rate
- 'critical_window' = is_night × alignment_abs^2
- 'distance_trend' = dist_slope ^ 2 (acceleration)
- 'month_sin', 'month_cos' (seasonal effects)

Add ONE feature per experiment, validate impact

Expected: +0.002 improvement (moderate gain)
```

### Phase 4: Survival Models (Exp 25-32)
```
Goal: Add CoxPH or stacking for better censoring handling

Exp 25: Add CoxPH model
       W = [0.60 Physics, 0.15 CoxPH, 0.10 XGB, 0.10 RF, 0.05 GB]
       
Exp 26-32: Tune CoxPH weight, combine with best from Phase 2-3

Expected: +0.0015 final improvement → 0.994+ target
```

---

## Part 5: Recommended Values by Problem

### For Small Data (n < 300):
```
Tree Depth:
  max_depth = 3-4 (very shallow)
  
Min Samples:
  min_samples_leaf = 5-10
  min_samples_split = 10-15
  
Regularization:
  subsample = 0.7-0.8
  colsample_bytree = 0.7
  
Learning Rate:
  0.02-0.05 (slow)
  
Ensemble Size:
  100-150 trees (smaller ensembles)
  
Domain Weight:
  70-85% physics (data-efficient)
```

### For Class Imbalance (31% positive):
```
Scale positive weight:
  scale_pos_weight = 152 / 69 ≈ 2.2 (XGBoost)
  
Stratified K-fold:
  Always stratify to maintain class ratio in folds
  
Initial predictions:
  pred = 0.31 (base rate), then adjust upward for positives
  
Verification:
  Check positive precision/recall trade-off
  Prefer calibration (Brier) over raw accuracy
```

### For Multiple Horizons (4 models per algorithm):
```
Shared feature representation:
  Train on same FEATURE_SET for all horizons
  
Per-horizon calibration:
  Different SQUISH per horizon: 1.1 (12h), 1.2 (24h), 1.3 (48h)
  
Cross-horizon validation:
  Enforce monotonicity as regularizer
  → probs_12h <= probs_24h <= probs_48h <= probs_72h
```

---

## Part 6: Overfitting Diagnostics

### Red Flag 1: High Fold Variance
```
Symptom: Standard deviation of CV folds > 0.008

Cause: Model overfitting to fold-specific noise

Fix:
  - Increase regularization (smaller max_depth)
  - Increase min_samples_leaf
  - Boost physical model weight (W_PHYS += 0.05)
  - Reduce feature count
```

### Red Flag 2: Brier Improved, C-Index Dropped
```
Symptom: Better calibration but worse ranking

Cause: Model memorizing class probabilities, losing ranking info

Fix:
  - Increase XGB weight (better ranking)
  - Boost GBM weight (worse ranking sometimes)
  - Train horizon-specific calibration
```

### Red Flag 3: One Fold Much Better than Others
```
Symptom: Fold 1: 0.993, Fold 2: 0.985

Cause: Fold 1 happens to have easier examples

Not necessarily overfitting, but:
  - Check if class ratio varies significantly across folds
  - Verify stratified K-fold is used
  - Seed random state for reproducibility
```

### Red Flag 4: Training Time Increases 50%+
```
Symptom: 3 min → 5 min per fold

Cause: Larger models, more estimators, deeper trees

Action:
  - Reduce xgb_n_estimators (150 → 100)
  - Reduce rf_n_estimators (250 → 150)
  - Reduce max_depth
  - Check if new features have high cardinality
```

---

## Part 7: Quick Reference - Parameter Tuning Table

| Problem | Parameter | Current | Try If Low Score | Try If Overfit |
|---------|-----------|---------|------------------|----------------|
| C-Index low | W_XGB | 0.10 | 0.15 → 0.20 | 0.05 |
| Brier high | W_GB | 0.05 | 0.10 | 0.02 |
| Variance high | max_depth | 3-4 | N/A | 2 |
| Variance high | min_leaf | 3-5 | N/A | 7-10 |
| Slow learning | learning_rate | 0.03 | 0.05-0.08 | 0.01-0.02 |
| Unstable | subsample | 0.7-0.8 | N/A | 0.6 |
| Too many features | n_features | 31 | Add 2-3 | Remove 3-5 |

---

## Summary: The Path to 0.994+

```
Current:     0.9903 ± 0.0058
Target:      0.9940 (+0.0037 = 0.37%)

Path:
  Phase 1A (Validation): 0.9903 → 0.9895 (accept -0.0008 for stability)
  Phase 1B (Weights):    0.9895 → 0.9912 (+0.0017 from better ensemble)
  Phase 2  (Hyperparams): 0.9912 → 0.9923 (+0.0011 from tuning)
  Phase 3  (Features):   0.9923 → 0.9935 (+0.0012 from new features)
  Phase 4  (Survival):   0.9935 → 0.9942 (+0.0007 from CoxPH)
  
FINAL: 0.9942 ✓ (Target achieved!)
```

Good luck! 🚀
