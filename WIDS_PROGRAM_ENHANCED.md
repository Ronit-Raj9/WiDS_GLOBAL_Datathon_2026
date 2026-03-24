# WiDS 2026 Autonomous Research Program - Enhanced for Small Data

## Overview

**Mission:** Build robust survival analysis model for wildfire evacuation prediction

**Baseline:** 0.9903 (5-fold CV, physics-dominant ensemble)
**Target:** 0.9940+ (top 20 leaderboard position)
**Data regime:** Small (n=221, n_pos=69, n_neg=152)
**Key challenge:** Avoid overfitting with 34 features and only 69 positive examples

---

## System Architecture

### Three Core Files (Locked-Edit-Locked Pattern)

```
├── wids_prepare.py           [LOCKED] ✅ Evaluation harness
│   └─ Metrics, CV setup, data loading, physics model
│
├── wids_train_enhanced.py    [AGENT MODIFIES] 🔧 Training
│   └─ Ensemble weights, hyperparams, features, regularization
│
├── wids_experiment_runner.py [LOCKED] ✅ Orchestrator
│   └─ Runs training, logs results, decision making
│
└── ADVANCED_STRATEGIES.md    [REFERENCE] 📖 Small-data tactics
    └─ Theory, experiments, diagnostics
```

**Key principle:** Only ONE file modified by agent → reproducible, traceable experiments

---

## What You CAN Modify (in `wids_train_enhanced.py`)

### 1. **Ensemble Weights** (Highest Priority for 0.9903→0.9915)

```python
W_PHYS = 0.75      # Physics model (domain knowledge)
W_XGB = 0.15       # XGBoost (ranking for C-index)
W_RF = 0.05        # Random Forest (diversity)
W_GB = 0.05        # Gradient Boosting (calibration)
```

**Why physics dominates with small data:**
- 221 samples = limited signal for pure ML
- Physics encodes causal physics of fire spread
- Prevents overfitting to noise

**Weight tuning strategy:**
```
Baseline: PHYS=0.75, XGB=0.15, RF=0.05, GB=0.05

Try these sequences:

Direction 1 (Boost XGB for C-index ranking):
  1. PHYS=0.70, XGB=0.20, RF=0.05, GB=0.05
  2. PHYS=0.70, XGB=0.15, RF=0.10, GB=0.05
  3. PHYS=0.65, XGB=0.25, RF=0.05, GB=0.05

Direction 2 (Diversify ML ensemble):
  1. PHYS=0.75, XGB=0.10, RF=0.10, GB=0.05
  2. PHYS=0.70, XGB=0.10, RF=0.10, GB=0.10

Direction 3 (Boost GBM for calibration):
  1. PHYS=0.75, XGB=0.10, RF=0.05, GB=0.10
  2. PHYS=0.70, XGB=0.10, RF=0.05, GB=0.15
```

**Decision rule:**
- Keep if Hybrid >= +0.0010
- If C-Index improved but Brier worse → indicates good ranking but poor calibration (acceptable if net positive)
- If both improved → excellent direction
- If both worsened → discard

---

### 2. **Regularization Parameters** (Prevents Overfitting)

#### 2A: Tree Depths (Shallow is Better)

```python
class ModelParams:
    # XGBoost (fast learner, needs strong regularization)
    xgb_max_depth = 3          # Keep very shallow for n=221
                               # Try: 2 (more overfit-safe)
                               #      4 (if underfitting detected)
    
    # Random Forest (naturally stable, but still needs control)
    rf_max_depth = 4           # Try: 3 (safer)
                               #      5 (if underfitting)
    
    # Gradient Boosting (iterative learning, sensitive to depth)
    gb_max_depth = 2           # THE SHALLOWEST
                               # Try: 1 (if still overfitting)
                               #      3 (if underfitting bad)
```

**Experiment pattern:**
```
Exp 1: Reduce all depths (safest for overfitting):
  xgb_max_depth = 2
  rf_max_depth = 3
  gb_max_depth = 2
  
  Expected: Score drop -0.0005 but better generalization
  
Exp 2: If underfitting (train-val gap > 0.015), increase depths:
  xgb_max_depth = 4
  rf_max_depth = 5
  gb_max_depth = 3
  
  Expected: Score +0.0010, higher variance
```

#### 2B: Minimum Samples (Prevent Tiny Leaves)

```python
class ModelParams:
    # XGBoost (min weight of samples in leaf)
    xgb_min_child_weight = 5   # Try: 3 (less restrictive)
                               #      10 (more regularized)
    
    # Random Forest
    rf_min_samples_leaf = 5    # Minimum samples per leaf
    rf_min_samples_split = 10  # Minimum to create split
    
    # Gradient Boosting
    gb_min_samples_leaf = 5    # Try: 3 or 7
    gb_min_samples_split = 10
```

**Experiment pattern:**
```
Exp: Increase minimum samples (aggressive regularization):
  xgb_min_child_weight = 10
  rf_min_samples_leaf = 10
  rf_min_samples_split = 20
  gb_min_samples_leaf = 10
  gb_min_samples_split = 20
  
  Expected: Stable CV, possible -0.0010 score (trade-off)
```

#### 2C: Row/Column Sampling

```python
class ModelParams:
    # XGBoost
    xgb_subsample = 0.7        # 70% of rows per iteration
    xgb_colsample_bytree = 0.7 # 70% of features per tree
    
    # Gradient Boosting
    gb_subsample = 0.8         # 80% of rows
    
    # RandomForest
    rf_max_features = 'sqrt'   # sqrt(34) ≈ 6 features per split
```

**Experiment pattern:**
```
Exp: More aggressive sampling (reduce to 60%):
  xgb_subsample = 0.6
  xgb_colsample_bytree = 0.6
  gb_subsample = 0.7
  
  Expected: Higher variance, maybe +0.0005 from reduced overfitting
```

#### 2D: Learning Rate

```python
class ModelParams:
    # XGBoost: Currently 0.05 (moderate)
    xgb_learning_rate = 0.05   # Try: 0.02 (slower, more stable)
                               #      0.08 (faster, riskier)
    
    # Gradient Boosting: Currently 0.03 (conservative)
    gb_learning_rate = 0.03    # Try: 0.01 (very stable)
                               #      0.05 (more aggressive)
```

**Experiment pattern:**
```
Exp 1 (Slower learning):
  xgb_learning_rate = 0.02
  gb_learning_rate = 0.01
  xgb_n_estimators = 200  (more iterations needed)
  gb_n_estimators = 150
  
  Expected: Slower training, higher fold stability, maybe +0.0005

Exp 2 (Faster learning, more capacity):
  xgb_learning_rate = 0.08
  gb_learning_rate = 0.05
  Keep n_estimators same
  
  Expected: Faster, potential overfitting, risk of -0.0010
```

---

### 3. **Feature Selection** (Reduce Noise)

Current: 18 features (already reduced from 31)

```python
FEATURE_SET = [
    # MUST-KEEP (physics-backed)
    'dist', 'v_stable', 'alignment_abs', 'area_first_ha',
    
    # HIGH-VALUE INTERACTIONS
    'eta_kinetic', 'density_metric', 'speed_alignment',
    'kinetic_energy', 'approach_rate',
    
    # DISTANCE & TIME
    'dist_to_initial_ratio', 'dist_slope_ci_0_5h', 'closing_speed_m_per_h',
    
    # TEMPORAL & GROWTH
    'is_night', 'hour_sin', 'hour_cos',
    'area_growth_rel_0_5h', 'radial_growth_rate_m_per_h',
]

# Features NOT in this set (removed for safety):
# - 'v_stable_squared' (noise in small data)
# - 'alignment_squared' (redundant with alignment_abs)
# - 'dist_squared' (high-order polynomial)
# - 'closing_to_centroid_ratio' (derived, possibly noisy)
```

**Experiment pattern:**

```
Exp 1 (ADD feature - conservative):
  Add ONE new feature with clear physical meaning:
  
  Candidates:
  - 'wind_dynamic' = closing_speed × alignment_squared^2 (stronger alignment effect)
  - 'critical_window' = is_night × alignment_abs (nighttime alignment focus)
  - 'growth_trend' = area_growth_rel × radial_growth_rate
  
  Expected: +0.0005 if good signal, -0.0002 if noise

Exp 2 (REMOVE feature - if space constrained):
  If adding features degraded: Remove lowest-importance feature
  
  Candidates to remove:
  - 'hour_cos' (check: is both sin/cos necessary?)
  - 'dist_to_initial_ratio' (redundant with dist_slope?)
  
  Expected: Cleaner model, maybe -0.0003 score but better stability
```

**Feature validation checklist:**
```
Before each experiment:
  ✓ All features exist in train_df and test_df
  ✓ No NaN values in any feature
  ✓ Feature names match exactly (case-sensitive!)
  ✓ Feature has non-zero variance
```

---

### 4. **Calibration & Post-Processing**

```python
# Calibration power (squish factor)
SQUISH = 1.2  # Current value (>1 = squish to extremes)

# Post-processing in train_and_evaluate():
probs_final[df_val['num_perimeters_0_5h'] == 0] *= 0.95

close_mask = df_val['dist_min_ci_0_5h'] < 5000
probs_final[close_mask] = np.clip(probs_final[close_mask], 0.01, 0.99)
```

**Experiment pattern:**

```
Exp 1 (Adjust SQUISH):
  SQUISH = 1.1  (less extreme squishing)
  
  Expected: Higher Brier (less confident), maybe same C-index
  Decision: Keep if Hybrid improves

Exp 2 (Adjust uncertainty for low-data fires):
  probs_final[df_val['num_perimeters_0_5h'] == 0] *= 0.90  (was 0.95)
  
  Expected: Less confident on uncertain fires, better Brier

Exp 3 (Adjust close-fire bounds):
  probs_final[close_mask] = np.clip(probs_final[close_mask], 0.05, 0.95)
  
  Expected: Slightly more uncertain even at close range
```

---

## What You CANNOT Modify

❌ `wids_prepare.py` — Locked evaluation harness
❌ Physics probabilities — `get_physics_probs()` immutable
❌ CV structure — 5-fold StratifiedKFold(random_state=42) fixed
❌ Metrics calculation — C-index, Brier, Hybrid immutable
❌ Data paths — CSV loading from `data/` fixed

---

## Experiment Workflow

### Step 1: Propose Hypothesis

Always start with **WHY**:

```
Example 1:
  Observation: Baseline hybrid=0.9903, but C-index=0.8521 (30% weight)
  Hypothesis: Physics strong but ML ranking weak → need better C-index
  Action: Increase W_XGB from 0.10 to 0.15
  Expected: C-index +0.005 (0.8521→0.8571), Hybrid +0.001 (0.7×0 + 0.3×005)

Example 2:
  Observation: Fold variance 0.008 (high)
  Hypothesis: Trees too deep for n=221 → overfitting on fold-specific noise
  Action: Reduce xgb_max_depth from 4 to 3
  Expected: Fold std -0.003 (more stable), Hybrid -0.0005 (acceptable trade-off)

Example 3:
  Observation: Last 5 experiments no improvement
  Hypothesis: Feature set has noise, ML models stuck
  Action: Add 'critical_window' = is_night × alignment_abs^2
  Expected: Better nighttime fire ranking, +0.0010 Hybrid (speculative)
```

### Step 2: Make Changes (Max 3 per Cycle)

Edit `wids_train_enhanced.py`:

```python
# Option A: Change one weight
W_PHYS = 0.70      # was 0.75
W_XGB = 0.20       # was 0.15
W_RF = 0.05
W_GB = 0.05

# Option B: Change hyperparameter
xgb_max_depth = 3  # was 4

# Option C: Add feature
FEATURE_SET = [
    ... (existing features)
    'critical_window',  # NEW
]
```

**Constraint:** Max 3 changes per experiment maintains isolation of effects

### Step 3: Run Experiment

```bash
cd autoresearch/
python wids_experiment_runner.py "Exp #X: Brief description of what you changed"
```

**Wait ~3-5 minutes** for 5-fold CV to complete

### Step 4: Read Results

```
===== Output shows =====
Fold 1: hybrid=0.989201, c_idx=0.8524, brier=0.1743
Fold 2: hybrid=0.994320, c_idx=0.8701, brier=0.1621
Fold 3: hybrid=0.982110, c_idx=0.8310, brier=0.1850
Fold 4: hybrid=0.991864, c_idx=0.8589, brier=0.1752
Fold 5: hybrid=0.992786, c_idx=0.8601, brier=0.1741
---
MEAN: hybrid=0.990056 (was 0.9903, change=-0.0002)
Stability: 0.9842 (fold std=0.0051)
===== End output =====
```

**Key metrics to check:**
1. **Hybrid Score**: Did main metric improve?
2. **C-Index**: Ranking quality (30% weight)
3. **Brier Score**: Calibration (70% weight)
4. **Fold Std**: Variance (want < 0.008)
5. **Stability**: 1 - std/mean (want > 0.98)

### Step 5: Decide: Keep or Discard

```
IF hybrid >= baseline + 0.0010
  → ✅ KEEP
  → git commit -m "Exp #X: ..."

IF baseline - 0.0005 < hybrid < baseline + 0.0010
  → ⚠️  MARGINAL
  → Check fold stability and complexity
  → KEEP only if clearer/simpler code or stability >> baseline
  
IF hybrid < baseline - 0.0005
  → ❌ DISCARD
  → git reset --hard
  → Try different direction
  
IF fold std > 0.015 (huge variance)
  → ⚠️  WARNING: Likely overfitting
  → Immediately revert and add regularization
```

### Step 6: Log Results

Auto-logged to `results.tsv`:
```
baseline       0.9903   0.8521   0.1843      keep       147.3    Initial
exp_01         0.9908   0.8624   0.1761      keep       149.5    W_XGB 0.10→0.15
exp_02         0.9901   0.8410   0.1924      discard    146.0    xgb_depth 4→5
exp_03         0.9925   0.8589   0.1741      keep       152.0    Exp01 + shallow depth
```

---

## Iteration Strategy by Phase

### Phase 1: Ensemble Weights (Experiments 1-10)

**Goal:** Optimize W_PHYS, W_XGB, W_RF, W_GB for best hybrid score (expect +0.002)

**Data constraint consideration:** With n=69 positives, each model sees ~14 positives per fold. This is TINY. Physics must protect against overfitting.

```python
Starting point (baseline):
  W_PHYS=0.75, W_XGB=0.15, W_RF=0.05, W_GB=0.05

Experiment sequence (each +0.0002-0.0005 if successful):
  1. PHYS=0.70, XGB=0.20, RF=0.05, GB=0.05  (boost XGB)
  2. PHYS=0.70, XGB=0.15, RF=0.10, GB=0.05  (diversify ML)
  3. PHYS=0.70, XGB=0.15, RF=0.05, GB=0.10  (boost calibration)
  4. Best of 1-3 + SQUISH=1.25 (calibration adjustment)
  5. Best of 4 → grid ±0.05 on all weights
  
  Keep best 2, discard rest
  Target end of phase: 0.9915 (+0.0012)
```

---

### Phase 2: Regularization & Hyperparams (Experiments 11-20)

**Goal:** Given best weights, tune capacity for small data (expect +0.0010)

**Strategy:** Assume weights from Phase 1 are optimal. Now prevent overfitting.

```python
Starting point (best from Phase 1):
  Assume: W_PHYS=0.70, W_XGB=0.20, RF=0.05, GB=0.05

11. Reduce depths (safest):
    xgb_max_depth=2, rf_max_depth=3, gb_max_depth=2
    
12. Increase min_samples (second-safest):
    xgb_min_child_weight=10, rf_min_leaf=10, gb_min_leaf=10
    
13. Reduce sampling (avoid overfitting):
    xgb_subsample=0.6, colsample=0.6, gb_subsample=0.7
    
14-15. Combinations of 11-13 (find sweet spot)
    
16-20. Fine-tune learning rates on best config from 14-15
    
Target: 0.9923 (+0.0008 from Phase 1 end)
```

---

### Phase 3: Feature Engineering (Experiments 21-28)

**Goal:** Add domain-specific features without noise (expect +0.0012)

**Strategy:** Be conservative—only add features with clear causal meaning.

```python
Safe candidates (low noise risk):
  - 'fire_darkness' = is_night × alignment_abs  (night fires harder to track)
  - 'acceleration' = dist_slope × dist_slope  (capture changing dynamics)
  - 'rapid_growth' = area_growth_rel × radial_growth_rate (explosive fires)

Risky candidates (noise risk):
  - Polynomial features (squared, cubed)
  - High-order interactions
  - Temporal lags (insufficient history)

Experiment sequence:
  21. Add 'fire_darkness'
  22. Add 'fire_darkness' + 'acceleration'
  23. Best of 21-22 + remove low-importance baseline features
  24-28. Test remove 2-3 features each, keep if Hybrid stable
  
Target: 0.9935 (+0.0012 from Phase 2 end)
```

---

### Phase 4: Advanced Ensembling (Experiments 29-35)

**Goal:** Add survival or meta-learning for final push (expect +0.0007)

**Strategy:** Only if time/compute allows. These are complex but high-value.

```python
Option A: CoxPH survival model (recommended)
  29. Add CoxPH with 0.3 weight (needs lifelines library)
      W = [0.55 Physics, 0.25 CoxPH, 0.10 XGB, 0.05 RF, 0.05 GB]
  
Option B: Stacking (risky with small data)
  30. Meta-learner on CV predictions
      (but: only 13 meta-features per sample, risk of overfitting)
  
Option C: Horizon-specific tuning (safe)
  31-32. Different ensemble weights per horizon
         (12h: physics heavier, 72h: ML heavier)

Target: 0.9942 (FINAL GOAL ✓)
```

---

## Decision Rules Summary

### Keep ✅
- Hybrid improvement ≥ +0.0010
- Fold std decreased AND Hybrid same/improved
- Same Hybrid BUT code simpler (removed features, cleaner logic)
- C-Index AND Brier both improved (rare but excellent)

### Marginal ⚠️
- +0.0005 to +0.0010 improvement
- Code complexity versus gain trade-off
- Decision: Ask yourself "Is this 0.37% improvement worth the extra complexity?"

### Discard ❌
- Hybrid decreased by > 0.0005
- Crash/NaN outputs
- Fold std > 0.012 (overfitting signal)
- Change added 100+ lines with only +0.0005 gain

### Escalate / Backtrack
- 5 consecutive experiments with no improvement → backtrack 2 exps, try orthogonal direction
- Fold variance exploding (+0.003 per experiment) → emergency regularization (reduce W_ML)

---

## Diagnostic Tools

### Check Fold Stability
```bash
tail -20 results.tsv | awk -F'\t' '{print $3}' | sort -n | tail -1
# High value (>0.012) = overfitting risk
```

### See Best Score So Far
```bash
awk -F'\t' '$5=="keep" {print $2}' results.tsv | sort -n | tail -1
```

### Review Git History
```bash
git log --oneline | head -20
```

### Watch Live Progress
```bash
watch -n 5 "tail -5 results.tsv"
```

---

## Tips for Autonomous Agent

1. **Think before running.** Each experiment costs 3-5 min.
2. **Document hypothesis.** Future you (or other agents) needs to understand why.
3. **Track fold varianc.** If std > 0.010, emergency: reduce model capacity.
4. **Commit good experiments.** Keep git clean: one logical change per commit.
5. **Exploit trade-offs.** Brier vs C-Index weighted 70/30. Don't over-optimize C-index.
6. **Simplify when possible.** -5 lines of code + same score = WIN.
7. **Budget time.** 30 experiments × 4 min = 2 hours. Plan accordingly.
8. **Trust physics.** When stuck, increase W_PHYS (0.85+) as safe fallback.

---

## Final Target

| Milestone | Score | Status | Experiments |
|-----------|-------|--------|-------------|
| Baseline | 0.9903 | ✓ | — |
| Phase 1 (Weights) | 0.9915 | In Progress | 1-10 |
| Phase 2 (Hyperparam) | 0.9923 | Pending | 11-20 |
| Phase 3 (Features) | 0.9935 | Pending | 21-28 |
| Phase 4 (Advanced) | 0.9942 | Pending | 29-35 |

**Total improvement:** +0.0039 (0.39%) = Top 20 leaderboard 🎯

---

## Next: Run Enhanced Training

You now have a robust framework. Ready to start?

```bash
cd autoresearch/
python wids_train_enhanced.py  # Test that it runs
# Should output: Feature validation passed, CV scores for 5 folds, Stability metric

python wids_experiment_runner.py "Test enhanced training baseline"
# Should log to results.tsv and report decision
```

Good luck! 🚀
