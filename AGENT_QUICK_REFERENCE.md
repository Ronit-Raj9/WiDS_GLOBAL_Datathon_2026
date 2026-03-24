# WiDS 2026: Agent Quick Reference Card

A **1-page** guide for AI agents to keep open during iteration.

---

## 🎯 THE GOAL
```
Current: 0.9903 hybrid score (5-fold CV)
Target:  0.9942 hybrid score (top 20)
Need:    +0.0039 improvement (35 experiments, ~2.5 hours)
```

## 📊 DECISION RULES (Memorize These)

```
Hybrid improvement ≥ +0.0010  → ✅ KEEP  (git commit)
Hybrid improvement 0.0005-0.0010 → ⚠️ MARGINAL  (check complexity)
Hybrid improvement < +0.0005  → ❌ DISCARD  (git reset --hard)
Fold std > 0.012              → 🛑 OVERFITTING  (add regularization)
```

---

## 🔧 WHAT TO MODIFY (in wids_train_enhanced.py)

### 1️⃣ Ensemble Weights
```python
W_PHYS = 0.75      # Try: 0.70-0.80  (physics)
W_XGB = 0.15       # Try: 0.10-0.25  (ranking)
W_RF = 0.05        # Try: 0.05-0.10  (diversity)
W_GB = 0.05        # Try: 0.05-0.10  (calibration)
```

**When to adjust:**
- C-Index low → increase W_XGB
- Brier high → increase W_GB
- Both bad → keep W_PHYS high (0.75+)

### 2️⃣ Hyperparameters (ModelParams class)
```python
# SAFE: Make shallower / more regularized
xgb_max_depth = 3          # Try: 2-4
xgb_min_child_weight = 5   # Try: 5-10
rf_max_depth = 4           # Try: 3-5
gb_max_depth = 2           # Try: 1-3

# MODERATE: Adjust learning
xgb_learning_rate = 0.05   # Try: 0.02-0.08
gb_learning_rate = 0.03    # Try: 0.01-0.05

# RISKY: Larger models
xgb_n_estimators = 100     # Try: 80-150
rf_n_estimators = 150      # Try: 100-250
```

### 3️⃣ Features (FEATURE_SET list)
```python
FEATURE_SET = [
    'dist', 'v_stable', 'alignment_abs',  # Must keep
    'eta_kinetic', 'speed_alignment',     # High value
    # ... 18 total
    'new_feature_here',  # Add new
]
```

### 4️⃣ Calibration
```python
# Adjust confidence for uncertain fires
probs_final[no_perimeter] *= 0.95  # Try: 0.90-0.98

# Clip very close/far fires
probs_final[close] = np.clip(probs_final[close], 0.01, 0.99)
```

---

## 🔄 THE 5-STEP ITERATION LOOP

```
STEP 1: PROPOSE
  What + Why + Expected?
  "W_XGB 0.15→0.20 (boost C-index from 0.8521) → +0.0015"

STEP 2: EDIT
  Change max 3 things in wids_train_enhanced.py
  Example: W_XGB = 0.20  # line 19

STEP 3: RUN
  cd autoresearch/
  python wids_experiment_runner.py "Exp #N: <description>"
  Wait 3-5 min

STEP 4: ANALYZE
  Hybrid score improved? (main metric)
  C-Index improved? (30% weight, ranking)
  Brier improved? (70% weight, calibration)
  Fold std < 0.012? (no overfitting)

STEP 5: DECIDE & COMMIT
  IF hybrid ≥ baseline + 0.0010:
    git commit -m "Exp #N: Description"
  ELSE:
    git reset --hard HEAD~1
  
  Proceed to STEP 1 for next experiment
```

---

## ⚡ QUICK REFERENCE: Common Changes

### Problem: C-Index Low (0.85 target: 0.86+)
```python
Solution 1 - Boost XGBoost (fast learner):
  W_PHYS = 0.70          (was 0.75)
  W_XGB = 0.20           (was 0.15)
  Expected: +0.005 C-Index, +0.0015 Hybrid

Solution 2 - Sharper XGBoost:
  xgb_max_depth = 4      (was 3, more flexible)
  xgb_learning_rate = 0.08  (was 0.05, faster)
  Expected: +0.003 C-Index, +0.001 Hybrid
  Risk: Overfitting if fold_std increases
```

### Problem: Brier High (0.184 target: 0.17+)
```python
Solution 1 - Boost Calibration:
  W_GB = 0.10            (was 0.05)
  gb_learning_rate = 0.05  (was 0.03)
  Expected: -0.005 Brier, +0.0015 Hybrid

Solution 2 - Ensemble with Physics:
  W_PHYS = 0.80          (was 0.75, more stable)
  W_GB = 0.05            (revert)
  Expected: More stable, -0.001 Brier
  
Solution 3 - Adjust SQUISH calibration:
  SQUISH = 1.1           (was 1.2, less extreme)
  Expected: -0.003 Brier
```

### Problem: Fold Variance High (std > 0.010)
```
RED FLAG - OVERFITTING DETECTED!

Emergency Actions:
1. Reduce depths:
   xgb_max_depth = 2
   rf_max_depth = 3
   gb_max_depth = 1
   Expected: std -0.005, Hybrid -0.002

2. Increase min_samples:
   xgb_min_child_weight = 10
   rf_min_samples_leaf = 10
   Expected: std -0.003, Hybrid -0.001

3. Boost physics (safe):
   W_PHYS = 0.85         (was 0.75)
   W_XGB = 0.10          (reduced)
   Expected: std -0.008, Hybrid -0.0005

Choose ONE, re-run, check if std ↓
```

### Problem: No Improvement (5+ consecutive experiments)
```
Strategy: Backtrack + Change Direction

1. Revert to last successful commit:
   git log --oneline | head
   git checkout <successful_commit>

2. Try orthogonal direction:
   If Phase 1 (weights) → skip to Phase 3 (features)
   If tuning depth → try learning_rate instead
   If boosting XGB → try diversifying ML (boost RF)

3. Add regularization while changing direction:
   Reduce more aggressively
   Example: xgb_max_depth=2, min_child_weight=10, subsample=0.6

4. Add domain feature:
   'fire_darkness' = is_night × alignment_abs
   Expected: +0.0005 if good signal
```

---

## 📈 EXPERIMENT STRATEGY BY PHASE

### Phase 1: Weights (Exp 1-10)
```
Sequence:
  1. PHYS=0.70, XGB=0.20, RF=0.05, GB=0.05    (boost XGB)
  2. PHYS=0.70, XGB=0.15, RF=0.10, GB=0.05    (diversify)
  3. PHYS=0.70, XGB=0.15, RF=0.05, GB=0.10    (boost GB)
  4-10. Grid on best, fine-tune

Expected: 0.9903 → 0.9915 (+0.0012)
```

### Phase 2: Hyperparams (Exp 11-20)
```
Given best weights from Phase 1:
  1. Reduce depths (safety): xgb=2, rf=3, gb=1
  2. Increase min_samples: min_leaf=10, min_split=20
  3. Adjust learning rates: xgb=0.03, gb=0.02
  4-10. Combinations of 1-3

Expected: 0.9915 → 0.9923 (+0.0008)
```

### Phase 3: Features (Exp 21-28)
```
Add ONE safe feature per exp:
  1. 'fire_darkness' = is_night × alignment_abs
  2. 'growth_momentum' = area_growth_rel × radial_growth
  3-8. Remove low-importance features or add new

Expected: 0.9923 → 0.9935 (+0.0012)
```

### Phase 4: Advanced (Exp 29-35)
```
If time permits:
  1. Add CoxPH survival model (need lifelines)
     W = [0.50 Phys, 0.25 CoxPH, 0.10 XGB, 0.10 RF, 0.05 GB]
  2-7. Fine-tune weights, stacking, or horizon-specific

Expected: 0.9935 → 0.9942 (+0.0007)
GOAL ACHIEVED ✓
```

---

## 🔴 RED FLAGS: WHAT TO WATCH

| Symptom | Severity | Action |
|---------|----------|--------|
| Fold std > 0.010 | 🔴 High | Add regularization NOW |
| One fold >> others | 🟡 Med | Verify stratification |
| C-idx down, Brier up | 🟡 Med | Check trade-off is net positive |
| Training time > 10min | 🟡 Med | Reduce estimators or depth |
| Hybrid decreasing 3 exp | 🔴 High | Backtrack, try new direction |
| Crash/NaN | 🔴 High | Check feature names, params |

---

## 📋 EXPERIMENT TEMPLATE

Copy-paste and fill in:

```
EXPERIMENT #N
──────────────
Title: 
  [Concise name, e.g., "W_XGB Boost Test"]

Hypothesis:
  [Why this change: "C-index 30% weight but low ranking"]

Expected Impact:
  Hybrid: [+0.001?], C-Index: [+0.005?], Brier: [no change?]

Change:
  [Line 19: W_XGB = 0.20  (was 0.15)]
  [Max 3 changes]

Command:
  python wids_experiment_runner.py "Exp #N: W_XGB Boost Test"

Result:
  Hybrid: [actual from results.tsv]
  C-Index: [actual]
  Brier: [actual]
  Fold std: [actual]

Decision:
  ✅ KEEP / ⚠️ MARGINAL / ❌ DISCARD

Commit:
  git commit -m "Exp #N: W_XGB Boost Test (+0.001X)"
```

---

## 🔑 KEY METRICS TO TRACK

```
Metric              Baseline   Target    What It Means
───────────────────────────────────────────────────────
Hybrid Score        0.9903     0.9942    Overall performance
C-Index             0.8521     0.86+     Ranking quality
Brier Score         0.1843     0.17+     Calibration
Fold Std            0.008      <0.008    Generalization
Stability           0.98       >0.98     Robustness
Time/fold           30s        30s       Computational budget
```

---

## 🛑 EMERGENCY STOP

If any of these happen, **STOP and REVERT**:

```
✋ Fold std > 0.012  (overfitting crisis)
✋ Hybrid < previous - 0.002  (major regression)
✋ Code crashes repeatedly  (bug in logic)
✋ Training time > 15min  (resource issue)

Action: git reset --hard HEAD~1
Then:   Add heavy regularization
```

---

## 📊 TRACKING TEMPLATE

After each experiment, update this:

```
Session Progress (Current Time: HH:MM)
──────────────────────────────────────
Baseline:      0.9903
Best so far:   0.9915  (Exp #4)
Latest:        0.9912  (Exp #N)
Improvement:   +0.0012 (0.12%)

Experiments completed: N/35
Experiments KEPT: X
Experiments DISCARDED: Y
Experiments MARGINAL: Z

Next phase: Phase [1/2/3/4], Exp #[N+1]

Notes:
- [Observation: C-index boosting consistently helps]
- [Observation: GBM undershoots, try less weight]
- [Next try: horizon-specific weights]
```

---

## 🚀 QUICK START (First-Time Agent)

```
1. Read this card (2 min) ✓ DONE
2. Read WIDS_PROGRAM_ENHANCED.md Phase 1 (5 min)
3. Propose Experiment #1 using template above
4. Edit wids_train_enhanced.py (1 min)
5. Run: python wids_experiment_runner.py "Exp #1: ..."
6. Check results.tsv (1 min)
7. Decide KEEP/DISCARD (1 min)
8. Git commit or reset (1 min)
9. Loop → Step 3 for Exp #2

Total per experiment: ~5-7 min including wait
Total for 30 experiments: ~2-2.5 hours

READY? Start with Exp #1! 🚀
```

---

## 📞 STUCK? CONSULT THIS

```
Question                              See This
─────────────────────────────────────────────────────────
"What do I modify?"                   Section "WHAT TO MODIFY"
"How do I increase C-index?"          Section "Common Changes"
"What if fold variance is high?"      Section "RED FLAGS"
"What's Phase 1?"                     Section "STRATEGY BY PHASE"
"Do I pass this experiment?"          Section "DECISION RULES"
"How to add a feature?"               FILE_GUIDE.md Section "wids_train_enhanced.py"
"Metric explanation?"                 SYSTEM_ARCHITECTURE.md "Metrics Explained"
"System overview?"                    README_ENHANCED.md
```

---

## ✅ SUCCESS CHECKLIST

Before EACH experiment:

- [ ] Hypothesis is clear (Why this change?)
- [ ] Expected impact is quantified (How much improvement?)
- [ ] Code change is ≤ 3 modifications
- [ ] All features exist in data or engineer_base_features()
- [ ] Hyperparameters in valid ranges (depth=2-5, LR=0.01-0.1)
- [ ] Experiment description ready for command

Before COMMITTING:

- [ ] Hybrid improved by ≥ 0.0005 (minimum acceptable)
- [ ] Fold std hasn't exploded (< 0.015)
- [ ] Can explain why this change helped
- [ ] Git message clear and concise

---

*WiDS 2026 Quick Reference*  
*Keep this open during iteration*  
*Optimize wildfire evacuation prediction*  
*Target: 0.9942+ hybrid score* 🎯
