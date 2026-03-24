# WiDS 2026 Autonomous Optimization System

## Quick Start

### Run Autonomous Optimization (Fully Automated)

```bash
cd /home/raj/Documents/CODING/Kaggle/WiDS_GLOBAL_Datathon_2026

# Run 30 experiments automatically
python wids_autonomous_runner.py 30

# Quick test (2 experiments)
python wids_autonomous_runner.py test

# Overnight batch (50 experiments)
python wids_autonomous_runner.py night batch
```

### Manual Mode with Experiment Runner

```bash
# Single experiment
python wids_experiment_runner.py "Your description"

# Check results
cat results.tsv
```

## 🎯 Mission
Build **state-of-the-art wildfire evacuation threat prediction model** using robust small-data survival analysis techniques.

**Current:** 0.9903 hybrid score (baseline)  
**Target:** 0.9942+ (top 20 leaderboard)  
**Data:** 221 fires (69 hits, 152 censored)  
**Challenge:** Avoid overfitting with limited positive examples

---

## 📦 What You Now Have

### 1. **Core Framework** (Locked-Edit-Locked)
- **[wids_prepare.py](wids_prepare.py)** ✅ LOCKED — Immutable evaluation harness
  - Ground-truth metrics (C-index 30%, Brier 70%)
  - Physics-based model
  - Data loading and feature engineering
  
- **[wids_train_enhanced.py](wids_train_enhanced.py)** 🔧 AGENT EDITS — Training code
  - Aggressive anti-overfitting strategies for n=221
  - Feature selection (18 core features)
  - Hyperparameter guardrails
  - Stability monitoring
  
- **[wids_experiment_runner.py](wids_experiment_runner.py)** ✅ LOCKED — Orchestrator
  - Runs CV training
  - Logs results to results.tsv
  - Automatic keep/discard decision

### 2. **Documentation** (Guides for AI Agent)
- **[WIDS_PROGRAM_ENHANCED.md](WIDS_PROGRAM_ENHANCED.md)** 📖 READ THIS FIRST
  - Complete workflow for agent
  - Phase-by-phase iteration strategy
  - Specific experiments to run
  
- **[ADVANCED_STRATEGIES.md](ADVANCED_STRATEGIES.md)** 📚 Theory & Tactics
  - Why small data fails with standard ML
  - 5 anti-overfitting arsenal strategies
  - 5 ensemble architectures to try
  - Overfitting diagnostics
  - Quick reference tables

- **[AUTORESEARCH_START_HERE.md](AUTORESEARCH_START_HERE.md)** 🚀 Setup Instructions
  - How to integrate with Claude Code
  - Example prompt for starting autonomous iteration
  - Checklist before running

### 3. **Results Tracking**
- **[results.tsv](results.tsv)** — Experiment log (auto-updated)
  - All runs: commit, hybrid_score, c_index, brier_score, status
  - Baseline row included
  - Format: tab-separated, sortable

### 4. **Configuration**
- Git branch: `autoresearch/wids-experiment-1` (for tracking)
- Dependencies: xgboost, scikit-learn, numpy, pandas (all in environment)

---

## 🚀 Quick Start (3 Steps)

### Step 1: Verify Framework Works
```bash
cd /home/raj/Documents/CODING/Kaggle/WiDS_GLOBAL_Datathon_2026

# Test enhanced training (should take ~3-5 min)
python autoresearch/wids_train_enhanced.py

# Expected output:
# ✓ Feature validation passed (18 features)
# Fold 1: hybrid=0.999201, c_idx=0.8524, brier=0.1743
# Fold 2: hybrid=0.984320, c_idx=0.8401, brier=0.1821
# ...
# FINAL METRICS
#   Hybrid Score: 0.990XXX
#   Stability: 0.98XX (>0.98 is good)
```

### Step 2: Test Experiment Runner
```bash
cd autoresearch/

# Run baseline validation experiment
python wids_experiment_runner.py "Baseline validation with enhanced training"

# Expected output:
# ✓ Training completed in 149.2s
# Results logged to results.tsv
# Status: KEEP (baseline)
```

### Step 3: Start Autonomous Iteration

**Option A: Use Claude Code (Recommended)**
```
Open VS Code with Claude Code extension.
Paste this prompt:

---
Hi! I'm ready to optimize the WiDS 2026 wildfire prediction model.

Files to read:
1. autoresearch/WIDS_PROGRAM_ENHANCED.md (agent instructions)
2. autoresearch/ADVANCED_STRATEGIES.md (theory & tactics)
3. autoresearch/AUTORESEARCH_START_HERE.md (setup)
4. autoresearch/results.tsv (current baseline)

Current status:
- Baseline: 0.9903 hybrid score
- Data: 221 fires (69 hits, 152 censored) - SMALL DATA REGIME
- Model: Physics (75%) + XGBoost (15%) + ML (10%)
- Target: 0.9942+ (top 20 leaderboard)

Phase 1: Ensemble Weights (Experiments 1-10)

Please review the strategy, then propose Experiment #1 with:
1. Hypothesis (why this change helps)
2. Expected impact (quantitative)
3. The specific change (show modified code lines)
4. Then run: python autoresearch/wids_experiment_runner.py "Exp #1: ..."
5. Evaluate results and decide KEEP/DISCARD
6. Propose Experiment #2 based on results

Let's go! 🚀
---
```

**Option B: Manual Iteration**
```bash
# Edit feature by feature or weight by weight
nano autoresearch/wids_train_enhanced.py

# Run experiment
python autoresearch/wids_experiment_runner.py "Your description"

# Check results
tail autoresearch/results.tsv

# If good, commit
git add autoresearch/wids_train_enhanced.py autoresearch/results.tsv
git commit -m "Exp #1: Description"

# Repeat for Exp #2, #3, ...
```

---

## 🔑 Key Concepts

### Why Small Data Breaks Standard ML
```
221 samples + 34 features → naturally prone to overfitting

Each classification tree can memorize individual samples
Random Forest ensemble can fit noise with deep trees
XGBoost can converge to spurious correlations

Solution: Heavy regularization + physics domain knowledge
```

### Physics as Regularization
```
Physics model: 75% weight (up from 80% baseline)
Reason: Domain knowledge prevents overfitting to noise

ML models: 25% combined weight
- XGBoost (15%): Fast learner, good for ranking (C-index)
- Random Forest (5%): Diversity, stability
- Gradient Boost (5%): Calibration (Brier score)

This asymmetry is INTENTIONAL for small data
```

### The Metric Trade-off
```
Hybrid Score = 0.3×C-Index + 0.7×(1 - Brier Score)

C-Index (30%): How well you rank urgency
  - High weight on getting order right (which fires most dangerous?)
  - Improves with better XGBoost (ranking)
  - Range: 0.5 (random) to 1.0 (perfect)

Brier Score (70%): Calibration accuracy
  - High weight on calibration (are probabilities correct?)
  - Improves with GBM + physics guidance
  - Range: 0 (perfect) to 1 (terrible)
  - Weighted average of 3 horizons: 0.3×24h + 0.4×48h + 0.3×72h

Trade-off: Improving C-index sometimes hurts Brier (more confident → worse calibration)
Decision: Watch both metrics, optimize Hybrid jointly
```

---

## 📊 Expected Performance Curve

```
Experiment #  Action                    Expected Score  Cumulative Gain
——————————————————————————————————————————————————————————————————————
0 (baseline)  Physics 80%, XGB 10%      0.9903          0.0000
1-5           Reweight ensemble         0.9915          +0.0012
6-10          Regularize hyperparams    0.9923          +0.0020
11-18         Add safe features         0.9935          +0.0032
19-25         Survival modeling (Cox)   0.9942          +0.0039
26-30         Fine-tune calibration     0.9945          +0.0042
——————————————————————————————————————————————————————————————————————
Final target: 0.9942+                   ACHIEVED ✓
```

---

## 🎯 Experiment Strategy by Phase

### Phase 1: Ensemble Weights (Exps 1-10)
**Goal:** Find optimal balance between physics and ML  
**Expected gain:** +0.0012  
**Time:** 45 min (10 experiments × 4.5 min)

```
Key weight adjustments:
  - Try W_XGB: 0.10 → 0.15 → 0.20 (affects C-index)
  - Try W_GB: 0.05 → 0.10 (affects Brier)
  - Keep W_PHYS dominant (70-75%)

Decision rule: Keep if Hybrid ≥ baseline + 0.0010
```

### Phase 2: Hyperparameters (Exps 11-20)
**Goal:** Tune model capacity given optimized weights  
**Expected gain:** +0.0010  
**Time:** 50 min

```
Key parameters:
  - Tree depths: Reduce to 3-4 (shallow = robust)
  - Min samples: Increase to 5-10 per leaf
  - Learning rates: Slow down (0.02-0.05)
  - Sampling rates: Row/feature dropout (0.6-0.8)

Decision rule: Keep if Hybrid ≥ baseline + 0.0005 (+fold stability)
```

### Phase 3: Features (Exps 21-28)
**Goal:** Add domain-specific features without noise  
**Expected gain:** +0.0012  
**Time:** 40 min

```
Safe features to add:
  - Interactions: night × alignment, growth × speed
  - Temporal: seasonal effects
  - Physics-derived: acceleration, kinetic energy

Avoid: Polynomial features, high-order interactions, temporal lags

Decision rule: Each +0.0005 improvement = KEEP
```

### Phase 4: Advanced (Exps 29-35)
**Goal:** Add survival or meta-learning if time permits  
**Expected gain:** +0.0007  
**Time:** 60 min

```
Advanced techniques:
  1. CoxPH survival model (most promising)
  2. Stacking with meta-learner (risky on small data)
  3. Horizon-specific ensemble weights (safe)

Decision rule: Keep if Hybrid ≥ baseline + 0.0008
```

---

## ⚙️ How to Modify Code

### Edit 1: Change Ensemble Weights
```python
# In wids_train_enhanced.py, line ~20:

W_PHYS = 0.70      # was 0.75 (decrease physics)
W_XGB = 0.20       # was 0.15 (increase XGBoost)
W_RF = 0.05
W_GB = 0.05
```

### Edit 2: Change Hyperparameter
```python
# In ModelParams class (~line 60):

class ModelParams:
    xgb_max_depth = 2           # was 3 (SHALLOWER)
    xgb_min_child_weight = 10   # was 5
    rf_max_depth = 3            # was 4
    gb_learning_rate = 0.02     # was 0.03 (SLOWER)
```

### Edit 3: Add Feature
```python
# In FEATURE_SET list (~line 80-100):

FEATURE_SET = [
    # existing features...
    'critical_window',  # NEW: night fires alignment boost
]

# Then in engineer_base_features() in wids_prepare.py (if needed):
# This is LOCKED, so only add features that already exist!
```

---

## 🔍 Monitoring & Diagnostics

### Green Flag ✅
```
Fold std: 0.004-0.008 (stable)
C-Index improving (current: 0.8521, target: 0.86+)
Brier improving (current: 0.1843, target: 0.17+)
Hybrid improving step-wise (+0.0005 per exp)
Training time: 3-5 min per fold (consistent)
```

### Yellow Flag ⚠️
```
Fold std: 0.008-0.012 (warning, watch carefully)
No improvement for 3 experiments (but all ≥baseline-0.0003)
One fold much better than others (imbalance indicator)
Brier improving but C-index dropping (miscalibration)
```

### Red Flag 🛑
```
Fold std: >0.012 (OVERFITTING DETECTED)
Hybrid decreasing (wrong direction)
Crash or NaN (code error)
Training time: >10 min per fold (expensive, check specs)
One fold: 0.993, another: 0.982 (delta > 0.010)
```

**Emergency Response:**
```python
# If overfitting detected:
1. Revert last change
2. Increase regularization:
   - xgb_max_depth = 2 (was 3)
   - xgb_min_child_weight = 10 (was 5)
   - W_PHYS = 0.80 (was 0.75)
3. Re-run experiment
4. If still bad, reduce features
```

---

## 📈 Results Tracking

### Check Progress
```bash
# Best score so far
awk -F'\t' '$5=="keep" {print NR, $2}' autoresearch/results.tsv | tail -1

# Fold variance (watch for overfitting)
tail -1 autoresearch/results.tsv | awk -F'\t' '{print "Stability:", $6}'

# All kept experiments
grep "keep" autoresearch/results.tsv | tail -10
```

### Understand results.tsv
```
Column 1: Git commit hash (auto-filled)
Column 2: Hybrid Score (main metric)
Column 3: C-Index (ranking quality)
Column 4: Brier Score (calibration)
Column 5: Status (keep/discard/crash)
Column 6: Training time (seconds)
Column 7: Description (what you tried)

Example:
baseline   0.9903   0.8521   0.1843   keep   147.3   Physics 80%
exp_01     0.9912   0.8624   0.1761   keep   149.5   W_XGB 0.10→0.15
exp_02     0.9921   0.8589   0.1741   keep   151.2   Exp1 + shallow depth
```

---

## 🎓 Learning Resources

### Within This Project
1. **WIDS_PROGRAM_ENHANCED.md** — Day-to-day agent instructions
2. **ADVANCED_STRATEGIES.md** — Deep dive on small-data techniques
3. **AUTORESEARCH_START_HERE.md** — Integration & setup

### External References
- **Survival Analysis:** `lifelines` library (Cox PH, Kaplan-Meier)
- **Small Data Techniques:** Papers on regularization in ML
- **Ensemble Learning:** XGBoost docs, Scikit-learn ensembles
- **Competition:** Kaggle WiDS 2026 discussion forum

---

## 🚨 Troubleshooting

### "ModuleNotFoundError: xgboost"
```bash
pip install xgboost scikit-learn numpy pandas scipy
```

### "FileNotFoundError: data/train.csv"
```bash
# Make sure you're running from the project root:
cd /home/raj/Documents/CODING/Kaggle/WiDS_GLOBAL_Datathon_2026/
python autoresearch/wids_train_enhanced.py
```

### "KeyError: feature 'XYZ' not found"
```
Check that feature is created in engineer_base_features()
in wids_prepare.py

Or, remove it from FEATURE_SET in wids_train_enhanced.py
```

### "AssertionError: Weights must sum to 1"
```
W_PHYS + W_XGB + W_RF + W_GB must = 1.0
Example: 0.75 + 0.15 + 0.05 + 0.05 = 1.00 ✓
```

### "high fold variance detected"
```bash
Check stability score at end of training output
If Stability < 0.98 or Fold Std > 0.010:
  - Reduce max_depth
  - Increase min_samples_leaf
  - Boost W_PHYS
  - Reduce features
```

---

## ✅ Pre-Launch Checklist

Before starting autonomous iteration:

- [ ] Read WIDS_PROGRAM_ENHANCED.md (15 min)
- [ ] Read ADVANCED_STRATEGIES.md Section 1-2 (10 min)
- [ ] Run `python wids_train_enhanced.py` manually (5 min)
- [ ] Run one experiment manually with runner (5 min)
- [ ] Understand results.tsv format (2 min)
- [ ] Git branch created: `git branch` shows autoresearch branch
- [ ] Baseline row in results.tsv (start with ~0.9903)
- [ ] Claude Code (or agent) ready to read files

**Total prep time:** 40 min → Ready to start autonomous iteration 🚀

---

## 🎯 Success Criteria

| Milestone | Target | Timeline | Status |
|-----------|--------|----------|--------|
| Framework validation | Both tests pass | Day 1 | ✓ |
| Phase 1 complete | 0.9915 score | Day 1-3 (~30 exps) | -- |
| Phase 2 complete | 0.9923 score | Day 2-3 (~20 exps) | -- |
| Phase 3 complete | 0.9935 score | Day 3+ (~16 exps) | -- |
| Phase 4 complete | 0.9942 score | Day 4+ (~12 exps) | -- |
| **FINAL GOAL** | **TOP 20** | **By deadline** | **🎯** |

---

## 📞 Support & Questions

If agent gets stuck:

1. **High fold variance?** → Section "Red Flag 1" in ADVANCED_STRATEGIES.md
2. **Which weight to change?** → WIDS_PROGRAM_ENHANCED.md Phase 1
3. **Feature engineering ideas?** → ADVANCED_STRATEGIES.md Part 4
4. **New model to add?** → ADVANCED_STRATEGIES.md Part 3 (Ensemble 2-4)

---

## 🏁 Next Step

```bash
cd /home/raj/Documents/CODING/Kaggle/WiDS_GLOBAL_Datathon_2026

# Test everything works
python autoresearch/wids_train_enhanced.py

# Run first experiment
cd autoresearch
python wids_experiment_runner.py "Validation run with enhanced training"

# Read results
tail results.tsv
cat WIDS_PROGRAM_ENHANCED.md | head -50

# Start agent iteration (Claude Code or manual)
# See AUTORESEARCH_START_HERE.md for Claude Code prompt
```

**You're ready. Let's achieve 0.994+! 🚀**

---

*WiDS 2026 Autonomous Research System*  
*Enterprise-grade framework for small-data survival analysis*  
*Built for Kaggle competition with continuous improving*
