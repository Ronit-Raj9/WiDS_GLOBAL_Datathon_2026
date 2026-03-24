# WiDS 2026 Autonomous Research: Complete File Guide

## 📚 Documentation Hierarchy

### ⭐ **START HERE**

1. **[README_ENHANCED.md](README_ENHANCED.md)** ← **YOU ARE HERE**
   - System overview (3 min read)
   - Quick start guide (3 steps)
   - What's included and why
   - Expected performance curve
   - Pre-launch checklist

### 👤 **FOR AUTONOMOUS AGENT**

2. **[WIDS_PROGRAM_ENHANCED.md](WIDS_PROGRAM_ENHANCED.md)** ← **AGENT READS THIS DAILY**
   - Phase-by-phase iteration strategy (Phase 1-4)
   - Exactly what to modify in code
   - Experiment workflow (propose → run → decide)
   - Decision rules (KEEP vs DISCARD)
   - Phase 1 details: (10 weight-tuning experiments)
   - Phase 2 details: (10 hyperparameter experiments)
   - Phase 3 details: (8 feature engineering experiments)
   - Phase 4 details: (7 advanced ensemble experiments)

3. **[ADVANCED_STRATEGIES.md](ADVANCED_STRATEGIES.md)** ← **AGENT CONSULTS WHEN STUCK**
   - Why small data (n=221) breaks standard ML
   - 6 anti-overfitting strategies:
     1. Depth control tables
     2. Minimum samples per split/leaf
     3. Ensemble shrinkage & learning rates
     4. Subsample & column sampling
     5. L1/L2 regularization
     6. Feature selection & importance
   - 5 ensemble architectures to try:
     1. Physics-dominant (current baseline)
     2. Physics + Survival models (Cox PH)
     3. Horizon-specific weights
     4. Stacking with meta-learner
     5. Cross-horizon consistency
   - Overfitting diagnostics (Red flag list)
   - Quick reference parameter tuning table

4. **[SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)** ← **AGENT CHECKS FOR CONFUSION**
   - Data flow diagrams
   - Component explanations
   - File organization map
   - Metrics explained (Hybrid vs C-Index vs Brier)
   - Emergency actions reference

### 🔧 **FOR SETUP & INTEGRATION**

5. **[AUTORESEARCH_START_HERE.md](AUTORESEARCH_START_HERE.md)** ← **BEFORE TURNING ON AGENT**
   - How to integrate with Claude Code
   - Example prompt to copy-paste
   - Checklist before running
   - Troubleshooting common issues
   - Success criteria

---

## 🗂️ Code Files (autoresearch/ directory)

### ✅ LOCKED CODE (Do Not Modify)

#### `wids_prepare.py` (440 lines)
**Role:** Immutable evaluation harness  
**Purpose:** Ground-truth metrics and data loading  
**Exported functions agent uses:**
```python
load_data()                    # CSV loading
engineer_base_features()       # 31 base features
get_cv_splits()                # 5-fold stratified
get_physics_probs()            # Fixed physics model
calculate_hybrid_score()       # Metric evaluation
ensure_monotonicity()          # Constraint
```
**Why locked:** Can't be modified without invalidating evaluation

---

#### `wids_experiment_runner.py` (310 lines)
**Role:** Immutable orchestrator  
**Purpose:** Run training, parse metrics, log results, decide KEEP/DISCARD  
**Functions:**
```python
run_experiment()               # subprocess + timeout
extract_metrics_from_output()  # Regex parsing
load_results()                 # Read results.tsv
save_results()                 # Append row
log_experiment()               # Decision logic
```
**Input:** Description string  
**Output:** Exit code (0 = success), results auto-logged  
**Example call:**
```bash
python wids_experiment_runner.py "Exp #1: W_XGB boost to 0.20"
```

---

### 🔧 AGENT-EDITABLE CODE

#### `wids_train_enhanced.py` (360 lines)
**Role:** Modifiable training code  
**Purpose:** Core model training and CV evaluation  

**MODIFIABLE SECTIONS:**

**Section 1: Ensemble Weights (~line 20-26)**
```python
W_PHYS = 0.75      # ← Edit this
W_XGB = 0.15       # ← Edit this
W_RF = 0.05        # ← Edit this
W_GB = 0.05        # ← Edit this
```

**Section 2: Hyperparameters (~line 35-60)**
```python
class ModelParams:
    # XGBoost
    xgb_n_estimators = 100        # ← Edit
    xgb_max_depth = 3             # ← Edit
    xgb_learning_rate = 0.05      # ← Edit
    xgb_subsample = 0.7           # ← Edit
    xgb_colsample_bytree = 0.7    # ← Edit
    xgb_min_child_weight = 5      # ← Edit
    
    # RandomForest
    rf_n_estimators = 150         # ← Edit
    rf_max_depth = 4              # ← Edit
    rf_min_samples_leaf = 5       # ← Edit
    
    # GradientBoosting
    gb_n_estimators = 80          # ← Edit
    gb_max_depth = 2              # ← Edit
    gb_learning_rate = 0.03       # ← Edit
```

**Section 3: Feature Selection (~line 65-100)**
```python
FEATURE_SET = [
    'dist',                # ← Keep/remove
    'v_stable',            # ← Keep/remove
    # ... (18 total)
    'critical_window',     # ← Add new feature
]
```

**Section 4: Post-Processing (~line 220-230 in train_and_evaluate)**
```python
probs_final[df_val['num_perimeters_0_5h'] == 0] *= 0.95  # ← Edit

close_mask = df_val['dist_min_ci_0_5h'] < 5000
probs_final[close_mask] = np.clip(probs_final[close_mask], 0.01, 0.99)  # ← Edit
```

**DO NOT EDIT:**
- `train_and_evaluate()` function logic (can't change core training)
- `get_physics_probs()` import (physics model is fixed)
- `calculate_hybrid_score()` import (metrics are fixed)

---

## 📊 Results & Tracking

### `results.tsv` (Tab-Separated Values)
**Format:** Auto-updated experiment log  
**Columns:**
```
commit          | Git commit hash (7 char) or "baseline"
hybrid_score    | Primary metric (0.9903 baseline)
c_index         | Ranking quality (0.8521 baseline)
brier_score     | Calibration (0.1843 baseline)
status          | keep / discard / marginal / crash
time_sec        | Wall-clock duration for 5-fold CV
description     | What was tried (e.g., "W_XGB 0.15→0.20")
```

**Example:**
```
baseline    0.9903   0.8521   0.1843   keep    147.3   Initial weights
exp_01      0.9912   0.8624   0.1761   keep    149.5   W_XGB boosted
exp_02      0.9901   0.8410   0.1924   discard 146.0   XGB depth increased
exp_03      0.9925   0.8589   0.1741   keep    152.0   Exp_01 + shallow depth
```

**How to read:**
- Highest `hybrid_score` in file = best so far
- `status` = KEEP means it was committed (good change)
- `time_sec` should be 145-155s (stable, 5 folds × ~30s each)
- Watch for `std_hybrid` > 0.010 (overfitting warning)

---

### `logs/` directory
**Auto-generated:** Per-experiment stdout  
**Files:** `experiment_001.log`, `experiment_002.log`, etc.  
**Use:** Debug if experiment crashes

---

## 🎬 Execution Flow

```
Agent reads:
  WIDS_PROGRAM_ENHANCED.md      ← Phase 1 strategy
      ↓
Agent proposes:
  "Hypothesis: C-index low, boost XGBoost
   Change: W_XGB 0.15 → 0.20
   Expected: +0.0015 hybrid"
      ↓
Agent modifies:
  wids_train_enhanced.py
    W_XGB = 0.20   (line 19)
      ↓
Agent runs:
  python wids_experiment_runner.py "Exp #1: W_XGB boost"
      ↓
Runner executes:
  python wids_train_enhanced.py
    → 5-fold CV loop
    → Fold 1-5: hybrid, c_idx, brier printed
    → Final metrics and stability printed
      ↓
Runner parses:
  Extract: hybrid=0.9912, c_idx=0.8624, brier=0.1761
  Compare: +0.0009 vs baseline 0.9903
  Decide: MARGINAL (between 0.0005-0.0010)
      ↓
Runner logs:
  Append row to results.tsv:
  "abc1234  0.9912  0.8624  0.1761  marginal  149.5  Exp #1: W_XGB..."
      ↓
Agent decides:
  "C-index improved (0.8624 > 0.8521), Brier improved
   Net: KEEP (marginal but positive on both)"
      ↓
Agent commits:
  git commit -m "Exp #1: W_XGB 0.15→0.20 (+0.0009)"
      ↓
Agent proposes Exp #2:
  ...similar cycle...
```

---

## 📖 Documentation Reading Order

```
New to system?
  1. README_ENHANCED.md (5 min)
  2. SYSTEM_ARCHITECTURE.md (5 min)
  3. AUTORESEARCH_START_HERE.md (5 min)
  → Ready to start!

Ready to iterate?
  1. WIDS_PROGRAM_ENHANCED.md Phase 1 (15 min)
  2. Run first experiment manually
  3. Check result in results.tsv
  4. Propose Exp #2 using Phase 1 strategy
  5. Continue Phase 1 until all passed

Getting stuck?
  1. Consult ADVANCED_STRATEGIES.md Section matching symptom
  2. Read "Red Flag" diagnostics
  3. Adjust parameters per table
  4. Re-run experiment

Need system diagram?
  → SYSTEM_ARCHITECTURE.md (architecture section)

Need metric explanation?
  → SYSTEM_ARCHITECTURE.md ("Metrics Explained" section)

Need emergency action?
  → SYSTEM_ARCHITECTURE.md ("Emergency Actions" table)
```

---

## 🎯 Experiment Checklist Template

Before each experiment, agent should verify:

```
[ ] Hypothesis is clear and testable
    "I will increase W_XGB because C-index is 30% of metric"
    
[ ] Expected impact is quantified
    "Expected: +0.001 hybrid, +0.005 C-index, -0.002 Brier"
    
[ ] Change is ≤ 3 modifications
    "Modified: W_XGB only" ✓
    
[ ] Code change is visible and minimal
    "Changed 1 line: W_XGB = 0.20  (was 0.15)" ✓
    
[ ] All features in FEATURE_SET exist in data
    Verify: feature in engineer_base_features() or raw CSV
    
[ ] All hyperparameters in valid ranges
    XGB depth: 2-5 ✓, LR: 0.01-0.1 ✓
    
[ ] Decision rule is pre-agreed
    "If hybrid >= 0.9903+0.0010: KEEP" ✓
    
[ ] Description for experiment_runner.py is ready
    "Exp #1: W_XGB 0.15→0.20 (boost ranking)" ✓

Proceed to run!
```

---

## 🚨 Failure Recovery

### If training crashes:
```bash
tail logs/experiment_*.log  # See error
# Usually: feature name typo, invalid parameter value
# Fix in wids_train_enhanced.py and re-run
```

### If results seem wrong:
```bash
# Verify baseline still passes
python wids_train_enhanced.py

# Should output hybrid ≈ 0.9903 baseline
# If very different, data or code corrupted
```

### If git messed up:
```bash
# See recent commits
git log --oneline | head -10

# Undo last commit (keep changes)
git reset --soft HEAD~1

# or Undo last commit AND changes
git reset --hard HEAD~1
```

---

## 📞 Quick Help

| Question | Answer | File |
|----------|--------|------|
| "What's the algorithm?" | Physics (75%) + ML (25%) ensemble | SYSTEM_ARCHITECTURE.md |
| "How to increase C-index?" | Boost W_XGB or improve XGB hyperparams | WIDS_PROGRAM_ENHANCED.md Phase 1 |
| "Why is fold variance high?" | Trees too deep, needs regularization | ADVANCED_STRATEGIES.md Red Flag 1 |
| "Can I change feature set?" | Yes, edit FEATURE_SET list carefully | wids_train_enhanced.py ~line 80 |
| "How to add new feature?" | Must exist in engineer_base_features() or raw data | wids_prepare.py |
| "Why is overfitting happening?" | Small data (n=221) + high model capacity | ADVANCED_STRATEGIES.md Part 1 |
| "What's the target score?" | 0.9942 hybrid (top 20 leaderboard) | README_ENHANCED.md |
| "How many experiments per night?" | 20-30 × 4min = 2-2.5 hours | AUTORESEARCH_START_HERE.md |

---

## ✅ Success Criteria

| Milestone | Target | Check How |
|-----------|--------|-----------|
| Framework setup | All 3 code files working | `python wids_train_enhanced.py` |
| Baseline validation | Hybrid ≈ 0.9903 | `tail results.tsv` |
| Phase 1 complete | Hybrid ≥ 0.9915 | Check results.tsv status=keep |
| Phase 2 complete | Hybrid ≥ 0.9923 | Same |
| Phase 3 complete | Hybrid ≥ 0.9935 | Same |
| Phase 4 complete | Hybrid ≥ 0.9942 | **GOAL ACHIEVED** 🎯 |

---

## 🚀 Next Step

```bash
# Verify everything is set up
cd /home/raj/Documents/CODING/Kaggle/WiDS_GLOBAL_Datathon_2026

# Test training
python autoresearch/wids_train_enhanced.py

# Review output
tail -20  # Should show Hybrid ≈ 0.9903, Stability > 0.98

# If all pass → Ready to start autonomous iteration!
# See: AUTORESEARCH_START_HERE.md for Claude Code integration
```

---

*WiDS 2026 Complete File Guide*  
*Enterprise-grade autonomous research framework*  
*Ready to optimize wildfire evacuation prediction* 🔥→🛡️
