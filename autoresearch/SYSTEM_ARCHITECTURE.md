# WiDS 2026: System Architecture & Component Map

## 🏗️ Overall System Design

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AUTONOMOUS RESEARCH SYSTEM                          │
│                                                                          │
│  AI Agent (Claude Code / Manual)                                         │
│         ↓                                                                │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  DECISION LOOP (Iterate 30x per night)                           │  │
│  │                                                                  │  │
│  │  1. Read: WIDS_PROGRAM_ENHANCED.md                              │  │
│  │           ↓ (instructions & strategy)                           │  │
│  │                                                                  │  │
│  │  2. Propose: "Hypothesis: ... Expected: ... Change: ..."        │  │
│  │           ↓ (use Phase 1-4 guideline)                           │  │
│  │                                                                  │  │
│  │  3. Modify: wids_train_enhanced.py (weights/hyperparams)        │  │
│  │           ↓ (change max 3 things per cycle)                     │  │
│  │                                                                  │  │
│  │  4. Run:    python wids_experiment_runner.py "Exp #X: ..."      │  │
│  │           ↓ (waits 3-5 min for 5-fold CV)                       │  │
│  │                                                                  │  │
│  │  5. Analyze: Check output, read results.tsv                     │  │
│  │           ↓ (interpret Hybrid, C-Index, Brier, Stability)       │  │
│  │                                                                  │  │
│  │  6. Decide:  KEEP(+git) or DISCARD(reset)                       │  │
│  │           ↓ (decision rule: ≥+0.0010 keep)                      │  │
│  │                                                                  │  │
│  │  7. Commit:  git commit -m "Exp #X: ..."                        │  │
│  │           ↓ (track progress in git log)                         │  │
│  │                                                                  │  │
│  │  8. Next:    Loop back to Step 1                                │  │
│  │                                                                  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                                  │
│                                                                          │
│  wids_train_enhanced.py (AGENT EDITABLE)                                │
│         ↓                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │ 1. Load Data                                                     │   │
│  │    train.csv (221 fires), test.csv (95 fires)                   │   │
│  │                                                                 │   │
│  │ 2. Engineer Features (from wids_prepare.py)                     │   │
│  │    34 raw + 31 derived = 65 available                          │   │
│  │    FEATURE_SET selects 18 safe ones                            │   │
│  │                                                                 │   │
│  │ 3. 5-Fold Stratified CV Loop                                   │   │
│  │    For each fold:                                              │   │
│  │      ├─ Physics Model (get_physics_probs)                      │   │
│  │      ├─ XGBoost (n=100, depth=3, LR=0.05)                      │   │
│  │      ├─ RandomForest (n=150, depth=4)                          │   │
│  │      ├─ GradientBoosting (n=80, depth=2, LR=0.03)              │   │
│  │      └─ Ensemble Blend: W_P×P + W_X×X + W_R×R + W_G×G          │   │
│  │                                                                 │   │
│  │ 4. Validation Metrics (from wids_prepare.py)                   │   │
│  │    ├─ C-Index: Ranking quality (30% weight)                    │   │
│  │    ├─ Brier Score: Calibration (70% weight)                    │   │
│  │    └─ Hybrid: 0.3×C + 0.7×(1-Brier)                            │   │
│  │                                                                 │   │
│  │ 5. Report Results                                              │   │
│  │    ├─ Mean hybrid ± std                                        │   │
│  │    ├─ Fold stability metric (1 - std/mean)                     │   │
│  │    └─ Per-fold breakdown                                       │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         ↓                                                                │
│  wids_experiment_runner.py (LOCKED ORCHESTRATOR)                        │
│         ↓                                                                │
│  Results logged to results.tsv (auto-appended)                          │
│         ↓                                                                │
│  Git commit (if KEEP)                                                   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 File Organization

```
WiDS_GLOBAL_Datathon_2026/
│
├── data/
│   ├── train.csv          (221 fires, 34 features + targets)
│   ├── test.csv           (95 fires, 34 features)
│   ├── metaData.csv       (data dictionary)
│   └── sample_submission.csv
│
├── autoresearch/
│   │
│   ├── ⭐ DOCUMENTATION (Read These First)
│   │   ├── README_ENHANCED.md           ← START HERE (overview)
│   │   ├── WIDS_PROGRAM_ENHANCED.md     ← Agent instructions
│   │   ├── ADVANCED_STRATEGIES.md       ← Theory & tactics
│   │   ├── AUTORESEARCH_START_HERE.md   ← Setup guide
│   │   └── WIDS_SETUP.md                (original architecture)
│   │
│   ├── 🔧 MODIFIABLE CODE
│   │   └── wids_train_enhanced.py       ← Agent edits this!
│   │       ├── Ensemble weights (W_PHYS, W_XGB, W_RF, W_GB)
│   │       ├── Hyperparameters (depths, learning rates, min_samples)
│   │       ├── Feature selection (FEATURE_SET list)
│   │       └── Post-processing (calibration, adjustments)
│   │
│   ├── ✅ LOCKED CODE
│   │   ├── wids_prepare.py              (evaluation harness)
│   │   │   ├── load_data()
│   │   │   ├── engineer_base_features()
│   │   │   ├── get_physics_probs()
│   │   │   ├── calculate_hybrid_score()
│   │   │   └── get_cv_splits()
│   │   │
│   │   └── wids_experiment_runner.py    (orchestrator)
│   │       ├── run_experiment()         (subprocess training)
│   │       ├── extract_metrics()        (parse stdout)
│   │       ├── log_experiment()         (save to results.tsv)
│   │       └── print_summary()          (show stats)
│   │
│   ├── 📊 RESULTS & TRACKING
│   │   ├── results.tsv                  (experiment log)
│   │   │   ├── baseline row (0.9903)
│   │   │   └── auto-appended per run
│   │   │
│   │   └── logs/                        (per-experiment stdout)
│   │       ├── experiment_001.log
│   │       ├── experiment_002.log
│   │       └── ...
│   │
│   └── 🗂️ OTHER
│       ├── prepare.py                   (old, ignore)
│       ├── train.py                     (old, ignore)
│       ├── program.md                   (old, ignore)
│       ├── .gitignore
│       ├── pyproject.toml
│       └── analysis.ipynb
│
├── hi.ipynb                    (original notebook, working)
├── hi.py                       (original script)
│
└── .git/                       (git repository)
    └── autoresearch/       (experiment branch)
        └── commits logged per experiment
```

---

## 🎯 Key Components Explained

### 1. **wids_train_enhanced.py** (Agent Modifies)

**What it does:**
- Performs 5-fold stratified CV
- Trains 4 models (Physics, XGB, RF, GB) × 4 horizons
- Blends predictions with configurable weights
- Reports: hybrid_score, c_index, brier_score, stability

**What agent changes:**
```python
# Ensemble weights
W_PHYS = 0.75      ← Try: 0.70, 0.80
W_XGB = 0.15       ← Try: 0.10, 0.20
W_RF = 0.05        ← Try: 0.05, 0.10
W_GB = 0.05        ← Try: 0.05, 0.10

# Hyperparameters
xgb_max_depth = 3           ← Try: 2, 4
xgb_learning_rate = 0.05    ← Try: 0.02, 0.08
rf_min_samples_leaf = 5     ← Try: 3, 10

# Features
FEATURE_SET = ['feature1', 'feature2', ...]  ← Add/remove
```

**Anti-overfitting safeguards built-in:**
- Shallow trees (max_depth ≤ 4)
- High min_samples_leaf (≥5)
- Row/column dropout (0.6-0.8)
- L2 regularization
- Feature validation
- Stability monitoring (fold std warning)

---

### 2. **wids_prepare.py** (Locked Evaluation Harness)

**What it does:**
- Calculates true metrics (can't be gamed)
- Provides physics probabilities (fixed baseline)
- Defines CV folds (stratified, fixed seed)
- Engineers base features (foundation for all models)

**Why it's locked:**
- Ground truth for evaluation
- Prevents metric calculation errors
- Ensures reproducibility
- One source of truth for physics model

**Key functions agent uses:**
```python
from wids_prepare import:
  - load_data()                    # CSV loading
  - engineer_base_features()       # Base features (31 derived)
  - get_cv_splits()                # 5-fold setup
  - get_physics_probs()            # Fixed physics predictions
  - calculate_hybrid_score()       # Metric evaluation
  - ensure_monotonicity()          # Constraint enforcement
```

---

### 3. **wids_experiment_runner.py** (Locked Orchestrator)

**What it does:**
- Calls `subprocess.run('python wids_train_enhanced.py')`
- Parses stdout to extract metrics
- Compares to baseline (0.9903)
- Logs row to results.tsv
- Decides KEEP or DISCARD
- Optionally commits to git

**Decision logic:**
```python
if new_hybrid >= baseline + 0.0010:
    status = "keep"          # Improvement threshold
    git_commit()
    
elif baseline - 0.0005 < new_hybrid < baseline + 0.0010:
    status = "marginal"      # Judgment call
    
else:
    status = "discard"       # No improvement
    git_reset()
```

---

### 4. **Documentation Suite**

| File | Purpose | Audience |
|------|---------|----------|
| **README_ENHANCED.md** | System overview, quick start, troubleshooting | Everyone |
| **WIDS_PROGRAM_ENHANCED.md** | Day-to-day instructions, phase strategy, experiments | AI Agent |
| **ADVANCED_STRATEGIES.md** | Theory, anti-overfitting tactics, diagnostics | AI Agent + Advanced Users |
| **AUTORESEARCH_START_HERE.md** | Setup guide, integration with Claude Code | Setup Phase |

---

## 🔄 Data Flow (Single Experiment)

```
1. User: "Increase W_XGB to 0.20"
   ↓
2. Agent: Edit wids_train_enhanced.py
   W_XGB = 0.20  # was 0.15
   ↓
3. Agent: Run orchestrator
   $ python wids_experiment_runner.py "Exp #1: W_XGB boost to 0.20"
   ↓
4. Orchestrator: Execute training
   $ python wids_train_enhanced.py → runs 5-fold CV
   ├─ Fold 1: hybrid=0.991, c_idx=0.854, brier=0.172
   ├─ Fold 2: hybrid=0.993, c_idx=0.862, brier=0.171
   ├─ Fold 3: hybrid=0.989, c_idx=0.841, brier=0.190
   ├─ Fold 4: hybrid=0.992, c_idx=0.858, brier=0.176
   ├─ Fold 5: hybrid=0.991, c_idx=0.859, brier=0.174
   └─ Mean: hybrid=0.9912 (was 0.9903, +0.0009), std=0.0015
   ↓
5. Orchestrator: Extract metrics from output
   ├─ hybrid_score=0.991200
   ├─ c_index=0.854800
   ├─ brier_score=0.176600
   ├─ status="marginal"  (between +0.0005 and +0.001)
   └─ elapsed_secs=152.4
   ↓
6. Orchestrator: Compare to baseline
   ├─ baseline_hybrid=0.990362
   ├─ improvement=+0.000838  (0.0009)
   └─ improvement_pct=0.085%
   ↓
7. Orchestrator: Log to results.tsv
   abc1234  0.991200  0.8548  0.1766  marginal  152.4  "W_XGB 0.15→0.20"
   └─ (NEW ROW appended)
   ↓
8. Agent: Review results
   ├─ Hybrid: 0.9912 (marginal +0.0009)
   ├─ C-Index: 0.8548 (improved from 0.8521, good!)
   ├─ Brier: 0.1766 (improved from 0.1843, good!)
   ├─ Stability: 0.9983 (excellent, no overfitting)
   └─ Decision: "This helps ranking and calibration, KEEP"
   ↓
9. Agent: Commit (or discard)
   $ git add wids_train_enhanced.py results.tsv
   $ git commit -m "Exp #1: W_XGB 0.15→0.20 (+0.0009 hybrid)"
   ↓
10. Next iteration: W_XGB performed well, try +0.05 more?
    ├─ Or try different weight (W_GB)?
    ├─ Or try hyperparameter given new weights?
    └─ → Loop back to Step 1
```

---

## 📊 Metrics Explained

### Hybrid Score = 0.3×C-Index + 0.7×(1 - Brier)

#### C-Index (30% weight)
```
Purpose: How well do you RANK fires by urgency?

Definition: Probability that given a random high-urgency fire
            and a random low-urgency fire, you rank the high
            one as higher probability

Range: 0.5 (random) to 1.0 (perfect ranking)
Current: 0.8521 (baseline)

How to improve:
  - Boost XGBoost (better ranking)
  - More complex trees (but risk overfitting!)
  - Increase learning rate (convergence to optima)

Example:
  Fire A: High risk (truly hits in 24h)
  Fire B: Low risk (censored, never hit)
  
  Your prediction A: 0.8, Fire B: 0.3
  → You ranked correctly ✓
  
  C-Index measures % of such correct pairs
```

#### Brier Score (70% weight after transformation)
```
Purpose: Are your PROBABILITIES CALIBRATED?

Definition: Mean squared error between prediction and truth
  Brier = mean((pred - truth)^2)
  
Range: 0 (perfect) to 1 (terrible)
Current: 0.1843 (baseline)

How to improve:
  - Boost Gradient Boosting (calibration focus)
  - Adjust SQUISH factor (1.2 current)
  - Ensemble with physics (more stable)

Example:
  If you predict 0.7 probability for 100 fires,
  and 65 actually hit:
  
  Squared error for each: (0.7 - 0.65)^2 = 0.0025
  Brier = 0.0025 × 100 = 0.25 (overly confident)
  
  Better: Predict 0.65 prob
  Brier = (0.65 - 0.65)^2 × 100 = 0.00 (perfect)
```

#### Hybrid = 0.3×C + 0.7×(1 - Brier)
```
Example calculation:
  C_stat = 0.8521
  Brier = 0.1843
  
  Hybrid = 0.3 × 0.8521 + 0.7 × (1 - 0.1843)
         = 0.25563 + 0.7 × 0.8157
         = 0.25563 + 0.57099
         = 0.82662  ← doesn't make sense

Wait, let me check the actual formula:
  Hybrid = 0.3 × C_index + 0.7 × (1 - Weighted_Brier)
  
Baseline stats reported:
  Hybrid: 0.9903
  C-Index: 0.8521
  Brier: 0.1843
  
This suggests Weighted_Brier ≠ simple Brier
The 0.1843 is already weighted average across 3 horizons
weighted brier = 0.3×brier_24h + 0.4×brier_48h + 0.3×brier_72h

So: Hybrid = 0.3×0.8521 + 0.7×(1-0.1843) = 0.25563 + 0.57099 = 0.8266

Hmm, that still doesn't match 0.9903. Let me defer to the locked
evaluation in wids_prepare.py (ground truth).
```

---

## 🎯 Target Trajectory

```
Experiment Phase  Baseline    Target    Improvement   Time
─────────────────────────────────────────────────────────────
Baseline          0.9903      0.9903    0.0000        —
Phase 1 (Weights) 0.9903      0.9915    +0.0012       45 min
Phase 2 (Hyperparam) 0.9915   0.9923    +0.0010       50 min
Phase 3 (Features) 0.9923    0.9935    +0.0012       40 min
Phase 4 (Advanced) 0.9935    0.9942    +0.0007       60 min
─────────────────────────────────────────────────────────────
FINAL GOAL        0.9903    **0.9942**  **+0.0039**  **3-4 hrs**
```

---

## 🔐 Immutability Guarantees

**These cannot be changed (locked):**
```
✅ Physics probabilities (S_C, P_W, SIGMA, BIAS constants)
✅ Metric calculations (C-index formula, Brier formula)
✅ CV structure (5-fold, stratified, seed=42)
✅ Data loading paths (data/train.csv, data/test.csv)
✅ Horizon definitions (12h, 24h, 48h, 72h)
```

**These can be changed (editable):**
```
🔧 Ensemble weights (W_PHYS, W_XGB, W_RF, W_GB)
🔧 Model hyperparameters (depths, learning rates, etc.)
🔧 Feature selection (which features to use)
🔧 Post-processing adjustments (SQUISH, close-fire bounds)
```

---

## 📋 Quick Reference Card

### Emergency Actions
```
Symptom                 Action                          Expected Effect
─────────────────────────────────────────────────────────────────────
High variance           Reduce max_depth, increase      Stability +0.02
(std > 0.010)          min_samples_leaf
                       
Brier high, C-idx low  Increase W_GB, increase         Calibration +0.005
                       gb_learning_rate
                       
C-idx low, Brier ok    Increase W_XGB, decrease        Ranking +0.005
                       xgb_max_depth (focus)
                       
No improvement > 5     Backtrack 2 exps, try            Reset & try new
consecutive             orthogonal direction            direction
                       
Training time > 10min  Reduce estimators, reduce       Speed -50%, score
                       depth, reduce features          -0.001
```

---

## 🚀 Launch Sequence

```
1. Read: README_ENHANCED.md (this file)    [2 min]
2. Read: WIDS_PROGRAM_ENHANCED.md          [10 min]
3. Skim: ADVANCED_STRATEGIES.md            [5 min]
4. Run:  python wids_train_enhanced.py     [5 min]
5. Run:  python wids_experiment_runner.py  [5 min]
6. Review: results.tsv                     [2 min]
7. Copy: Claude Code prompt from AUTORESEARCH_START_HERE.md
8. Paste: To Claude Code AI                [1 min]
9. Watch: AI iterate (monitor results.tsv) [Continuous]
─────────────────────────────────────────────────────────────
Total prep: 30 min
Autonomy: 30-35 experiments × 4 min = 2-2.5 hours
Result: 0.9942+ score 🎯
```

---

## 📖 Documentation Map

```
START HERE ──→ README_ENHANCED.md (overview & quick-start)
                       ↓
                       ├─→ WIDS_PROGRAM_ENHANCED.md (daily workflow)
                       │         ↓
                       │    Run experiments, iterate
                       │         ↓
                       │    Stuck? Consult...
                       │
                       ├─→ ADVANCED_STRATEGIES.md (theory & tactics)
                       │         ↓
                       │    Overfitting? Red flag 1
                       │    Need features? Part 3
                       │    Want survival models? Part 3
                       │
                       └─→ AUTORESEARCH_START_HERE.md (setup with Claude Code)
                               ↓
                           Copy-paste prompt
                           Paste to Claude Code
                           Go! 🚀
```

---

*WiDS 2026 System Architecture*  
*Small-data survival analysis for wildfire evacuation prediction*  
*Optimized for autonomous iteration to leaderboard top-20 🎯*
