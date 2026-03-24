# WiDS Autonomous Research Setup - Complete Guide

## What You Now Have

A **Karpathy-style autonomous research system** for the WiDS 2026 wildfire competition. The AI (Claude Code) can:

- ✅ Iterate on ensemble weights
- ✅ Tune hyperparameters
- ✅ Engineer new features
- ✅ Add/test new models
- ✅ Track all experiments
- ✅ Commit working changes to git
- ✅ Run 20-30 experiments per night (~3 min each)

---

## File Structure

```
autoresearch/
├── WIDS_SETUP.md              ← Architecture & philosophy
├── README.md                  ← Original (ignore, use WIDS_SETUP.md)
├── wids_prepare.py            ← [LOCKED] Data, metrics, CV setup
├── wids_train.py              ← [AGENT MODIFIES] Ensemble definition
├── wids_program.md            ← [HUMAN UPDATES] AI agent instructions
├── wids_experiment_runner.py   ← [LOCKED] Orchestrator
├── results.tsv                ← [AUTO] Experiment log
├── logs/                       ← [AUTO] Per-experiment output
├── best_model.pkl             ← [AUTO] Best model weights (future)
└── [old files]                ← (prepare.py, train.py, etc - ignore)
```

**Key principle:** Only 1 file is modified by the agent (`wids_train.py`). Everything else locked.

---

## How It Works (4 Steps)

### Step 1: AI Proposes Experiment

Claude AI reads `wids_program.md`:
- Currently at 0.990362 (baseline)
- Target: 0.994+
- Physics dominates but C-index (30% weight) needs work
- **Proposal:** Increase XGBoost weight from 0.10 → 0.15

### Step 2: AI Modifies Code

Edit `wids_train.py`:
```python
W_PHYS = 0.75        # decreased from 0.80
W_XGB = 0.15         # increased from 0.10
W_RF = 0.05          # unchanged
W_GB = 0.05          # unchanged
```

### Step 3: AI Runs Experiment

```bash
python wids_experiment_runner.py "increased W_XGB to 0.15"
```

The orchestrator:
1. Calls `wids_train.py` → 5-fold CV
2. Extracts: hybrid_score, c_index, brier_score, time
3. Compares to baseline (0.990362)
4. Logs to `results.tsv`
5. Decides: KEEP (+0.001?) or DISCARD (-0.0005)

### Step 4: Commit or Iterate

- **If +0.001 improvement:** Git commit, move to next experiment
- **If marginal (+0.0005):** Discuss with user complexity trade-off
- **If negative:** Discard, try different direction

---

## Running the System

### Setup (One-time)

```bash
cd /home/raj/Documents/CODING/Kaggle/WiDS_GLOBAL_Datathon_2026

# Create experiment branch
git checkout -b autoresearch/wids-experiment-1

# Verify setup
cd autoresearch
python wids_prepare.py      # Should succeed (data loading check)
python wids_train.py        # Should complete baseline (takes ~3 min)
```

### Running with Claude Code

1. Open this project in VS Code
2. Open Claude Code (or your AI assistant)
3. Paste this prompt:

```
Hi! I'm ready to start autonomous WiDS experiments.

Read these files first:
- autoresearch/WIDS_SETUP.md (architecture)
- autoresearch/wids_program.md (agent instructions)
- autoresearch/results.tsv (current baseline)
- autoresearch/wids_train.py (ensemble definition)

Current baseline: 0.990362 hybrid score

Please review the system, then propose Experiment #1 (with hypothesis and expected impact), make the code change, and run it!
```

4. Claude will:
   - Read files
   - Propose experiment
   - Modify `wids_train.py`
   - Run: `python wids_experiment_runner.py "your description"`
   - Log results
   - Suggest next experiment

5. **Loop:** Keep going until you say stop or reach target

### Manual Check During Experiments

```bash
# Watch progress
tail -f autoresearch/results.tsv

# Best score so far
awk -F'\t' '{if($4=="keep") print $2}' autoresearch/results.tsv | sort -rn | head -1

# Git history
git log --oneline | head -20

# Latest experiment log
ls -ltr autoresearch/logs/ | tail -1 | awk '{print $NF}' | xargs cat
```

---

## Experiment Strategy (Template)

### Phase 1: Ensemble Weights (Experiments 1-8)

**Hypothesis:** Physics (0.80) strong, but C-index (30%) lagging → boost XGB/RF

**Experiments:**
1. W_PHYS=0.75, W_XGB=0.15, RF=0.05, GB=0.05
2. W_PHYS=0.70, W_XGB=0.20, RF=0.05, GB=0.05
3. W_PHYS=0.68, W_XGB=0.12, RF=0.10, GB=0.10 (diversify)
4. Same as #3 + SQUISH=1.30 (calibration)
5-8. Grid search best 2×2 from top 4

**Expected:** +0.002 hybrid score improvement

### Phase 2: Hyperparameters (Experiments 9-16)

**Hypothesis:** Given optimized weights, tune model capacity

**Focus:** XGBoost (since boosted), Random Forest

```
If #1 succeeded (W_XGB=0.15):
- xgb_max_depth: 4 → 5 → 6
- xgb_n_estimators: 150 → 200
- rf_n_estimators: 250 → 350
```

**Expected:** +0.001 improvement

### Phase 3: Features (Experiments 17-24)

**Hypothesis:** 31 features sufficient? Can we add 2-3 high-value features?

**Examples:**
```python
# New features to try:
'wind_dynamic' = closing_speed * alignment_abs**2
'critical_zone' = (is_night * alignment_abs) > 0.5
'growth_trend' = area_growth_rate / (radial_growth_rate + 1)
```

**Expected:** +0.002 improvement

### Phase 4: Advanced (Experiments 25+)

**Option A:** New model type (CoxPH, SurvivalForest)
**Option B:** Stacking (meta-model on CV predictions)
**Option C:** Horizon-specific weights

**Expected:** +0.001 improvement → **Target: 0.994**

---

## Key Metrics (Ground Truth)

These are **immutable** in `wids_prepare.py`:

### Hybrid Score (Primary)
```
Hybrid = 0.3 × C-Index + 0.7 × (1 - Weighted Brier)
```

### C-Index (30% weight)
- Measures ranking: how well you prioritize urgent fires
- Range: 0.5 (random) to 1.0 (perfect)
- Calculated on 72h predictions

### Weighted Brier Score (70% weight)
```
Weighted = 0.3 × Brier(24h) + 0.4 × Brier(48h) + 0.3 × Brier(72h)
```
- Measures calibration: are your probabilities correct?
- Lower is better (0-1)
- Censor-aware (excludes missing data)

---

## Decision Thresholds

### Keep (✅)
```
- Hybrid improvement ≥ +0.0010, OR
- Same score + simpler code, OR
- +0.0005 improvement + significantly cleaner code
```

### Marginal (⚠️)
```
- +0.0005 to +0.0010 improvement
- Check code complexity before commit
```

### Discard (❌)
```
- <+0.0005 improvement
- Hybrid score decreased
- Crash/NaN outputs
- +0.0002 improvement + 50+ lines of code added
```

---

## Results.TSV Format

Tracked automatically:

```
commit              hybrid_score    c_index    brier_score    status      time_sec    description
baseline            0.990362        0.8521     0.1843         keep        147.3       Initial weights
abc1234             0.991850        0.8624     0.1761         keep        148.5       W_PHYS=0.75, W_XGB=0.15
def5678             0.989900        0.8410     0.1920         discard     146.0       Decreased SQUISH
```

**Columns:**
- `commit`: Git hash (short)
- `hybrid_score`: Primary metric (higher better)
- `c_index`: From 0.5-1.0
- `brier_score`: Lower is better (0-1)
- `status`: keep/discard/marginal/crash
- `time_sec`: Wall-clock seconds for CV
- `description`: What was tried

---

## Failure Handling

### "Command failed with code X"
→ Check `logs/experiment_*.log` for error details
→ Usually: missing features, typo in hyperparameter range
→ Fix and re-run

### "Could not parse metrics"
→ `wids_train.py` might have syntax error
→ Check Python syntax: `python -m py_compile wids_train.py`
→ Verify it runs manually: `python wids_train.py`

### "VRAM exceeded"
→ Reduce model capacity: fewer estimators, smaller depth
→ Or: run with fewer folds (for testing only)

### "No improvement for 5 experiments"
→ Try orthogonal direction (different part of parameter space)
→ Or add features (Phase 3)
→ Or backtrack and try simpler ensemble

---

## Integration with Your Submission

Once you find best model (say, 0.994 hybrid):

1. Train final model on full train set (no CV)
2. Generate test predictions
3. Save to `submission_final.csv`
4. Upload to Kaggle competition

```python
# In wids_train.py, add after CV loop:
# Train on full set
best_ensemble_config = {...}  # From best CV experiment
# Train XGB, RF, GB on full train
# Generate test predictions
# Save submission
```

---

## Success Criteria

| Target | Timeline | Status |
|--------|----------|--------|
| Baseline: 0.990362 | ✅ Achieved | Baseline |
| +0.002 (0.992362) | Week 1 (Experiments 1-8) | Phase 1 |
| +0.003 (0.993362) | Week 2 (Experiments 9-16) | Phase 2 |
| +0.004 (0.994362) | Week 3 (Experiments 17-24) | Phase 3 |
| Top 20 leaderboard | Week 4 (Experiments 25+) | Advanced |

---

## Checklist

Before starting AI agent:

- [ ] `wids_prepare.py` loads data OK
- [ ] `wids_train.py` runs to completion manually
- [ ] `wids_experiment_runner.py` works and logs results
- [ ] `results.tsv` has baseline entry
- [ ] `wids_program.md` is clear and matches your understanding
- [ ] Git branch `autoresearch/wids-experiment-1` created
- [ ] Claude Code or AI assistant is ready

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'xgboost'"

```bash
cd autoresearch
pip install xgboost scikit-learn scikit-survival lifelines
# or if using conda
conda install xgboost scikit-learn scikit-survival lifelines
```

### "FileNotFoundError: data/train.csv"

```bash
# From autoresearch/ dir, run:
cd ..
# Verify data exists:
ls -la data/
# Then run experiments from autoresearch/:
cd autoresearch
python wids_experiment_runner.py "test"
```

### Experiment takes >5 min per fold

→ Hyperparameters too expensive
→ Reduce: n_estimators, max_depth, or learning_rate
→ Example: `xgb_n_estimators = 100` (instead of 150)

### Results.tsv keeps showing "crash"

→ Check latest log: `autoresearch/logs/experiment_*.log`
→ Usually: feature name typo, hyperparameter value invalid
→ Fix in `wids_train.py` and re-run

---

## Questions?

Refer to:
1. **`wids_program.md`** - What to modify and why
2. **`wids_prepare.py`** - Metrics and data (locked, read-only)
3. **`wids_train.py`** - Ensemble definition (agent edits here)
4. **`results.tsv`** - Track progress
5. **`WIDS_SETUP.md`** - System philosophy

Good luck! 🚀

---

**Next step:** Open Claude Code and paste the setup prompt above to begin autonomous iteration.
