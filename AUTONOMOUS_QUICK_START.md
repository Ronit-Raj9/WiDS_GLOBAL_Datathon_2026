# WiDS 2026 Autonomous Optimization - Quick Reference

## Fastest Way to Run Fully Autonomous

### Option 1: Single Command (Recommended)

```bash
cd /home/raj/Documents/CODING/Kaggle/WiDS_GLOBAL_Datathon_2026

# Run 30 experiments autonomously
python wids_autonomous_runner.py 30
```

### Option 2: Overnight Unattended Run

```bash
# Start overnight batch (50 experiments)
nohup python wids_autonomous_runner.py night batch > night.out 2>&1 &

# Check progress
tail -f night.out
cat results.tsv
```

### Option 3: Quick Test

```bash
# Test with 2 experiments
python wids_autonomous_runner.py test
```

---

## How It Works

The autonomous runner:
1. **Proposes experiments** based on predefined strategy phases
2. **Modifies** `wids_train_enhanced.py` (weights, hyperparameters, calibration)
3. **Runs training** and evaluates results
4. **Keeps improvements** (>= 0.0010 hybrid score improvement)
5. **Reverts** unsuccessful changes
6. **Tracks progress** in `results.tsv`
7. **Stops early** if no improvement after 5 experiments

---

## Strategy Phases

| Phase | Experiments | What Changes | Expected Gain |
|-------|-------------|--------------|---------------|
| 1. Ensemble Weights | 1-10 | W_PHYS, W_XGB, W_RF, W_GB | +0.0012 |
| 2. Hyperparameters | 11-20 | Tree depth, learning rates, regularization | +0.0010 |
| 3. Calibration | 21-25 | Probability squish power | +0.0005 |
| 4. Random Explore | 26-30 | Mixed exploration | +0.0003 |

---

## Monitoring Progress

### Check Results

```bash
# View all results
cat results.tsv

# Best score so far
grep "keep" results.tsv | sort -t$'\t' -k2 -rn | head -1

# Latest 5 experiments
tail -5 results.tsv
```

### Check Logs

```bash
# Latest experiment log
ls -lt logs/ | head -2
cat logs/experiment_*.log | tail -50
```

---

## File Roles (Locked-Edit-Locked)

| File | Role | Editable? |
|------|------|-----------|
| `wids_prepare.py` | Evaluation harness | LOCKED |
| `wids_experiment_runner.py` | Orchestrator | LOCKED |
| `wids_train_enhanced.py` | Training code | EDITABLE by agent |
| `wids_autonomous_runner.py` | Autonomous loop | DO NOT EDIT |
| `results.tsv` | Progress tracking | AUTO-UPDATED |

---

## Command Examples

```bash
# Run specific number of experiments
python wids_autonomous_runner.py 20

# Quick validation (2 experiments)
python wids_autonomous_runner.py test

# Overnight batch (50 experiments)
python wids_autonomous_runner.py night batch

# Manual single experiment
python wids_experiment_runner.py "Manual test description"
```

---

## Expected Output

```
==================================================
EXPERIMENT 1/30
Phase: 1 | Type: weights
==================================================

Configuration: Phase 1: More XGBoost for C-index

======================================================================
Experiment: Phase 1: More XGBoost for C-index
======================================================================

 KEEP: 0.9550 (+0.0038 vs best)

PROGRESS SUMMARY
======================================================================
Total experiments: 1
Kept: 1

 Best score: 0.9550
   C-Index: 1.0000
   Brier: 0.0643
```

---

## Troubleshooting

### Script crashes
- Check `logs/` for detailed error logs
- Verify data files exist in `data/` directory
- Ensure dependencies installed: `pip install xgboost scikit-learn pandas numpy`

### No improvements found
- Check if baseline is already optimal
- Review `ADVANCED_STRATEGIES.md` for advanced tactics
- Consider adding new features instead of hyperparameter tuning

### High fold variance (>0.010)
- Increase regularization (higher min_child_weight, lower max_depth)
- Increase W_PHYS weight (more physics, less ML)
- Reduce number of features

---

## Current Best Configuration

Check `results.tsv` for the current best:

```bash
grep "keep" results.tsv | awk -F'\t' 'BEGIN{max=0} {if($2>max){max=$2; line=$0}} END{print line}'
```

---

## Next Steps After Autonomous Run

1. **Review results**: Identify which changes were kept
2. **Analyze patterns**: What types of changes consistently improve?
3. **Plan Phase 2**: Based on results, adjust strategy
4. **Manual refinement**: Fine-tune promising configurations

---

*Built for WiDS 2026 Kaggle Competition*
*Autonomous optimization for small-data survival analysis*
