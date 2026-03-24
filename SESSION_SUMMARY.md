# WiDS Autonomous Optimization - Session Summary

## System Status: READY

The autonomous optimization system is fully configured and tested.

---

## Quick Start Commands

### Run 30 Experiments (Recommended)

```bash
cd /home/raj/Documents/CODING/Kaggle/WiDS_GLOBAL_Datathon_2026
python wids_autonomous_runner.py 30
```

### Overnight Unattended Run

```bash
nohup python wids_autonomous_runner.py night batch > night.out 2>&1 &
```

### Monitor Progress

```bash
# Real-time results
watch -n 5 'cat results.tsv'

# Best score
grep "keep" results.tsv | sort -t$'\t' -k2 -rn | head -1
```

---

## Test Results (Initial Validation)

**Experiments Run:** 8
**Best Score:** 0.956781
**Improvement:** +0.005656 from initial baseline (~0.951)

### Kept Configurations

| Exp | Score | C-Index | Brier | Configuration |
|-----|-------|---------|-------|---------------|
| 001 | 0.95497 | 1.0 | 0.064329 | W_PHYS=0.70, W_XGB=0.20 |
| 003 | 0.95678 | 1.0 | 0.061741 | W_PHYS=0.65, W_XGB=0.25 (BEST) |

### Key Findings

1. **Higher XGBoost weight improves score** - W_XGB=0.25 outperforms lower weights
2. **C-Index is perfect (1.0)** - Ranking is excellent across all configurations
3. **Brier score drives improvements** - Lower Brier = better calibration
4. **Optimal balance:** W_PHYS=0.65, W_XGB=0.25

---

## Next Steps

### Option 1: Continue from Current Best

The system will automatically continue from the current best configuration (0.956781).

```bash
python wids_autonomous_runner.py 30
```

### Option 2: Reset and Start Fresh

```bash
# Reset results
echo -e "commit\thybrid_score\tc_index\tbrier_score\tstatus\ttime_sec\tdescription" > results.tsv

# Start fresh optimization
python wids_autonomous_runner.py 30
```

### Option 3: Manual Fine-Tuning

Based on findings, manually adjust `wids_train_enhanced.py`:

```python
# Current best weights
W_PHYS = 0.65
W_XGB = 0.25
W_RF = 0.05
W_GB = 0.05
```

---

## Strategy Phases

### Phase 1: Ensemble Weights (Exp 1-10) ✅ IN PROGRESS

Testing different balances of physics vs ML models.

**Key insight:** Higher XGBoost weight (0.25) improves performance.

### Phase 2: Hyperparameters (Exp 11-20)

Will tune:
- Tree depths (2-4)
- Learning rates (0.02-0.10)
- Regularization (min_child_weight, reg_lambda)
- Sampling rates (subsample, colsample_bytree)

### Phase 3: Calibration (Exp 21-25)

Will tune probability calibration power (1.0-1.4).

### Phase 4: Random Exploration (Exp 26-30)

Mixed exploration of promising configurations.

---

## File Structure

```
WiDS_GLOBAL_Datathon_2026/
├── wids_autonomous_runner.py    # Main autonomous loop
├── wids_experiment_runner.py    # Single experiment runner
├── wids_train_enhanced.py       # Training code (EDITABLE)
├── wids_prepare.py              # Evaluation harness (LOCKED)
├── results.tsv                   # Progress tracking
├── logs/                         # Experiment logs
└── data/                         # Train/test CSV files
```

---

## Expected Runtime

- **Per experiment:** ~6-8 seconds (5-fold CV)
- **30 experiments:** ~3-4 minutes
- **Overnight batch (50):** ~5-7 minutes

---

## Success Criteria

| Target | Score | Status |
|--------|-------|--------|
| Baseline | 0.951 | ✅ Achieved |
| Phase 1 | 0.955 | ✅ Achieved (0.95678) |
| Phase 2 | 0.960 | 🎯 Next target |
| Phase 3 | 0.965 | Pending |
| Final | 0.970+ | Pending |

---

*System ready for full 30-experiment run*
*Last updated: 2026-03-24 21:16*
