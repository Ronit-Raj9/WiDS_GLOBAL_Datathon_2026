# WiDS 2026 Autonomous Optimization System

## Fastest Way to Run Fully Autonomous

```bash
cd /home/raj/Documents/CODING/Kaggle/WiDS_GLOBAL_Datathon_2026

# Run 30 experiments automatically (recommended)
python wids_autonomous_runner.py 30

# Quick test (2 experiments)
python wids_autonomous_runner.py test

# Overnight unattended batch (50 experiments)
nohup python wids_autonomous_runner.py night batch > night.out 2>&1 &
```

**That's it!** The autonomous runner will:
- Propose and run experiments
- Keep improvements (>= 0.0010)
- Revert unsuccessful changes
- Track progress in `results.tsv`
- Stop early if no improvement after 5 experiments

---

## Quick Reference

### Monitor Progress

```bash
# View results
cat results.tsv

# Best score
grep "keep" results.tsv | sort -t$'\t' -k2 -rn | head -1

# Latest logs
ls -lt logs/ | head -2
```

### Manual Mode

```bash
# Single experiment with custom description
python wids_experiment_runner.py "My experiment description"
```

---

## System Overview

### File Roles (Locked-Edit-Locked)

| File | Role | Editable? |
|------|------|-----------|
| `wids_prepare.py` | Evaluation harness | LOCKED |
| `wids_experiment_runner.py` | Orchestrator | LOCKED |
| `wids_train_enhanced.py` | Training code | EDITABLE by agent |
| `wids_autonomous_runner.py` | Autonomous loop | DO NOT EDIT |
| `results.tsv` | Progress tracking | AUTO-UPDATED |

### Strategy Phases

| Phase | Experiments | What Changes |
|-------|-------------|--------------|
| 1. Ensemble Weights | 1-10 | W_PHYS, W_XGB, W_RF, W_GB |
| 2. Hyperparameters | 11-20 | Tree depth, learning rates, regularization |
| 3. Calibration | 21-25 | Probability calibration power |
| 4. Random Explore | 26-30 | Mixed exploration |

---

## Documentation

- **[AUTONOMOUS_QUICK_START.md](AUTONOMOUS_QUICK_START.md)** - Complete quick reference
- **[ADVANCED_STRATEGIES.md](ADVANCED_STRATEGIES.md)** - Theory and tactics
- **[WIDS_PROGRAM_ENHANCED.md](WIDS_PROGRAM_ENHANCED.md)** - Detailed workflow

---

## Current Status

**Baseline:** ~0.951-0.953 hybrid score
**Target:** 0.994+ (top 20 leaderboard)
**Data:** 221 fires (69 hits, 152 censored)

---

*Built for WiDS 2026 Kaggle Competition*
