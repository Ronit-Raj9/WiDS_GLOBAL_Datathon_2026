#!/bin/bash
set -euo pipefail

MAX=${1:-150}
ITER=0
DONE="WIDS_DONE"

cd "$(dirname "$0")"

echo "=== WiDS Autonomous Loop — $MAX iterations ==="
echo "Target: 0.990+ (Rank 1)"
echo "Current best: Check results.tsv"

autorevert() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git checkout -- wids_train_enhanced.py 2>/dev/null || true
  fi
}

while [ "$ITER" -lt "$MAX" ]; do
  ITER=$((ITER + 1))
  echo "--- Iteration $ITER / $MAX ---"

  OUTPUT=$(claude -p "
Read CLAUDE.md and results.tsv for current best score.

CURRENT STATUS:
- Best CV score so far: Check results.tsv for highest 'keep' score
- Target: 0.990094 (Rank 1 on leaderboard)
- Data: n=221 fires (small data regime)

YOUR TASK:
1. Read results.tsv to find the current best hybrid_score and what configuration achieved it
2. Read wids_train_enhanced.py to understand the current configuration
3. Make ONE targeted improvement to wids_train_enhanced.py:
   - Try advanced calibration methods (isotonic regression, Platt scaling)
   - Try different ensemble weights around the best known config
   - Try adding advanced features (interaction terms, polynomial features)
   - Try advanced ML techniques (CoxPH, survival forests if available)
   - Try probability post-processing adjustments
4. Run: python wids_experiment_runner.py 'auto-ralph-\$ITER'
5. If hybrid_score improved by >= 0.0003 vs current best:
   - Keep it and git commit with message 'auto keep \$ITER - score'
6. If not improved: run 'git checkout -- wids_train_enhanced.py' to revert
7. If hybrid_score >= 0.990 output exactly: \$DONE

Focus on small-data appropriate techniques to avoid overfitting.
" --allowedTools "Read,Edit,Bash,Glob,Grep" 2>&1 || true)

  echo "$OUTPUT"
  echo "---"

  if echo "$OUTPUT" | grep -q "$DONE"; then
    echo "=== TARGET REACHED ==="
    exit 0
  fi

  sleep 3
done

echo "=== $MAX iterations done. Check results.tsv ==="