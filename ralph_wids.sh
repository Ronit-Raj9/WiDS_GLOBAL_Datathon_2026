#!/bin/bash
set -euo pipefail

MAX=${1:-150}
ITER=0
DONE="WIDS_DONE"

cd "$(dirname "$0")"

echo "=== WiDS Autonomous Loop — $MAX iterations ==="

autorevert() {
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git checkout -- wids_train_enhanced.py 2>/dev/null || true
  fi
}

while [ "$ITER" -lt "$MAX" ]; do
  ITER=$((ITER + 1))
  echo "--- Iteration $ITER / $MAX ---"

  OUTPUT=$(claude -p "
Read CLAUDE.md for instructions.
Read results.tsv to find current best hybrid_score.
Make ONE improvement to wids_train_enhanced.py.
Run: python wids_experiment_runner.py 'auto-$ITER'
If score improved by >= 0.0010 vs current best: keep it and git commit with message 'auto keep $ITER'.
If not improved: run 'git checkout -- wids_train_enhanced.py' to revert.
If hybrid_score >= 0.9940 output exactly: $DONE
" --allowedTools "Read,Edit,Bash,Write" 2>&1 || true)

  echo "$OUTPUT"
  echo "---"

  if echo "$OUTPUT" | grep -q "$DONE"; then
    echo "=== TARGET REACHED ==="
    exit 0
  fi

  sleep 5
done

echo "=== $MAX iterations done. Check results.tsv ==="