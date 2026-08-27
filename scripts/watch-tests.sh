#!/usr/bin/env bash
# Run the cheap suites in a loop, so a long training run cannot quietly break
# the rest of the project while nobody is looking.
#
#     ./scripts/watch-tests.sh [seconds-between-rounds]   # default 300
#
# Deliberately NOT `npm run smoke`: that spawns four sidecars, and during a
# local train they fight the trainer for the same 16 threads. Smoke is the
# after-training check, not the during-training one.
set -u
cd "$(dirname "$0")/.."
GAP=${1:-300}
[ "${1:-}" = "--once" ] && GAP=0
PY=.venv/bin/python
round=0

while true; do
  round=$((round + 1))
  fails=0
  printf '\n=== round %d  %s\n' "$round" "$(date +%H:%M:%S)"
  while IFS='|' read -r name cmd; do
    [ -z "$name" ] && continue
    if out=$(eval "$cmd" 2>&1); then
      printf '  ok    %s\n' "$name"
    else
      fails=$((fails + 1))
      printf '  FAIL  %s\n%s\n' "$name" "$(echo "$out" | tail -15)"
    fi
  done <<EOF
tsc|npm run check
selfcheck|npm run selfcheck
sidecar|$PY ml/sidecar.py --selfcheck --camera X --source X
dataset|$PY ml/prepare_dataset.py --selfcheck
reader|OMP_NUM_THREADS=2 $PY ml/train_reader.py --selfcheck
EOF
  printf '=== round %d: %d failure(s)\n' "$round" "$fails"
  [ "$GAP" = 0 ] && exit $((fails > 0))
  sleep "$GAP"
done
