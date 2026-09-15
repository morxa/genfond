#!/bin/bash
#
# Fixed A/B benchmark protocol. Every hypothesis branch runs exactly this so results compare.
#
#   TAG=baseline scripts/ab_bench.sh
#
# Runs each suite in domains/suites/ (see SUITES) for each POLICY_TYPE, sequentially, seeded and
# single-threaded (see docs/experiments-log.md for why both are mandatory), under a memory cap
# and a wall-clock limit, and appends one row per suite x type to $RESDIR/stats.csv.
#
# Knobs (environment variables):
#   TAG          label; results land in results-<timestamp>-<TAG>/ (required)
#   SUITES       space-separated suite directories under domains/suites/
#   POLICY_TYPE  space-separated --type values, e.g. "datalog datalog-sig"
#   SEED         --seed for the execution RNG (default 0)
#   THREADS      -n for clingo (default 1; anything else makes runs non-reproducible)
#   MAX_MEMORY   --max-memory in MB (default 24000)
#   TIME_LIMIT   wall-clock limit per suite x type, passed to timeout(1) (default 2h)
#   CONFIG       extra --config file
#   RUNNER       how to invoke python; "poetry run" locally, or
#                "apptainer run --bind $PWD genfond_env.sif" on a machine with the image
#   EXTRA_ARGS   anything else to append to the genfond command line
set -u
TAG="${TAG:?set TAG to label this run}"
SUITES="${SUITES:-blocks3ops-local gripper-local miconic-local blocks4ops-clear-local delivery-local}"
POLICY_TYPE="${POLICY_TYPE:-datalog}"
SEED="${SEED:-0}"
# State is a frozenset of pddl atoms, so iteration order and hence the loop path follow the hash
# seed; without pinning it two seeded runs still differ.
export PYTHONHASHSEED="${PYTHONHASHSEED:-$SEED}"
THREADS="${THREADS:-1}"
MAX_MEMORY="${MAX_MEMORY:-24000}"
TIME_LIMIT="${TIME_LIMIT:-2h}"
CONFIG="${CONFIG:+--config $CONFIG}"
RUNNER="${RUNNER:-poetry run}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$(dirname "$0")/.."
RESDIR="results-$(date -Iseconds)-$TAG"
mkdir -p "$RESDIR/out"
{
  echo "commit: $(git rev-parse HEAD)"
  echo "branch: $(git branch --show-current)"
  echo "host: $(hostname)"
  echo "suites: $SUITES"
  echo "policy_type: $POLICY_TYPE"
  echo "seed: $SEED threads: $THREADS max_memory: $MAX_MEMORY time_limit: $TIME_LIMIT"
  echo "config: $CONFIG extra_args: $EXTRA_ARGS"
  echo "dirty:"
  git status --short | grep -v '^??'
} > "$RESDIR/run-info.txt"
echo "results in $RESDIR"

for suite in $SUITES; do
  dir="domains/suites/$suite"
  domainfile="$dir/domain.pddl"
  problemfiles=$(find -L "$dir" ! -name domain.pddl -name '*.pddl' | sort)
  for ptype in $POLICY_TYPE; do
    name="$suite-$ptype"
    log="$RESDIR/out/$name.log"
    echo "== $name ($(echo $problemfiles | wc -w) problems) $(date +%T)"
    start=$(date +%s)
    timeout --signal=INT --kill-after=60 "$TIME_LIMIT" \
      $RUNNER python -m genfond -v --name "$suite" -n "$THREADS" --seed "$SEED" \
        --max-memory "$MAX_MEMORY" --type "$ptype" $CONFIG $EXTRA_ARGS \
        --dump-config "$RESDIR/$name.yaml" -o "$RESDIR/$name.policy" \
        --stats "$RESDIR/stats.csv" $domainfile $problemfiles > "$log" 2>&1
    rc=$?
    echo "$name rc=$rc wall=$(( $(date +%s) - start ))s" >> "$RESDIR/exit-codes.txt"
    if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then echo "   TIMEOUT after $TIME_LIMIT"; fi
    grep -h 'Policy solves' "$log" | tail -1 | sed 's/^/   /'
  done
done
poetry run python scripts/ab_summary.py "$RESDIR" 2>/dev/null || python scripts/ab_summary.py "$RESDIR"
