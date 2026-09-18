#!/bin/bash
#
# Held-out generalisation check for scripts/ablation.sh results. Runs on the workstation, not
# SLURM (see AGENTS.md: local runs stay small -- this only executes already-trained policies).
#
#   scripts/heldout_eval.sh <results-dir>
#
# For every <arm>-<domain>-s<seed>.policy in <results-dir> whose domain has a held-out test
# directory below, evaluates it with scripts/eval_policy.py (--seed 0 -i 2) inside apptainer
# against that directory's problems, in parallel. Writes <results-dir>/heldout/<name>.log per
# policy and appends one line per policy to <results-dir>/heldout/summary.txt:
#   <arm> <domain> <seed> <solved>/<n> [rules=<k>]
#
# Knobs (environment variables):
#   PAR            parallel evaluations (default 8)
#   EVAL_TIMEOUT   per-policy timeout in seconds, passed to timeout(1) (default 1800)
set -u
RESDIR="${1:?usage: scripts/heldout_eval.sh <results-dir>}"
PAR="${PAR:-8}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-1800}"
export EVAL_TIMEOUT

cd "$(dirname "$0")/.."
mkdir -p "$RESDIR/heldout"
: > "$RESDIR/heldout/summary.txt"
export RESDIR

# The arm names actually present in this results dir (from its manifests), so a policy file name
# like "noH27-blocks4ops-flat-s0" -- where the domain itself contains a hyphen -- can still be
# split correctly into arm/domain/seed by stripping a known arm prefix, rather than guessing from
# hyphen positions. Falls back to ablation.sh's default arm names if no manifest is present.
arms=$(cd "$RESDIR" && ls manifest-*.txt 2>/dev/null | sed -e 's/^manifest-//' -e 's/\.txt$//')
if [ -z "$arms" ]; then
  arms="full noH27 noH29 noH30 noH32 noH33"
fi
export ABLATION_ARMS="$arms"

run_one() {
  local policy="$1"
  local name; name=$(basename "$policy" .policy)

  # Domain -> held-out test directory (old domain layout on this branch; see AGENTS.md and the
  # 2026-09-17 "benchmarks reorganised" note in docs/experiments-log.md). Domains not listed here
  # (blocks4ops-flat, blocks, miconic, visitall, barman, ...) have no held-out set and are skipped.
  declare -A test_dir=(
    [blocks3ops]="domains/deterministic/blocks3ops-heldout"
    [delivery]="domains/deterministic-new/delivery/test"
    [logistics]="domains/deterministic-new/logistics/test"
    [logistics_dp]="domains/deterministic-new/logistics_dp/test"
    [reward]="domains/deterministic-new/reward/test"
    [spanner]="domains/deterministic-new/spanner/test"
    [grid]="domains/deterministic-new/grid/test"
    [gripper]="domains/deterministic-new/gripper/test"
  )

  local arm="" rest=""
  for a in $ABLATION_ARMS; do
    case "$name" in
      "$a"-*)
        arm="$a"
        rest="${name#"$a"-}"
        break
        ;;
    esac
  done
  if [ -z "$arm" ]; then
    echo "skip (unknown arm prefix): $name" >&2
    return
  fi
  if [[ ! "$rest" =~ ^(.+)-s([0-9]+)$ ]]; then
    echo "skip (name doesn't match <domain>-s<seed>): $name" >&2
    return
  fi
  local domain="${BASH_REMATCH[1]}"
  local seed="${BASH_REMATCH[2]}"

  local testdir="${test_dir[$domain]:-}"
  [ -z "$testdir" ] && return
  if [ ! -d "$testdir" ]; then
    echo "skip ($domain): test dir missing: $testdir" >&2
    return
  fi
  local domainfile="$testdir/domain.pddl"
  local problems
  problems=$(find -L "$testdir" -maxdepth 1 -name '*.pddl' ! -name domain.pddl)
  if [ -z "$problems" ]; then
    echo "skip ($domain): no problems in $testdir" >&2
    return
  fi

  local log="$RESDIR/heldout/$name.log"
  timeout "$EVAL_TIMEOUT" apptainer run --bind "$PWD" --env PYTHONPATH="$PWD" genfond_env.sif \
    python scripts/eval_policy.py --seed 0 -i 2 "$domainfile" "$policy" $problems > "$log" 2>&1

  local solved="?" n="?"
  local m
  m=$(grep -o 'Policy solves [0-9]* out of [0-9]*' "$log" | tail -1)
  if [ -n "$m" ]; then
    solved=$(echo "$m" | awk '{print $3}')
    n=$(echo "$m" | awk '{print $6}')
  fi

  local rules=""
  local repr_line
  repr_line=$(apptainer run --bind "$PWD" --env PYTHONPATH="$PWD" genfond_env.sif \
    python print_policy.py "$policy" 2>/dev/null | head -1)
  if [[ "$repr_line" =~ ^([0-9]+)\ rules$ ]]; then
    rules="${BASH_REMATCH[1]}"
  fi

  {
    if [ -n "$rules" ]; then
      echo "$arm $domain $seed $solved/$n rules=$rules"
    else
      echo "$arm $domain $seed $solved/$n"
    fi
  } >> "$RESDIR/heldout/summary.txt"
}
export -f run_one

find "$RESDIR" -maxdepth 1 -name '*.policy' -print0 | xargs -0 -P "$PAR" -I{} bash -c 'run_one "$@"' _ {}
echo "summary in $RESDIR/heldout/summary.txt"
