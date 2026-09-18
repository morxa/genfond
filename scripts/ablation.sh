#!/bin/bash
#
# Generate configs and submit a SLURM job array per ablation arm.
#
#   TAG=abl scripts/ablation.sh
#   DRY=1 TAG=abl scripts/ablation.sh   # print the sbatch commands and generate the yaml
#                                        # files, but do not actually submit
#
# Each arm removes exactly one hypothesis (H27, H29, H30, H32, H33) from BASE, so the six default
# arms (full + five "noHxx" arms) let a single run measure each hypothesis's individual
# contribution on top of the others. See docs/experiments-log.md for what H27/H29/H30/H32/H33 do;
# the flags below are their config keys in genfond/config/default.yaml.
#
# Knobs (environment variables):
#   BASE       base config yaml, layered under --config (default claude-experiments/st.yaml)
#   SEEDS      space-separated seeds (default "0 1 2")
#   ARMS       space-separated arm names (default "full noH27 noH29 noH30 noH32 noH33")
#   DOMAINS    space-separated domain directories (default: the ten suites below)
#   PARTITION  sbatch partition (default rleap_cpu)
#   TIME       sbatch --time (default 2:00:00)
#   WALL_TIME  seconds passed to genfond's --max-wall-time (default 6600, i.e. 10 min under TIME
#              so a graceful SIGTERM stop -- see scripts/ablation_job.bash -- has time to finish)
#   CONC       array concurrency cap per arm, i.e. the %N in --array=1-N%CONC (default 10)
#   TAG        label; configs land in claude-experiments/<TAG>/, results in
#              results-<timestamp>-<TAG>/ (default abl)
#   DRY        1 = print the sbatch command lines instead of submitting (default 0)
set -u
BASE="${BASE:-claude-experiments/st.yaml}"
SEEDS="${SEEDS:-0 1 2}"
ARMS="${ARMS:-full noH27 noH29 noH30 noH32 noH33}"
DOMAINS="${DOMAINS:-domains/suites/blocks3ops domains/suites/blocks4ops-flat domains/deterministic-new/blocks domains/deterministic-new/delivery domains/deterministic/miconic domains/deterministic-new/logistics domains/deterministic-new/logistics_dp domains/deterministic-new/reward domains/deterministic-new/spanner domains/deterministic/visitall}"
PARTITION="${PARTITION:-rleap_cpu}"
TIME="${TIME:-2:00:00}"
WALL_TIME="${WALL_TIME:-6600}"
CONC="${CONC:-10}"
TAG="${TAG:-abl}"
DRY="${DRY:-0}"
CPUS=4

cd "$(dirname "$0")/.."

if [ ! -f "$BASE" ]; then
  echo "BASE config not found: $BASE" >&2
  exit 1
fi

# sbatch inherits the submitting shell's exported environment by default, which is how
# ablation_job.bash (started with only <manifest> <resdir> on its command line) sees WALL_TIME.
export WALL_TIME

# domains/suites/blocks3ops is a plain symlink to the domain directory elsewhere in the tree (see
# git status); domains/suites/blocks4ops-flat has to be a directory of per-file symlinks instead,
# because blocks4ops's own directory also holds an `old/` subdirectory of 18 extra instances that
# must NOT be part of this suite -- see docs/experiments-log.md's "Benchmark consolidation" entry.
if [ ! -e domains/suites/blocks3ops ]; then
  ln -s ../deterministic/blocks3ops domains/suites/blocks3ops
fi
if [ ! -e domains/suites/blocks4ops-flat ]; then
  mkdir -p domains/suites/blocks4ops-flat
  for f in domains/deterministic/blocks4ops/*.pddl; do
    ln -s "../../deterministic/blocks4ops/$(basename "$f")" "domains/suites/blocks4ops-flat/$(basename "$f")"
  done
fi

CFGDIR="claude-experiments/$TAG"
mkdir -p "$CFGDIR"

# One config per (arm, seed): domain-independent, so generated once and reused across every
# domain in DOMAINS. `planners.siw.restarts: 2` on every arm/seed so the SIW seed actually
# changes which plans are drawn -- with restarts: 1, restart 1 is SIW's identity permutation and
# the seed is inert (see the H32 finding in docs/experiments-log.md).
gen_config() {
  local arm="$1" seed="$2" out="$3"
  poetry run python3 - "$BASE" "$out" "$seed" "$arm" <<'PYEOF'
import sys

import yaml

base_path, out_path, seed_str, arm = sys.argv[1:5]
with open(base_path) as f:
    cfg = yaml.safe_load(f) or {}

cfg["seed"] = int(seed_str)
siw = cfg.setdefault("planners", {}).setdefault("siw", {})
siw["seed"] = int(seed_str)
siw["restarts"] = 2

# One flag switched off per "noHxx" arm; `full` (or any arm not listed here) leaves BASE as is.
arm_overrides = {
    "noH27": {"policy_conformant_plans": False},
    "noH29": {"anchor_policy_labels": False},
    "noH30": {"policy_prefix_plans": False},
    "noH32": {"resample_on_stall": False},
    "noH33": {"continue_after_max_complexity": False},
}
cfg.update(arm_overrides.get(arm, {}))

with open(out_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYEOF
}

STAMP="$(date -Iseconds)"
RESDIR="results-$STAMP-$TAG"
mkdir -p "$RESDIR/out"
echo "results in $RESDIR"

for arm in $ARMS; do
  manifest="$RESDIR/manifest-$arm.txt"
  : > "$manifest"
  maxmem=0
  for domain in $DOMAINS; do
    domainname=$(basename "$domain")
    if [ "$domainname" = "blocks3ops" ]; then mem=24000; else mem=16000; fi
    [ "$mem" -gt "$maxmem" ] && maxmem=$mem
    for seed in $SEEDS; do
      cfgpath="$CFGDIR/$arm-s$seed.yaml"
      if [ ! -f "$cfgpath" ]; then
        gen_config "$arm" "$seed" "$cfgpath"
      fi
      echo "$domainname|$domain|$seed|$cfgpath|$mem|$arm" >> "$manifest"
    done
  done
  n=$(wc -l < "$manifest")
  cmd=(sbatch "--array=1-$n%$CONC" "--partition=$PARTITION" "--cpus-per-task=$CPUS" \
    "--mem=$maxmem" "--time=$TIME" -J "$TAG-$arm" -o "$RESDIR/out/%x-%A_%a.out" \
    scripts/ablation_job.bash "$manifest" "$RESDIR")
  echo "${cmd[@]}"
  if [ "$DRY" != "1" ]; then
    "${cmd[@]}"
  fi
done
