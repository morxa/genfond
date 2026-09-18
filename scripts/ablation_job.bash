#!/bin/bash
#
#SBATCH --partition=rleap_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000
#SBATCH --time=2:00:00
#SBATCH --signal=B:TERM@600
#SBATCH --output=%x-%A_%a.out
#
# Array task for scripts/ablation.sh -- every directive above is overridden by the flags
# ablation.sh passes on the sbatch command line; they only matter if this is submitted by hand.
#
# `--signal=B:TERM@600` (see genfond.bash for the full rationale): SLURM sends SIGTERM to this
# batch shell 600s before --time is up, which must be forwarded to the apptainer child so genfond
# gets a chance to checkpoint (pickle its best policy, write a stats row) instead of being
# SIGKILLed with nothing written.
#
# Usage: scripts/ablation_job.bash <manifest> <resdir>
#   $SLURM_ARRAY_TASK_ID selects the manifest line (1-indexed):
#     <domainname>|<domaindir>|<seed>|<configpath>|<mem>|<arm>
#   WALL_TIME (seconds, inherited from ablation.sh's exported environment; default 6600) is
#   passed to genfond's own --max-wall-time.
set -u
manifest="$1"
resdir="$2"

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$manifest")
if [ -z "$line" ]; then
  echo "No manifest line $SLURM_ARRAY_TASK_ID in $manifest" >&2
  exit 1
fi
IFS='|' read -r domainname domaindir seed config mem arm <<< "$line"

# State is a frozenset of pddl atoms, so the loop path follows the hash seed; two runs of the
# same arm/seed are only comparable with it pinned (see run_benchmarks.bash).
export PYTHONHASHSEED="$seed"

domainfile="$domaindir/domain.pddl"
problemfiles=$(find -L "$domaindir" -maxdepth 1 ! -name domain.pddl -name '*.pddl')
maxmemory=$((mem * 2))
walltime="${WALL_TIME:-6600}"

mkdir -p "$resdir/out"
name="$arm-$domainname-s$seed"
echo "Running: arm=$arm domain=$domainname seed=$seed mem=${mem}MB max-memory=${maxmemory}MB max-wall-time=${walltime}s"
echo "Config: $config"
echo "Problems: $problemfiles"

apptainer run --bind "$PWD" genfond_env.sif \
  python -m genfond -v --name "$domainname" -n 1 --max-memory "$maxmemory" \
    --type datalog-sig --max-wall-time "$walltime" --config "$config" \
    --dump-failed-policies \
    --dump-config "$resdir/$name.yaml" -o "$resdir/$name.policy" \
    --stats "$resdir/stats.csv" \
    "$domainfile" $problemfiles &
child=$!
trap 'echo "Forwarding SIGTERM to apptainer (pid $child)"; kill -TERM "$child" 2>/dev/null' TERM
wait "$child"
