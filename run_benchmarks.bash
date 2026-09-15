#!/bin/bash

DOMAINS="${DOMAINS:-$(find -L domains/selected -mindepth 1 -maxdepth 1 -type d)}"
POLICY_TYPE="${POLICY_TYPE:-datalog}"
# Set EXCLUDE if some nodes should be excluded from the slurm job, e.g., `EXCLUDE="cn-[409-415]"`
EXCLUDE="${EXCLUDE:+--exclude=$EXCLUDE}"
CONFIG="${CONFIG:+--config $CONFIG}"
VERBOSE="${VERBOSE:+-v}"
# Queue on either CPU partition by default, so a run is not stuck behind whichever one happens
# to be busy -- rleap_cpu_modern is a single node and blocks easily. slurm takes a
# comma-separated list and starts the job on the partition that has resources first.
#
# Pin this to one partition (`PARTITION=rleap_cpu`) whenever runs have to be comparable to each
# other: the two partitions have different CPUs, so wall times measured across them do not mean
# the same thing, and with a --time limit a slower node also solves fewer problems.
PARTITION="--partition=${PARTITION:-rleap_cpu,rleap_cpu_modern}"
# Set TAG to label a set of runs, so several experiments can be told apart in squeue and in
# the results directory name, e.g. `TAG=frontier-off`
TAG="${TAG:+-$TAG}"
# clingo's parallel mode is nondeterministic: it may return any optimal model, and with
# frontier expansion the choice of model decides which states get expanded next, so two runs of
# the identical configuration diverge into different training sets. Set THREADS=1 whenever two
# runs have to be compared against each other; measured on blocks3ops, -n 8 gave 6 rounds on
# one repetition and 4 on the next, while -n 1 reproduced exactly.
THREADS="${THREADS:-32}"
# sbatch exports the environment; State is a frozenset of pddl atoms, so the loop path follows the
# hash seed and two runs are only comparable with it pinned.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
# Seconds passed to --max-wall-time; default unset, which leaves genfond's own run budget
# unbounded (the SBATCH --time limit is then the only cutoff, and a kill at it writes neither a
# policy nor a stats row -- see genfond/config/default.yaml's max_wall_time). Set this a bit
# below the sbatch --time in genfond.bash so a job finishes gracefully instead of being killed.
WALL_TIME="${WALL_TIME:-}"
MAX_WALL_TIME_ARGS="${WALL_TIME:+--max-wall-time $WALL_TIME}"

STAMP="$(date -Iseconds)"
RESDIR="results-$STAMP$TAG"
mkdir -p "$RESDIR/out"

for domain in $DOMAINS; do
  domainname=$(basename $domain)
  domainfile="$domain/domain.pddl"
  problemfiles=$(find -L $domain ! -name domain.pddl -name '*.pddl')
  for ptype in $POLICY_TYPE; do
    sbatch $EXCLUDE $PARTITION -J $domainname-$ptype$TAG -o $RESDIR/out/%x-%j.out genfond.bash python -m genfond $VERBOSE --name $domainname -n $THREADS --max-memory 120000 --type $ptype $MAX_WALL_TIME_ARGS $CONFIG --dump-failed-policies --dump-config $RESDIR/$domainname-$ptype.yaml -o $RESDIR/$domainname-$ptype.policy --stats $RESDIR/stats.csv $domainfile $problemfiles
  done
done
