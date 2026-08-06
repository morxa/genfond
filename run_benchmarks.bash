#!/bin/bash

DOMAINS="${DOMAINS:-$(find -L domains/selected -mindepth 1 -maxdepth 1 -type d)}"
POLICY_TYPE="${POLICY_TYPE:-datalog}"
# Set EXCLUDE if some nodes should be excluded from the slurm job, e.g., `EXCLUDE="cn-[409-415]"`
EXCLUDE="${EXCLUDE:+--exclude=$EXCLUDE}"
CONFIG="${CONFIG:+--config $CONFIG}"
VERBOSE="${VERBOSE:+-v}"
# Set PARTITION to override the one in genfond.bash, e.g. `PARTITION=rleap_cpu`
PARTITION="${PARTITION:+--partition=$PARTITION}"
# Set TAG to label a set of runs, so several experiments can be told apart in squeue and in
# the results directory name, e.g. `TAG=frontier-off`
TAG="${TAG:+-$TAG}"
# Set STATE_GRAPHS=1 to write a state space visualisation per solver round. Each job gets its
# own directory, because the file names are only unique within a single run.
STATE_GRAPHS="${STATE_GRAPHS:-}"

STAMP="$(date -Iseconds)"
RESDIR="results-$STAMP$TAG"
mkdir -p "$RESDIR/out"

for domain in $DOMAINS; do
  domainname=$(basename $domain)
  domainfile="$domain/domain.pddl"
  problemfiles=$(find -L $domain ! -name domain.pddl -name '*.pddl')
  for ptype in $POLICY_TYPE; do
    GRAPHS="${STATE_GRAPHS:+--state-graph-dir $RESDIR/graphs/$domainname-$ptype}"
    sbatch $EXCLUDE $PARTITION -J $domainname-$ptype$TAG -o $RESDIR/out/%x-%j.out genfond.bash python -m genfond $VERBOSE --name $domainname -n 32 --max-memory 120000 --type $ptype $CONFIG $GRAPHS --dump-failed-policies --dump-config $RESDIR/$domainname-$ptype.yaml -o $RESDIR/$domainname-$ptype.policy --stats $RESDIR/stats.csv $domainfile $problemfiles
  done
done
