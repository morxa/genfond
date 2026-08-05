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

STAMP="$(date -Iseconds)"
RESDIR="results-$STAMP$TAG"
mkdir -p "$RESDIR/out"

for domain in $DOMAINS; do
  domainname=$(basename $domain)
  domainfile="$domain/domain.pddl"
  problemfiles=$(find -L $domain ! -name domain.pddl -name '*.pddl')
  for ptype in $POLICY_TYPE; do
    sbatch $EXCLUDE $PARTITION -J $domainname-$ptype$TAG -o $RESDIR/out/%x-%j.out genfond.bash python -m genfond $VERBOSE --name $domainname -n 32 --max-memory 120000 --type $ptype $CONFIG --dump-failed-policies --dump-config $RESDIR/$domainname-$ptype.yaml -o $RESDIR/$domainname-$ptype.policy --stats $RESDIR/stats.csv $domainfile $problemfiles
  done
done
