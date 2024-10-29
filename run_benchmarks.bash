#!/bin/bash

DOMAINS="${DOMAINS:-$(find domains/deterministic domains/non-deterministic -mindepth 1 -maxdepth 1 -type d)}"
POLICY_TYPE="${POLICY_TYPE:-datalog}"
# Set EXCLUDE if some nodes should be excluded from the slurm job, e.g., `EXCLUDE="cn-[409-415]"`
EXCLUDE="${EXCLUDE:+--exclude=$EXCLUDE}"
CONFIG="${CONFIG:+--config $CONFIG}"
VERBOSE="${VERBOSE:+-v}"

STAMP="$(date -Iseconds)"
RESDIR="results-$STAMP"
mkdir -p "$RESDIR/out"

for domain in $DOMAINS; do
  domainname=$(basename $domain)
  domainfile="$domain/domain.pddl"
  problemfiles=$(find $domain ! -name domain.pddl -name '*.pddl')
  for ptype in $POLICY_TYPE; do
    sbatch $EXCLUDE -J $domainname-$ptype -o $RESDIR/out/%x-%j.out genfond.bash python -m genfond $VERBOSE --name $domainname -n 32 --max-memory 240000 --type $ptype $CONFIG --policy-steps 10000 --policy-iterations 1 --max-complexity 15 --dump-failed-policies --dump-config $RESDIR/$domainname-$ptype.yaml -o $RESDIR/$domainname-$ptype.policy --stats $RESDIR/stats.csv $domainfile $problemfiles
  done
done
