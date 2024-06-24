#!/bin/bash

DOMAINS="${DOMAINS:-$(find domains/non-deterministic -mindepth 1 -maxdepth 1 -type d)}"
POLICY_TYPE="${POLICY_TYPE:-datalog}"
# Set EXCLUDE if some nodes should be excluded from the slurm job, e.g., `EXCLUDE="cn-[409-415]"`
EXCLUDE="${EXCLUDE:+--exclude=$EXCLUDE}"


STAMP="$(date -Iseconds)"
RESDIR="results-$STAMP"
mkdir -p "$RESDIR"

for domain in $DOMAINS; do
  domainname=$(basename $domain)
  for ptype in $POLICY_TYPE; do
    sbatch $EXCLUDE -J $domainname-$ptype genfond.bash python genfond.py --name $domainname -n 32 --max-memory 240000 --type $ptype --policy-steps 10000 --policy-iterations 10 --max-complexity 15 --dump-failed-policies -o $RESDIR/$domainname-$constraint.policy --stats $RESDIR/stats.csv $domain/{domain.pddl,p*}
  done
done
