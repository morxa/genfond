#!/bin/bash

DOMAINS="${DOMAINS:-$(find domains/non-deterministic -mindepth 1 -maxdepth 1 -type d)}"
POLICY_TYPE="${POLICY_TYPE:-datalog}"

STAMP="$(date -Iseconds)"
RESDIR="results-$STAMP"
mkdir -p "$RESDIR"

for domain in $DOMAINS; do
  for ptype in $POLICY_TYPE; do
    sbatch --exclude='cn-[409-415]' -J $domain-$ptype genfond.bash python genfond.py --name $domain -n 32 --max-memory 240000 --type $ptype --policy-steps 10000 --policy-iterations 10 --max-complexity 15 --dump-failed-policies -o $RESDIR/$domain-$constraint.policy --stats $RESDIR/stats.csv $domain/{domain.pddl,p*}
  done
done
