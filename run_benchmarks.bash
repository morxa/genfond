#!/bin/bash

DOMAINS="${DOMAINS:-$(ls domains/non-deterministic/)}"
CONSTRAINTS="${CONSTRAINTS:-state trans}"

STAMP="$(date -Iseconds)"
RESDIR="results-$STAMP"
mkdir -p "$RESDIR"

for domain in $DOMAINS; do
  for constraint in $CONSTRAINTS; do
    sbatch --exclude='cn-[409-415]' -J $domain-$constraint genfond.bash python genfond.py --name $domain -n 32 --max-memory 240000 --constraints $constraint --policy-steps 10000 --policy-iterations 10 --max-complexity 15 --dump-failed-policies -o $RESDIR/$domain-$constraint.policy --stats $RESDIR/stats.csv domains/non-deterministic/$domain/{domain.pddl,p*}
  done
done
