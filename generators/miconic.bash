#!/bin/bash
SOURCE="${SOURCE:-/work/rleap1/samuel.stante/pddl-generators/miconic}"
TARGET="${TARGET:-/work/rleap1/samuel.stante/genfond/domains/deterministic/miconic}"

echo "Miconic"

source /work/rleap1/samuel.stante/genfond/generators/_.bash

# Add missing :typing requirement to domain
sed -i 's/(:requirements :strips)/(:requirements :strips :typing)/' $TARGET/domain.pddl
reformat_domain

for i in {1..5} # $i is the number of passengers
do
    for j in {1..5}
    do
        PROBLEM_NAME=problem-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        $SOURCE/miconic -p $i -f $(($i * 2)) > $TARGET/$PROBLEM_NAME.pddl
        
        reformat_problem
    done
done

