#!/bin/bash
SOURCE="${SOURCE:-/work/rleap1/samuel.stante/pddl-generators/blocksworld}"
TARGET="${TARGET:-/work/rleap1/samuel.stante/genfond/domains/deterministic/blocksworld-3ops}"

echo "Blocksworld 3ops"

source /work/rleap1/samuel.stante/genfond/generators/_.bash

for i in {3..20} # $i is the number of blocks
do
    for j in {1..5}
    do
        PROBLEM_NAME=problem-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        # Generator does not generate the arm-empty predicate in the initial state so it is added manually
        $SOURCE/blocksworld 3 $i | sed "s/(:init/(:init (arm-empty)/" > $TARGET/$PROBLEM_NAME.pddl

        reformat_problem
    done
done
