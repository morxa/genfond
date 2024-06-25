#!/bin/bash
SOURCE=/work/rleap1/samuel.stante/pddl-generators/blocksworld
TARGET=/work/rleap1/samuel.stante/genfond/domains/deterministic/aaai21/blocksworld-3ops

source /work/rleap1/samuel.stante/genfond/generators/_.bash

echo "Blocksworld 3ops"

for i in {3..20} # $i is the number of blocks
do
    for j in {1..5}
    do
        PROBLEM_NAME=problem-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        $SOURCE/blocksworld 3 $i | sed "s/(:init/(:init (arm-empty)/" > $TARGET/$PROBLEM_NAME.pddl

        reformat_problem
    done
done
