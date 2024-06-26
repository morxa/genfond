#!/bin/bash
SOURCE=/work/rleap1/samuel.stante/pddl-generators/blocksworld
TARGET=/work/rleap1/samuel.stante/genfond/domains/deterministic/blocksworld-4ops

source /work/rleap1/samuel.stante/genfond/generators/_.bash

echo "Blocksworld 4ops"

for i in {2..20} # $i is the number of blocks
do
    for j in {1..5}
    do
        PROBLEM_NAME=problem-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        $SOURCE/blocksworld 4 $i > $TARGET/$PROBLEM_NAME.pddl

        reformat_problem
    done
done
