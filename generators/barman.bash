#!/bin/bash
SOURCE=/work/rleap1/samuel.stante/pddl-generators/barman
TARGET=/work/rleap1/samuel.stante/genfond/domains/deterministic/barman

source /work/rleap1/samuel.stante/genfond/generators/_.bash

echo "Barman"
echo "# Manually remove total-cost from the domain file"

for i in {3..7} # $i is the number of cocktails
do
    for j in {1..5}
    do
        PROBLEM_NAME=problem-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        $SOURCE/barman-generator.py $i 3 $((1 + $RANDOM % $i)) > $TARGET/$PROBLEM_NAME.pddl

        reformat_problem
    done
done

