#!/bin/bash
SOURCE=/work/rleap1/samuel.stante/pddl-generators/hanoi
TARGET=/work/rleap1/samuel.stante/genfond/domains/deterministic/aaai21/hanoi

source /work/rleap1/samuel.stante/genfond/generators/_.bash

echo "Hanoi"

for i in {3..8} # $i is the number of balls
do
    for j in {1..5}
    do
        PROBLEM_NAME=problem-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        $SOURCE/hanoi -n $i > $TARGET/$PROBLEM_NAME.pddl
        
        reformat_problem
    done
done

