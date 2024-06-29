#!/bin/bash
SOURCE=/work/rleap1/samuel.stante/pddl-generators/gripper
TARGET=/work/rleap1/samuel.stante/genfond/domains/deterministic/gripper

echo "Gripper"

source /work/rleap1/samuel.stante/genfond/generators/_.bash

for i in {1..6} # $i is the number of balls
do
    for j in {1..5}
    do
        PROBLEM_NAME=problem-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        $SOURCE/gripper -n $i > $TARGET/$PROBLEM_NAME.pddl
        
        reformat_problem
    done
done

