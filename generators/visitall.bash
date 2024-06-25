#!/bin/bash
SOURCE=/work/rleap1/samuel.stante/pddl-generators/visitall
TARGET=/work/rleap1/samuel.stante/genfond/domains/deterministic/aaai21/visitall

source /work/rleap1/samuel.stante/genfond/generators/_.bash

echo "Visitall"

# Full
for i in {4..7} # $i is the size of the square grid
do
    for j in {1..4}
    do
        PROBLEM_NAME=problem-full-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        $SOURCE/grid -n $i -r 1 -u $((5 + $RANDOM % 10)) | sed '/^[0-9]*$/d' | sed -r 's/loc-x([0-9]*)-y([0-9]*)/loc_x\1_y\2/g' > $TARGET/$PROBLEM_NAME.pddl
        
        reformat_problem
    done
done

# Half
for i in {4..7} # $i is the size of the square grid
do
    for j in {1..4}
    do
        PROBLEM_NAME=problem-half-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        $SOURCE/grid -n $i -r 0.5 -u $((5 + $RANDOM % 10)) | sed '/^[0-9]*$/d' | sed -r 's/loc-x([0-9]*)-y([0-9]*)/loc_x\1_y\2/g' > $TARGET/$PROBLEM_NAME.pddl
        
        reformat_problem
    done
done