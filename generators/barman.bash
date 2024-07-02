#!/bin/bash
SOURCE=/work/rleap1/samuel.stante/pddl-generators/barman
TARGET=/work/rleap1/samuel.stante/genfond/domains/deterministic/barman

echo "Barman"

source /work/rleap1/samuel.stante/genfond/generators/_.bash

# Remove total-cost function from domain
sed -i "s/ (increase (total-cost) [0-9]\+)//g" $TARGET/domain.pddl
sed -i "/(:functions (total-cost) - number)/d" $TARGET/domain.pddl
reformat_domain

for i in {3..7} # $i is the number of cocktails
do
    for j in $(seq 1 $(($i + 1))) # $j is the number of shots
    do
        PROBLEM_NAME=problem-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        $SOURCE/barman-generator.py $i 3 $j > $TARGET/$PROBLEM_NAME.pddl

        # Remove invalid goals
        for k in $(seq $(($j + 1)) $(($i + 1)))
        do
            sed -i "s/(contains shot$k cocktail[0-9]\+)//g" $TARGET/$PROBLEM_NAME.pddl
        done

        reformat_problem
    done
done

