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
    for j in {1..5}
    do
        PROBLEM_NAME=problem-$i-$j
        echo $PROBLEM_NAME

        # Generate the problem file
        $SOURCE/barman-generator.py $i 3 $((3 + $RANDOM % ($i - 2))) > $TARGET/$PROBLEM_NAME.pddl

        reformat_problem
    done
done

