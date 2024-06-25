function reformat_domain {
    python -c """
from pddl import parse_domain
from pddl.formatter import domain_to_string

domain = parse_domain('$TARGET/domain.pddl')
with open('$TARGET/domain.pddl', 'w+') as f:
    f.write(domain_to_string(domain))
"""
}

function reformat_problem {
    python -c """
from pddl import parse_problem
from pddl.formatter import problem_to_string

problem = parse_problem('$TARGET/$PROBLEM_NAME.pddl')
problem._name = '$PROBLEM_NAME'
with open('$TARGET/$PROBLEM_NAME.pddl', 'w+') as f:
    f.write(problem_to_string(problem))
"""
}

function check_solvable_and_reformat_problem {
    SOLVABLE=$(python3 /work/rleap1/samuel.stante/genfond/is_solvable.py $TARGET/domain.pddl $TARGET/$PROBLEM_NAME.pddl)
    if [[ $SOLVABLE = "solvable" ]];
    then
        reformat_problem
        return 0
    else
        rm $TARGET/$PROBLEM_NAME.pddl
        return 1
    fi
}

mkdir -p $TARGET

# Copy the domain file
if [ ! -f $TARGET/domain.pddl ]; then
    cp $SOURCE/domain.pddl $TARGET
fi


# Reformat the domain file
reformat_domain $TARGET/domain.pddl