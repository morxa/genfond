import argparse
import itertools
import os.path
import random

import pddl
from pddl.core import Problem
from pddl.formatter import problem_to_string
from pddl.logic import Predicate, constants, variables
from pddl.logic.effects import AndEffect
from tqdm import trange


# Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def generate_problem(name, num_blocks):
    domain = pddl.parse_domain(os.path.join(os.path.dirname(__file__), "domain.pddl"))
    blocks = constants(" ".join([f"b{i}" for i in range(num_blocks)]))
    x, y = variables("x y")
    clear = Predicate("clear", x)
    ontable = Predicate("ontable", x)
    on = Predicate("on", x, y)
    init = []
    unpositioned_blocks = blocks.copy()
    while len(unpositioned_blocks) > 0:
        tower = random.sample(unpositioned_blocks, random.randint(1, len(unpositioned_blocks)))
        unpositioned_blocks = [b for b in unpositioned_blocks if b not in tower]
        init.append(ontable(tower[-1]))
        init.append(clear(tower[0]))
        for b1, b2 in pairwise(tower):
            init.append(on(b1, b2))
    goal_blocks = random.sample(blocks, random.randint(2, len(blocks)))
    goal = []
    for b1, b2 in pairwise(goal_blocks):
        goal.append(on(b1, b2))
    problem = Problem(
        name,
        domain=domain,
        objects=blocks,
        init=init,
        goal=AndEffect(clear(goal_blocks[0]), *goal, ontable(goal_blocks[-1])),
    )
    return problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-name", type=str, default="blocks")
    parser.add_argument("--min-num-blocks", type=int, default=2)
    parser.add_argument("--max-num-blocks", type=int, default=20)
    parser.add_argument(
        "-r", "--repetitions", help="number of problems for each number of blocks", type=int, default=5
    )
    parser.add_argument("-s", "--seed", help="random seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    for num_blocks in trange(args.min_num_blocks, args.max_num_blocks + 1):
        for i in range(args.repetitions):
            index = f"{num_blocks:03}-{i+1}"
            problem = generate_problem(f"{args.base_name}-{index}", num_blocks)
            with open(f"p{index}.pddl", "w") as f:
                f.write(problem_to_string(problem))


if __name__ == "__main__":
    main()
