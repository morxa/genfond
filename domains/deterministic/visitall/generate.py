import argparse
import itertools
import os.path
import random

import pddl
import tqdm
from pddl.core import Problem
from pddl.formatter import problem_to_string
from pddl.logic import Constant, Predicate, constants, variables
from pddl.logic.effects import AndEffect


# Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def generate_problem(max_x, max_y, name):
    domain = pddl.parse_domain(os.path.join(os.path.dirname(__file__), "domain.pddl"))
    locations = constants(
        " ".join([f"loc_x{x}_y{y}" for x, y in itertools.product(range(0, max_x), range(0, max_y))]), "place"
    )
    p, p1, p2 = variables("p p1 p2", ["place"])
    up = Predicate("up")
    at_robot = Predicate("at-robot", p)
    connected = Predicate("connected", p1, p2)
    visited = Predicate("visited", p)
    init_location = random.choice(locations)
    init = [at_robot(init_location), visited(init_location)]
    for x in range(0, max_x):
        for y in range(0, max_y - 1):
            from_l = Constant(f"loc_x{x}_y{y}", "place")
            to_l = Constant(f"loc_x{x}_y{y+1}", "place")
            init.append(connected(from_l, to_l))
            init.append(connected(to_l, from_l))
    for y in range(0, max_y):
        for x in range(0, max_x - 1):
            from_l = Constant(f"loc_x{x}_y{y}", "place")
            to_l = Constant(f"loc_x{x+1}_y{y}", "place")
            init.append(connected(from_l, to_l))
            init.append(connected(to_l, from_l))
    problem = Problem(
        name, domain=domain, objects=locations, init=init, goal=AndEffect(*[visited(l) for l in locations])
    )
    return problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-name", type=str, default="visitall")
    parser.add_argument("max_size", type=int)
    parser.add_argument("-n", help="Number of problems per configuration", type=int, default=1)
    args = parser.parse_args()
    for max_x, max_y, i in tqdm.tqdm(
        itertools.product(range(1, args.max_size + 1), range(1, args.max_size + 1), range(args.n))
    ):
        name = f"{args.base_name}-{max_x}-{max_y}-{i}"
        problem = generate_problem(max_x, max_y, name)
        with open(f"p-{name}.pddl", "w") as f:
            f.write(problem_to_string(problem))


if __name__ == "__main__":
    main()
