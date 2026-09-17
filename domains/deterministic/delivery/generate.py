import argparse
import itertools
import os.path
import random

import pddl
import tqdm
from pddl.core import Problem
from pddl.formatter import problem_to_string
from pddl.logic import Constant, Predicate, Variable, variables
from pddl.logic.effects import AndEffect


def generate_problem(name, trucks, max_x, max_y, packages):
    domain = pddl.parse_domain(os.path.join(os.path.dirname(__file__), "domain.pddl"))
    cells = [Constant(f"c_{x}_{y}", "cell") for x, y in itertools.product(range(0, max_x), range(0, max_y))]
    packages = [Constant(f"p{i}", "package") for i in range(0, packages)]
    trucks = [Constant(f"t{i}", "truck") for i in range(0, trucks)]
    c, c1, c2 = variables("c c1 c2", ["cell"])
    l = Variable("l", ["locatable"])
    p = Variable("p", ["package"])
    t = Variable("t", ["truck"])
    at_t = Predicate("at", t, c)
    at_p = Predicate("at", p, c)
    adjacent = Predicate("adjacent", c1, c2)
    empty = Predicate("empty", t)
    init = []
    for truck in trucks:
        init.append(empty(truck))
        init.append(at_t(truck, random.choice(cells)))
    for package in packages:
        init.append(at_p(package, random.choice(cells)))
    for x in range(0, max_x):
        for y in range(0, max_y - 1):
            from_l = Constant(f"c_{x}_{y}", "cell")
            to_l = Constant(f"c_{x}_{y+1}", "cell")
            init.append(adjacent(from_l, to_l))
            init.append(adjacent(to_l, from_l))
    for y in range(0, max_y):
        for x in range(0, max_x - 1):
            from_l = Constant(f"c_{x}_{y}", "cell")
            to_l = Constant(f"c_{x+1}_{y}", "cell")
            init.append(adjacent(from_l, to_l))
            init.append(adjacent(to_l, from_l))
    goal_cell = random.choice(cells)
    problem = Problem(
        name,
        domain=domain,
        objects=cells + packages + trucks,
        init=init,
        goal=AndEffect(*[at_p(p, goal_cell) for p in packages]),
    )
    return problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-name", type=str, default="delivery")
    parser.add_argument("-s", "--max-grid-size", type=int, default=5)
    parser.add_argument("-t", "--max-trucks", type=int, default=1)
    parser.add_argument("-p", "--max-packages", type=int, default=3)
    parser.add_argument("-n", help="Number of problems per configuration", type=int, default=3)
    args = parser.parse_args()
    for trucks, max_x, max_y, packages, i in tqdm.tqdm(
        itertools.product(
            range(1, args.max_trucks + 1),
            range(1, args.max_grid_size + 1),
            range(1, args.max_grid_size + 1),
            range(1, args.max_packages + 1),
            range(args.n),
        )
    ):
        name = f"{args.base_name}-{max_x}x{max_y}-{trucks}-{packages}-{i}"
        problem = generate_problem(name, trucks, max_x, max_y, packages)
        with open(f"p-{name}.pddl", "w") as f:
            f.write(problem_to_string(problem))


if __name__ == "__main__":
    main()
