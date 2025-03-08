import argparse
import itertools
import os.path
import random

import pddl
import pygraphviz
from pddl.core import Problem
from pddl.formatter import problem_to_string
from pddl.logic import Predicate, constants, variables
from tqdm import trange


# Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def draw(name, locations, facts, goal):
    g = pygraphviz.AGraph(strict=False, directed=False)
    g.node_attr["shape"] = "circle"
    for location in locations:
        g.add_node(location)
    for fact in facts:
        if fact.name == "road":
            if fact.terms[0] < fact.terms[1]:
                g.add_edge(fact.terms[0], fact.terms[1])
        elif fact.name == "spare-in":
            n = g.get_node(fact.terms[0])
            n.attr["color"] = "green"
        elif fact.name == "vehicle-at":
            n = g.get_node(fact.terms[0])
            n.attr["shape"] = "cds"
    gnode = g.get_node(goal)
    gnode.attr["peripheries"] = 2
    g.layout(prog="neato")
    g.draw(f"{name}.png")


def generate_problem(name, num_locations, draw_problem):
    domain = pddl.parse_domain(os.path.join(os.path.dirname(__file__), "domain.pddl"))
    locations = constants(" ".join([f"l{i}" for i in range(num_locations)]), "location")
    x, y = variables("x y", ["location"])
    spare_in = Predicate("spare-in", x)
    vehicle_at = Predicate("vehicle-at", x)
    road = Predicate("road", x, y)
    # randomly sample a safe road with spare tires
    safe_conn = random.sample(locations, random.randint(2, len(locations)))
    init_location = safe_conn[0]
    goal_location = safe_conn[-1]
    init = [vehicle_at(init_location)]
    init += [road(*pair) for pair in pairwise(safe_conn)]
    init += [road(*pair) for pair in pairwise(reversed(safe_conn))]
    init += [spare_in(loc) for loc in safe_conn[1:-1]]
    conns = [safe_conn]
    remaining_locations = set(locations) - set(safe_conn)
    while len(remaining_locations) > 0:  # and random.random() > 1 / math.log(len(remaining_locations)):
        # randomly sample another conn
        start_conn = random.choice(conns)
        start_location = random.choice(start_conn)
        end_conn = random.choice(conns)
        end_location = random.choice(end_conn)
        if start_location == end_location:
            continue
        conn = [
            start_location,
            *random.sample(sorted(remaining_locations), random.randint(1, len(remaining_locations))),
            end_location,
        ]
        init += [road(*pair) for pair in pairwise(conn)]
        init += [road(*pair) for pair in pairwise(reversed(conn))]
        conns.append(conn)
        remaining_locations -= set(conn)
    # randomly add some spares
    spare_prob = random.random()
    for location in locations:
        if location != init_location and location != goal_location and random.random() > spare_prob:
            init.append(spare_in(location))
    if draw_problem:
        draw(name, locations, init, goal_location)
    problem = Problem(name, domain=domain, objects=locations, init=init, goal=vehicle_at(goal_location))
    return problem


def generate_problems(min_num_locations, max_num_locations, repetitions, draw):
    for num_locations in trange(min_num_locations, max_num_locations + 1):
        for i in range(repetitions):
            index = f"{num_locations:03}-{i+1:02}"
            problem = generate_problem(f"tireworld-{index}", num_locations, draw)
            with open(f"p{index}.pddl", "w") as f:
                f.write(problem_to_string(problem))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-num-location", type=int, default=3)
    parser.add_argument("--max-num-locations", type=int, default=100)
    parser.add_argument(
        "-r", "--repetitions", help="number of problems for each number of locations", type=int, default=10
    )
    parser.add_argument("-s", "--seed", help="random seed", type=int, default=0)
    parser.add_argument("--draw", help="draw the generated problems", action="store_true")
    args = parser.parse_args()
    random.seed(args.seed)
    generate_problems(args.min_num_location, args.max_num_locations, args.repetitions, args.draw)


if __name__ == "__main__":
    main()
