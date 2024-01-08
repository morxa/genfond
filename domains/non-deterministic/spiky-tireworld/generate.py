import pddl
import itertools
from pddl.logic import Predicate, constants, variables
from pddl.logic.terms import Variable
from pddl.logic.effects import AndEffect
from pddl.core import Problem
from pddl.formatter import problem_to_string
import argparse
import os.path
from tqdm import trange
import random
import pygraphviz
import math

from draw import draw


# Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def add_conn(road, conn_locations, add_spares=True):
    init = [road(*pair) for pair in pairwise(conn_locations)]
    init += [road(*pair) for pair in pairwise(reversed(conn_locations))]
    if add_spares:
        init += [spare_in(l) for l in conn_locations[1:-1]]
    return init


def generate_problem(name, num_locations, draw_problem):
    domain = pddl.parse_domain(os.path.join(os.path.dirname(__file__), 'domain.pddl'))
    locations = constants(' '.join([f'l{i}' for i in range(num_locations)]), 'location')
    x, y = variables('x y', ['location'])
    t = Variable('t', ['tire'])
    tire_at = Predicate('tire-at', t, y)
    vehicle_at = Predicate('vehicle-at', x)
    road = Predicate('road', x, y)
    spiky_road = Predicate('spiky-road', x, y)
    init_location = locations[0]
    goal_location = locations[-1]
    unsafe_length = random.randint(3, int(num_locations / 2))
    unsafe_conn = locations[:unsafe_length] + [goal_location]
    first_spiky_unsafe = random.randint(1, unsafe_length - 2)
    unsafe_first_seg = unsafe_conn[0:first_spiky_unsafe + 1]
    unsafe_spiky_seg = unsafe_conn[first_spiky_unsafe:first_spiky_unsafe + 3]
    unsafe_second_seg = unsafe_conn[first_spiky_unsafe + 2:]
    safe_conn = [init_location] + locations[unsafe_length:]
    first_spiky_safe = random.randint(0, len(safe_conn) - 2)
    safe_first_seg = safe_conn[0:first_spiky_safe + 1]
    safe_spiky_seg = safe_conn[first_spiky_safe:first_spiky_safe + 2]
    safe_second_seg = safe_conn[first_spiky_safe + 1:]
    init = [vehicle_at(init_location), Predicate('not-flattire'), Predicate('not-hasspare')]
    for seg in [unsafe_first_seg, unsafe_second_seg, safe_first_seg, safe_second_seg]:
        init += [road(*pair) for pair in pairwise(seg)]
        init += [road(*pair) for pair in pairwise(reversed(seg))]
    for seg in [unsafe_spiky_seg, safe_spiky_seg]:
        init += [spiky_road(*pair) for pair in pairwise(seg)]
        init += [spiky_road(*pair) for pair in pairwise(reversed(seg))]
    num_spares = random.randint(1, len(unsafe_conn) - 1)
    spares = constants(' '.join([f't{i}' for i in range(num_spares)]), 'tire')
    init += [tire_at(s, unsafe_conn[1]) for s in spares]
    objects = locations + spares
    if draw_problem:
        draw(name, objects, init, vehicle_at(goal_location))
    problem = Problem(name, domain=domain, objects=objects, init=init, goal=vehicle_at(goal_location))
    return problem


def generate_problems(min_num_locations, max_num_locations, repetitions, draw):
    for num_locations in trange(min_num_locations, max_num_locations + 1):
        for i in range(repetitions):
            index = f'{num_locations:03}-{i+1:02}'
            problem = generate_problem(f'tireworld-{index}', num_locations, draw)
            with open(f'p{index}.pddl', 'w') as f:
                f.write(problem_to_string(problem))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-num-locations', type=int, default=8)
    parser.add_argument('--max-num-locations', type=int, default=100)
    parser.add_argument('-r',
                        '--repetitions',
                        help='number of problems for each number of locations',
                        type=int,
                        default=10)
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    parser.add_argument('--draw', help='draw the generated problems', action='store_true')
    args = parser.parse_args()
    random.seed(args.seed)
    generate_problems(args.min_num_locations, args.max_num_locations, args.repetitions, args.draw)


if __name__ == '__main__':
    main()
