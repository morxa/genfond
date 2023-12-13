import pddl
import itertools
from pddl.logic import Predicate, constants, variables
from pddl.logic.effects import AndEffect
from pddl.core import Problem
from pddl.formatter import problem_to_string
import argparse
import os.path
from tqdm import trange
import random


# Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def generate_problem(name, num_locations, prob_random_connection):
    domain = pddl.parse_domain(os.path.join(os.path.dirname(__file__), 'domain.pddl'))
    locations = constants(' '.join([f'l{i}' for i in range(num_locations)]), 'location')
    x, y = variables('x y', ['location'])
    spare_in = Predicate('spare-in', x)
    vehicle_at = Predicate('vehicle-at', x)
    road = Predicate('road', x, y)
    # randomly sample a safe road with spare tires
    road_locations = random.sample(locations, random.randint(2, len(locations)))
    init = [vehicle_at(road_locations[0])]
    init += [road(*pair) for pair in pairwise(road_locations)]
    init += [road(*pair) for pair in pairwise(reversed(road_locations))]
    init += [spare_in(l) for l in road_locations[1:-1]]
    # randomly add other conections
    for start, end in pairwise(locations):
        if random.random() > prob_random_connection:
            init.append(road(start, end))
            init.append(road(end, start))
    problem = Problem(name, domain=domain, objects=locations, init=init, goal=vehicle_at(road_locations[-1]))
    return problem


def generate_problems(min_num_locations, max_num_locations, repetitions, prob_random_connection):
    for num_locations in trange(min_num_locations, max_num_locations + 1):
        for i in range(repetitions):
            index = f'{num_locations:03}-{i+1:02}'
            problem = generate_problem(f'tireworld-{index}', num_locations, prob_random_connection)
            with open(f'p{index}.pddl', 'w') as f:
                f.write(problem_to_string(problem))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-num-location', type=int, default=2)
    parser.add_argument('--max-num-locations', type=int, default=20)
    parser.add_argument('--prob-random-connection', type=float, default=0.2)
    parser.add_argument('-r',
                        '--repetitions',
                        help='number of problems for each number of locations',
                        type=int,
                        default=20)
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    generate_problems(args.min_num_location, args.max_num_locations, args.repetitions, args.prob_random_connection)


if __name__ == '__main__':
    main()
