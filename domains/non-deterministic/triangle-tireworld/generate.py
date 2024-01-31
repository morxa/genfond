import pddl
import itertools
from pddl.logic import Predicate, variables
from pddl.logic.terms import Constant
from pddl.core import Problem
from pddl.formatter import problem_to_string
import argparse
import os.path
from tqdm import trange
import pygraphviz


# Taken from https://docs.python.org/3/library/itertools.html#itertools-recipes
def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def draw(name, locations, facts, goal):
    g = pygraphviz.AGraph(strict=False, directed=True)
    g.node_attr['shape'] = 'box'
    for location in locations:
        g.add_node(location)
    for fact in facts:
        if fact.name == 'road':
            g.add_edge(fact.terms[0], fact.terms[1])
        elif fact.name == 'spare-in':
            n = g.get_node(fact.terms[0])
            n.attr['color'] = 'green'
        elif fact.name == 'vehicle-at':
            n = g.get_node(fact.terms[0])
            n.attr['shape'] = 'cds'
    gnode = g.get_node(goal)
    gnode.attr['peripheries'] = 2
    g.layout(prog='neato')
    g.draw(f'{name}.png')


def generate_problem(name, size, draw_problem):
    domain = pddl.parse_domain(os.path.join(os.path.dirname(__file__), 'domain.pddl'))
    x, y = variables('x y', ['location'])
    spare_in = Predicate('spare-in', x)
    vehicle_at = Predicate('vehicle-at', x)
    road = Predicate('road', x, y)
    not_flattire = Predicate('not-flattire')
    grid = dict()
    locations = []
    for x in range(0, 2 * size + 1):
        grid[x] = dict()
        for y in range(x, 4 * size - x + 2):
            if x % 2 == y % 2:
                loc = Constant(f'l{x}_{y}', 'location')
                locations.append(loc)
                grid[x][y] = loc
    start_location = grid[0][0]
    init = [vehicle_at(start_location), not_flattire]
    goal_location = grid[0][4 * size]
    for x, xs in grid.items():
        for y, location in xs.items():
            if (x % 2 == 1 or x == y or x + y == 4 * size) and location not in [start_location, goal_location]:
                init.append(spare_in(location))
            try:
                init.append(road(location, grid[x + 1][y + 1]))
            except KeyError:
                pass
            try:
                init.append(road(location, grid[x - 1][y + 1]))
            except KeyError:
                pass
            if x % 2 == 0:
                try:
                    init.append(road(location, grid[x][y + 2]))
                except KeyError:
                    pass

    if draw_problem:
        draw(name, locations, init, goal_location)
    problem = Problem(name, domain=domain, objects=locations, init=init, goal=vehicle_at(goal_location))
    return problem


def generate_problems(min_size, max_size, draw):
    for size in trange(min_size, max_size + 1):
        index = f'{size:03}'
        problem = generate_problem(f'triangle-tireworld-{index}', size, draw)
        with open(f'p{index}.pddl', 'w') as f:
            f.write(problem_to_string(problem))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-size', type=int, default=1, help='the shortest path will be 2*size+1')
    parser.add_argument('--max-size', type=int, default=10)
    parser.add_argument('--draw', help='draw the generated problems', action='store_true')
    args = parser.parse_args()
    generate_problems(args.min_size, args.max_size, args.draw)


if __name__ == '__main__':
    main()
