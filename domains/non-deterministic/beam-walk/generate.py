import argparse
import itertools
import os.path

import pddl
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


def generate_problem(num_beams, initially_up=True, base_name='beam-walk'):
    domain = pddl.parse_domain(os.path.join(os.path.dirname(__file__), 'domain.pddl'))
    beams = constants(' '.join([f'p{i}' for i in range(num_beams)]), 'location')
    p, p1, p2 = variables('p p1 p2', ['location'])
    up = Predicate('up')
    position = Predicate('position', p)
    next_fwd = Predicate('next-fwd', p1, p2)
    next_bwd = Predicate('next-bwd', p1, p2)
    ladder_at = Predicate('ladder-at', p)
    init = [ladder_at(beams[0]), position(beams[0])]
    if initially_up:
        init.append(up)
    for b1, b2 in pairwise(beams):
        init.append(next_fwd(b1, b2))
        init.append(next_bwd(b2, b1))
    problem = Problem(f'{base_name}-{num_beams}{"" if initially_up else "-down"}',
                      domain=domain,
                      objects=beams,
                      init=init,
                      goal=up & position(beams[-1]))
    return problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exponential', action='store_true', help='Generate problems with exponentially many beams')
    parser.add_argument('--base-name', type=str, default='beam-walk')
    parser.add_argument('min_num_beams', type=int)
    parser.add_argument('max_num_beams', type=int)
    args = parser.parse_args()
    for num_beams in trange(args.min_num_beams, args.max_num_beams + 1):
        if args.exponential:
            num_beams = 2**num_beams
        for initially_up in [True, False]:
            problem = generate_problem(num_beams, initially_up, args.base_name)
            with open(f'p{num_beams:04}{"" if initially_up else "-down"}.pddl', 'w') as f:
                f.write(problem_to_string(problem))


if __name__ == '__main__':
    main()
