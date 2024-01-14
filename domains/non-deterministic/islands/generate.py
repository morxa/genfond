import argparse
import pddl
import os.path
from pddl.formatter import problem_to_string
import random
from pddl.logic.terms import Variable
from pddl.logic import Predicate
from pddl.core import Problem
import tqdm

import logging

log = logging.getLogger(__name__)


def generate_variant(domain, problem, name):
    init = []
    bridge_clear = True
    x = Variable('x', ['monkey'])
    monkey_on_bridge = Predicate('monkey-on-bridge', x)
    for fact in problem.init:
        if fact.name == 'bridge-clear':
            continue
        elif fact.name == 'monkey-at' and random.choice([True, False]):
            bridge_clear = False
            f = monkey_on_bridge(fact.terms[0])
            log.debug(f'Changing {fact} to {f}')
            init.append(f)
        else:
            log.debug(f'Keeping {fact}')
            init.append(fact)
    if bridge_clear:
        init.append(Predicate('bridge-clear'))
    else:
        init.append(Predicate('bridge-occupied'))
    return Problem(name=name, domain=domain, objects=problem.objects, init=init, goal=problem.goal)


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('problem', type=str, help='original problem file', nargs='+')
    parser.add_argument('--num-variants', '-n', type=int, help='number of variants to generate', default=5)
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)
    domain = pddl.parse_domain(os.path.join(os.path.dirname(__file__), 'domain.pddl'))
    for pfile in tqdm.tqdm(args.problem):
        for i in tqdm.trange(args.num_variants, leave=False):
            problem = pddl.parse_problem(pfile)
            name = f'{problem.name}-{i+1:02}'
            variant = generate_variant(domain, problem, name)
            with open(f'{variant.name}.pddl', 'w') as f:
                f.write(problem_to_string(variant))


if __name__ == '__main__':
    main()
