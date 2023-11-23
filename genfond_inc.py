import argparse
from genfond.feature_generator import FeaturePool
from genfond.solver import Solver
from genfond.policy import Policy, generate_policy
from genfond.execute_policy import execute_policy
import logging
import sys
import pddl

log = logging.getLogger(__name__)


def solve(domain, problems, complexity):
    feature_pool = FeaturePool(domain, problems, complexity)
    asp_instance = feature_pool.to_clingo()
    log.info('Solving {} with complexity {}'.format(", ".join([p.name for p in problems]), complexity))
    solver = Solver(asp_instance)
    if not solver.solve():
        log.info('No solution found')
        return None
    solution = solver.solution
    policy = generate_policy(solution)
    log.info('Found policy: {}'.format(policy))
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domain_file')
    parser.add_argument('problem_file', nargs='*')
    parser.add_argument('--output', '-o', help='Output file for the resulting policy (as pickle dump)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--min-complexity', type=int, default=2, help='start policy search with this max complexity')
    parser.add_argument('--max-complexity', type=int, default=9, help='stop policy search with this max complexity')
    args = parser.parse_args()
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(level=loglevel)
    log.debug('Parsing domain ...')
    domain = pddl.parse_domain(args.domain_file)
    log.debug('Parsing problems ...')
    problems = [pddl.parse_problem(f) for f in args.problem_file]
    policy = Policy({}, {})
    solver_problems = []
    last_complexity = args.min_complexity
    for problem in problems:
        try:
            execute_policy(domain, problem, policy)
            log.info('Policy already solves {}'.format(problem.name))
        except RuntimeError:
            log.info('Policy does not solve {}'.format(problem.name))
            solver_problems.append(problem)
            for i in range(last_complexity, args.max_complexity):
                policy = solve(domain, solver_problems, i)
                if policy is not None:
                    last_complexity = i
                    break
            if policy is None:
                log.error('No policy found for {} with max complexity {}'.format(problem.name, i))
                sys.exit(1)
    log.info('All problems solved!')
    log.info('Verifying policy ...')
    for problem in problems:
        execute_policy(domain, problem, policy)
    log.info('Policy verified!')
    log.info('Final policy: {}'.format(policy))


if __name__ == '__main__':
    main()
