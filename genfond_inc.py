import argparse
from genfond.feature_generator import FeaturePool
from genfond.solver import Solver
from genfond.policy import Policy, generate_policy
from genfond.execute_policy import execute_policy
import logging
import sys
import pddl
import resource

log = logging.getLogger(__name__)


def solve(domain, problems, num_threads, complexity):
    feature_pool = FeaturePool(domain, problems, complexity)
    asp_instance = feature_pool.to_clingo()
    log.info('Solving {} with complexity {}'.format(", ".join([p.name for p in problems]), complexity))
    solver = Solver(asp_instance, num_threads)
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
    parser.add_argument('-n',
                        '--num-threads',
                        type=int,
                        default=None,
                        help='number of threads to use; "None" uses all available threads')
    parser.add_argument('--max-memory', type=int, default=None, help='maximum memory to use in MB')
    args = parser.parse_args()
    if args.verbose:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO
    logging.basicConfig(level=loglevel)
    logging.getLogger('genfond').setLevel(logging.CRITICAL)
    if args.max_memory:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (args.max_memory * 1024 * 1024, hard))
    log.debug('Parsing domain ...')
    domain = pddl.parse_domain(args.domain_file)
    log.info('Starting policy generation for domain {}'.format(domain.name))
    log.debug('Parsing problems ...')
    problems = [pddl.parse_problem(f) for f in args.problem_file]
    policy = Policy({}, {})
    solver_problems = []
    last_complexity = args.min_complexity
    for problem in problems:
        try:
            log.info(f'Testing policy on {problem.name}')
            execute_policy(domain, problem, policy, 10000)
            log.info(f'Policy already solves {problem.name}')
            continue
        except RuntimeError:
            pass
        log.info('Policy does not solve {}'.format(problem.name))
        solver_problems.append(problem)
        for i in range(last_complexity, args.max_complexity):
            try:
                log.info(f'Starting solver for {", ".join([p.name for p in solver_problems])} with max complexity {i}')
                new_policy = solve(domain, solver_problems, args.num_threads, i)
            except (RuntimeError, MemoryError) as e:
                log.warning(f'Error during policy generation for {problem.name} with max complexity {i}: {e}')
                break
            if new_policy is not None:
                policy = new_policy
                last_complexity = i
                break
        if new_policy is None:
            log.error('No policy found for {} with max complexity {}'.format(problem.name, i))
            # Delete last element in solver_problems
            solver_problems.pop()
            continue
        else:
            policy = new_policy
    log.info('Verifying policy ...')
    succs = []
    for problem in problems:
        try:
            execute_policy(domain, problem, policy, 10000)
            succs.append(problem)
        except RuntimeError:
            log.error('Policy does not solve {}'.format(problem.name))
    log.info('Policy solves {} out of {} problems, unsolved: {}'.format(
        len(succs), len(problems), ", ".join([p.name for p in problems if p not in succs])))
    log.info('Final policy: {}'.format(policy))
    if len(succs) == len(problems):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()