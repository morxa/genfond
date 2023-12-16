import argparse
from genfond.feature_generator import FeaturePool
from genfond.solver import Solver
from genfond.policy import Policy, generate_policy
from genfond.execute_policy import execute_policy
import logging
import sys
import pddl
import resource
import tqdm
import pickle
import time
import signal
import os
import itertools
import resource

log = logging.getLogger(__name__)


def solve(domain,
          problems,
          num_threads,
          complexity,
          use_new_solver=False,
          max_cost=None,
          all_generators=True,
          enforce_highest_complexity=False):
    feature_pool = FeaturePool(domain, problems, complexity, all_generators=all_generators)
    asp_instance = feature_pool.to_clingo()
    sg_sizes = [len(sg.nodes) for sg in feature_pool.state_graphs.values()]
    if max_cost and enforce_highest_complexity and max_cost < complexity:
        log.info(f'No solution possible for {pnames(problems)}'
                 f' with enforced max complexity {complexity} and max cost {max_cost}')
        return None
    log.info('Solving {} with {} {} features up to {}complexity {},{} and {} = {} states'.format(
        ", ".join([p.name for p in problems]), len(feature_pool.features),
        'unrestricted' if all_generators else 'restricted', "enforced " if enforce_highest_complexity else "",
        complexity, f' max cost {max_cost},' if max_cost else '', " + ".join([str(s)
                                                                              for s in sg_sizes]), sum(sg_sizes)))
    solver = Solver(asp_instance,
                    num_threads,
                    max_cost=max_cost,
                    min_feature_complexity=complexity if enforce_highest_complexity else None,
                    solve_prog='solve_new.lp' if use_new_solver else 'solve.lp')
    if not solver.solve():
        log.info('No solution found')
        return None
    solution = solver.solution
    policy = generate_policy(solution)
    return policy


def pnames(problems):
    return ", ".join([p.name for p in problems])


def signal_handler(sig, frame):
    log.info(f'Received {signal.Signals(sig).name}, exiting ...')
    os.kill(os.getpid(), signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domain_file')
    parser.add_argument('problem_file', nargs='*')
    parser.add_argument('--output', '-o', help='Output file for the resulting policy (as pickle dump)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--min-complexity', type=int, default=2, help='start policy search with this max complexity')
    parser.add_argument('--max-complexity', type=int, default=9, help='stop policy search with this max complexity')
    parser.add_argument('--continue-steps',
                        type=int,
                        default=None,
                        help='continue policy search for this number of steps after finding a policy')
    parser.add_argument('-i',
                        '--policy-iterations',
                        type=int,
                        default=100,
                        help='number of policy iterations for testing')
    parser.add_argument('--policy-steps',
                        type=int,
                        default=10000,
                        help='number of steps to execute policy for testing (0 for no limit)')
    parser.add_argument('-n',
                        '--num-threads',
                        type=int,
                        default=None,
                        help='number of threads to use; "None" uses all available threads')
    parser.add_argument('--max-memory', type=int, default=None, help='maximum memory to use in MB')
    parser.add_argument('--new-solver', action='store_true', help='use new solver')
    parser.add_argument('--dump-failed-policies', action='store_true', help='dump failed policies to file')
    parser.add_argument('--keep-going', action='store_true', help='keep going after one training problem failed')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')
    if not args.verbose:
        logging.getLogger('genfond.execute_policy').setLevel(logging.CRITICAL)
    signal.signal(signal.SIGINT, signal_handler)
    if args.max_memory:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (args.max_memory * 1024 * 1024, hard))
    total_wall_time_start = time.perf_counter()
    total_cpu_time_start = time.process_time()
    total_solve_cpu_time = 0
    best_solve_cpu_time = 0
    best_solve_wall_time = 0
    log.info('Parsing domain ...')
    domain = pddl.parse_domain(args.domain_file)
    log.info('Parsing problems ...')
    problems = []
    for f in tqdm.tqdm(args.problem_file, disable=None):
        problems.append(pddl.parse_problem(f))
    log.info('Starting policy generation for domain {}'.format(domain.name))
    policy = Policy({}, {})
    solver_problems = []
    last_complexity = args.min_complexity
    verified = []
    queue = problems.copy()
    queue.sort(key=lambda p: len(p.objects))
    for problem in queue:
        try:
            log.info(f'Testing policy on {problem.name} {args.policy_iterations} times ...')
            # Execute policy policy_iterations times
            for _ in tqdm.trange(args.policy_iterations, disable=None):
                execute_policy(domain, problem, policy, args.policy_steps)
            log.info(f'Policy already solves {problem.name}')
            verified.append(problem)
            continue
        except RuntimeError:
            pass
        log.info('Policy does not solve {}'.format(problem.name))
        solver_problems.append(problem)
        new_policy = None
        for i, all_generators in itertools.product(range(last_complexity, args.max_complexity + 1), [False, True]):
            if new_policy and (not args.continue_steps or i > last_complexity + args.continue_steps):
                break
            try:
                log.info(f'Starting solver for {pnames(solver_problems)} with max complexity {i}')
                solve_wall_time_start = time.perf_counter()
                solve_cpu_time_start = time.process_time()
                i_policy = solve(
                    domain,
                    solver_problems,
                    args.num_threads,
                    i,
                    args.new_solver,
                    all_generators=all_generators,
                    max_cost=new_policy.cost[0] - 1 if new_policy else None,
                    enforce_highest_complexity=True if i > last_complexity else False,
                )
            except (RuntimeError, MemoryError) as e:
                log.warning(
                    f'Error during policy generation for {pnames(solver_problems)} with max complexity {i}: {e}')
                break
            finally:
                solve_wall_time = time.perf_counter() - solve_wall_time_start
                solve_cpu_time = time.process_time() - solve_cpu_time_start
                log.info('Solver wall time: {:.2f}s'.format(solve_wall_time))
                log.info('Solver CPU time: {:.2f}s'.format(solve_cpu_time))
                total_solve_cpu_time += solve_cpu_time
            if i_policy:
                if new_policy and new_policy.cost <= i_policy.cost:
                    log.info('Found new policy, but not better than old policy')
                    continue
                if new_policy and new_policy.cost > i_policy.cost:
                    log.info(
                        f'Found another policy with lower cost ({i_policy.cost}) than old policy ({new_policy.cost})')
                else:
                    log.info(f'Found first policy with cost {i_policy.cost} for'
                             f' {pnames(solver_problems)} with max complexity {i}')
                log.info(f'New policy: {i_policy}')
                log.info('Verifying new policy on solved problems')
                try:
                    for problem in tqdm.tqdm(solver_problems, disable=None):
                        execute_policy(domain, problem, i_policy, args.policy_steps)
                except RuntimeError:
                    log.critical('New policy does not solve {}'.format(problem.name))
                    if args.dump_failed_policies:
                        h = hash(i_policy)
                        with open(f'failed_policy-{h}.pickle', 'wb') as f:
                            pickle.dump(i_policy, f)
                        log.critical(f'Dumped failed policy to failed_policy-{h}.pickle')
                    sys.exit(1)
                    continue
                last_complexity = i
                new_policy = i_policy
                best_solve_wall_time = solve_wall_time
                best_solve_cpu_time = solve_cpu_time
        if new_policy:
            policy = new_policy
            if args.output:
                with open(args.output, 'wb') as f:
                    pickle.dump(policy, f)
            # Re-add previously verified problems  to queue if not part of the solver set
            queue += [p for p in verified if p not in solver_problems]
            verified = solver_problems.copy()
        else:
            log.error('No policy found for {} with max complexity {}'.format(problem.name, i))
            if not args.keep_going:
                break
            # Delete last element in solver_problems
            solver_problems.pop()
            continue
    succs = verified
    log.info('Verifying policy ...')
    for problem in tqdm.tqdm([p for p in problems if p not in verified], disable=None):
        try:
            for _ in tqdm.trange(args.policy_iterations, leave=False, disable=None):
                execute_policy(domain, problem, policy, args.policy_steps)
            succs.append(problem)
        except RuntimeError:
            log.error('Policy does not solve {}'.format(problem.name))
    log.info('Policy solves {} out of {} problems, unsolved: {}'.format(
        len(succs), len(problems), pnames([p for p in problems if p not in succs])))
    log.info('Final policy: {}'.format(policy))
    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(policy, f)
    total_wall_time_end = time.perf_counter()
    total_cpu_time_end = time.process_time()
    log.info('Total wall time: {:.2f}s'.format(total_wall_time_end - total_wall_time_start))
    log.info('Best policy solver CPU time: {:.2f}s'.format(best_solve_cpu_time))
    log.info('Best policy solver wall time: {:.2f}s'.format(best_solve_wall_time))
    log.info('Total solver CPU time: {:.2f}s'.format(total_solve_cpu_time))
    log.info('Total CPU time: {:.2f}s'.format(total_cpu_time_end - total_cpu_time_start))
    mem_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * resource.getpagesize()) / 1024**2
    log.info('Total memory usage: {:.2f}MB'.format(mem_usage))
    if len(succs) == len(problems):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
