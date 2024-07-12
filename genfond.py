import argparse
from genfond.feature_generator import FeaturePool
from genfond.solver import Solver
from genfond.policy import Policy, PolicyType
from genfond.datalog_policy import DatalogPolicy
from genfond.generate_policy import generate_policy
from genfond.execute_policy import execute_policy
from genfond.config_handler import ConfigHandler
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
from filelock import FileLock
import csv

log = logging.getLogger(__name__)


def solve(
    domain,
    problems,
    type,
    config,
    complexity,
    max_cost=None,
    all_generators=True,
    enforce_highest_complexity=False,
):
    stats = dict()
    log.debug('Generating feature pool ...')
    feature_pool = FeaturePool(
        domain,
        problems,
        config=config,
        max_complexity=complexity,
        all_generators=all_generators,
    )
    stats['featurePoolSize'] = len(feature_pool.features)
    log.debug('Generating ASP instance ...')
    asp_instance = feature_pool.to_clingo()
    state_counts = [len(sg.nodes) for sg in feature_pool.state_graphs.values()]
    edge_counts = [len(node.children) for sg in feature_pool.state_graphs.values() for node in sg.nodes.values()]
    stats['numStates'] = sum(state_counts)
    stats['numTransitions'] = sum(edge_counts)
    if max_cost and enforce_highest_complexity and max_cost < complexity:
        log.info(f'No solution possible for {pnames(problems)}'
                 f' with enforced max complexity {complexity} and max cost {max_cost}')
        return None
    log.info(
        'Solving {} with {} features, {} concepts, {} roles ({}); up to {}complexity {},{} and {} = {} states'.format(
            ", ".join([p.name for p in problems]), len(feature_pool.features), len(feature_pool.concepts),
            len(feature_pool.roles), 'unrestricted' if all_generators else 'restricted',
            "enforced " if enforce_highest_complexity else "",
            complexity, f' max cost {max_cost},' if max_cost else '', " + ".join([str(s) for s in state_counts]),
            sum(state_counts)))
    if type == 'state':
        solve_prog = 'solve_state_constraints.lp'
        policy_type = PolicyType.CONSTRAINED
    elif type == 'trans':
        solve_prog = 'solve_trans_constraints.lp'
        policy_type = PolicyType.CONSTRAINED
    elif not type or type == 'exact':
        solve_prog = 'solve.lp'
        policy_type = PolicyType.EXACT
    elif type == 'datalog':
        solve_prog = 'solve_datalog.lp'
        policy_type = PolicyType.DATALOG
    else:
        raise ValueError(f'Unknown constraint type {constraint_type}')
    solver = Solver(asp_instance,
                    config['num_threads'],
                    max_cost=max_cost,
                    min_feature_complexity=complexity if enforce_highest_complexity else None,
                    solve_prog=solve_prog)
    if not solver.solve():
        log.info('No solution found')
        return None
    solution = solver.solution
    stats |= {
        'clingoAtoms': solver.statistics['problem']['lp']['atoms'],
        'clingoRules': solver.statistics['problem']['lp']['rules'],
        'clingoCpuTime': solver.statistics['summary']['times']['cpu'],
    }
    log.debug(f'ASP instance:\n{asp_instance}')
    log.debug(f'Solution: {solution}')
    log.debug(f'f_selected: {solution.get("f_selected", [])}')
    log.debug(f'f_distinguished: {solution.get("f_distinguished", [])}')
    try:
        policy = generate_policy(solution, policy_type=policy_type)
    except KeyError as e:
        log.error(f'Error during policy generation: {e}')
        raise
    return policy, stats


def pnames(problems):
    return ", ".join([p.name for p in problems])


def signal_handler(sig, frame):
    log.info(f'Received {signal.Signals(sig).name}, exiting ...')
    os.kill(os.getpid(), signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domain_file')
    parser.add_argument('problem_file', nargs='*')
    parser.add_argument('--name', help='Name of the problem set (default: domain name)')
    parser.add_argument('--output', '-o', help='Output file for the resulting policy (as pickle dump)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--stats', help='file to dump stats to')
    parser.add_argument('--config', help='config file for parameters')
    parser.add_argument('--type',
                        choices=['exact', 'state', 'trans', 'datalog'],
                        default='state',
                        help='generate policies of the given type')
    config_args = parser.add_argument_group('config', 'Overwrite config parameters')
    config_args.add_argument('--min-complexity', type=int, help='start policy search with this max complexity')
    config_args.add_argument('--max-complexity', type=int, help='stop policy search with this max complexity')
    config_args.add_argument('-i', '--policy-iterations', type=int, help='number of policy iterations for testing')
    config_args.add_argument('--policy-steps',
                             type=int,
                             help='number of steps to execute policy for testing (0 for no limit)')
    config_args.add_argument('-n',
                             '--num-threads',
                             type=int,
                             help='number of threads to use; "None" uses all available threads')
    config_args.add_argument('--max-memory', type=int, help='maximum memory to use in MB')
    config_args.add_argument('--dump-failed-policies', action='store_true', help='dump failed policies to file')
    config_args.add_argument('--keep-going', action='store_true', help='keep going after one training problem failed')
    config_args.add_argument('--continue-after-error',
                             action='store_true',
                             help='continue after error in policy execution')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-8s %(message)s')
    logging.getLogger('genfond.generate_rule_policy').setLevel(logging.ERROR)
    if not args.verbose:
        logging.getLogger('genfond.execute_policy').setLevel(logging.CRITICAL)
        logging.getLogger('genfond.execute_datalog_policy').setLevel(logging.CRITICAL)
        logging.getLogger('genfond.execute_rule_policy').setLevel(logging.CRITICAL)
    else:
        logging.getLogger('genfond.execute_policy').setLevel(logging.WARN)
        logging.getLogger('genfond.execute_datalog_policy').setLevel(logging.WARN)
        logging.getLogger('genfond.execute_rule_policy').setLevel(logging.WARN)
    signal.signal(signal.SIGINT, signal_handler)
    config = ConfigHandler(args.config, args.type, vars(args))
    if config['max_memory']:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (config['max_memory'] * 1024 * 1024, hard))
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
    name = args.name if args.name else domain.name
    log.info(f'Starting policy generation for domain {name}')
    log.info(f'Generating policies of type {args.type}')
    if args.type == 'datalog':
        policy = DatalogPolicy({}, {})
    else:
        policy = Policy({}, {})
    solver_problems = []
    last_complexity = config['min_complexity']
    succs = []
    considered_problems = problems.copy()
    considered_problems.sort(key=lambda p: len(p.objects))
    log.info(f'Considering problems: {pnames(considered_problems)}')
    failure_reason = ''
    i = 0
    max_i = -1

    log.info('')
    log.info('=' * 80)
    log.info('')

    while i < len(considered_problems):
        problem = considered_problems[i]
        try:
            log.info(f'Testing policy on {problem.name} {config["policy_iterations"]} times ...')
            # Execute policy policy_iterations times
            for _ in tqdm.trange(config['policy_iterations'], disable=None):
                execute_policy(domain, problem, policy, config)
            log.info(f'Policy already solves {problem.name}')
            log.info(f'Appending {problem.name} to successful problems')
            succs.append(problem)
            max_i = max(max_i, i)
            i += 1
            log.info('')
            continue
        except RuntimeError:
            pass
        log.info('Policy does not solve {}'.format(problem.name))
        log.info('')
        old_i = i
        if i > max_i:
            log.info(f'--- Resetting solver problems to just {problem.name} and starting over ---')
            solver_problems = [problem]
        else:
            log.info(f'--- Appending {problem.name} to solver problems and starting over ---')
            solver_problems.append(problem)
        log.info('')
        i = 0
        error = False
        new_policy = None
        for j, all_generators in itertools.product(range(last_complexity, config['max_complexity'] + 1),
                                                   [False, True]):
            if new_policy and new_policy.cost[0] <= j:
                break
            try:
                log.info(f'Starting solver for {pnames(solver_problems)} with max complexity {j}')
                solve_wall_time_start = time.perf_counter()
                solve_cpu_time_start = time.process_time()
                solution = solve(
                    domain,
                    solver_problems,
                    type=args.type,
                    config=config,
                    complexity=j,
                    all_generators=all_generators,
                    max_cost=new_policy.cost[0] - 1 if new_policy else None,
                    enforce_highest_complexity=True if j > last_complexity else False,
                )
            except (RuntimeError, MemoryError) as e:
                if 'Id out of range' in str(e):
                    failure_reason = 'id'
                elif isinstance(e, MemoryError):
                    failure_reason = 'memory'
                else:
                    failure_reason = str(e)
                log.warning(
                    f'Error during policy generation for {pnames(solver_problems)} with max complexity {j}: {e}')
                break
            finally:
                solve_wall_time = time.perf_counter() - solve_wall_time_start
                solve_cpu_time = time.process_time() - solve_cpu_time_start
                log.info('Solver wall time: {:.2f}s'.format(solve_wall_time))
                log.info('Solver CPU time: {:.2f}s'.format(solve_cpu_time))
                total_solve_cpu_time += solve_cpu_time
            if solution:
                i_policy, solve_stats = solution
                if new_policy and new_policy.cost <= i_policy.cost:
                    log.info('Found new policy, but not better than old policy')
                    continue
                if new_policy and new_policy.cost > i_policy.cost:
                    log.info(
                        f'Found another policy with lower cost ({i_policy.cost}) than old policy ({new_policy.cost})')
                else:
                    log.info(f'Found first policy with cost {i_policy.cost} for'
                             f' {pnames(solver_problems)} with max complexity {j}')
                log.info(f'New policy: {i_policy}')
                log.info('Verifying new policy on solved problems')
                try:
                    for problem in tqdm.tqdm(solver_problems, disable=None):
                        execute_policy(domain, problem, i_policy, config)
                except RuntimeError:
                    log.critical(f'New policy does not solve {problem.name}')
                    if config['dump_failed_policies']:
                        h = hash(i_policy)
                        with open(f'failed_policy-{h}.pickle', 'wb') as f:
                            pickle.dump(i_policy, f)
                        log.critical(f'Dumped failed policy to failed_policy-{h}.pickle')
                    if not config['continue_after_error']:
                        sys.exit(1)
                    continue
                last_complexity = j
                new_policy = i_policy
                stats = solve_stats
                best_solve_wall_time = solve_wall_time
                best_solve_cpu_time = solve_cpu_time
            elif not new_policy:
                failure_reason = 'maxcomplexity'
        if new_policy:
            policy = new_policy
            if args.output:
                with open(args.output, 'wb') as f:
                    pickle.dump(policy, f)
            succs = []
            log.info('Resetting successful problems')
            log.info('')
        else:
            log.error(f'No policy found for {problem.name} with max complexity {j}')
            # Delete last element in solver_problems
            if not config['keep_going']:
                error = True
                log.info('')
                break
            log.info(f'Removing {problem.name} from solver problems')
            log.info('')
            del considered_problems[old_i]
            if old_i <= max_i:
                max_i -= 1

    log.info('=' * 80)
    log.info('')

    if error:
        log.info('Verifying policy from start because an error occured ...')
        for problem in tqdm.tqdm([p for p in problems if p not in succs], disable=None):
            try:
                for _ in tqdm.trange(config['policy_iterations'], leave=False, disable=None):
                    execute_policy(domain, problem, policy, config)
                succs.append(problem)
            except RuntimeError:
                log.error('Policy does not solve {}'.format(problem.name))
        log.info('')
    log.info('Policy solves {} out of {} problems, unsolved: {}'.format(
        len(succs), len(problems), pnames([p for p in problems if p not in succs])))
    log.info('Final policy: {}'.format(policy))
    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(policy, f)
    total_wall_time = time.perf_counter() - total_wall_time_start
    total_cpu_time = time.process_time() - total_cpu_time_start
    mem_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024
    stats = {
        'domain': name,
        'constraintType': args.type,
        'totalWallTime': total_wall_time,
        'totalCpuTime': total_cpu_time,
        'bestSolveCpuTime': best_solve_cpu_time,
        'bestSolveWallTime': best_solve_wall_time,
        'totalSolveCpuTime': total_solve_cpu_time,
        'problems': len(problems),
        'solved': len(succs),
        'trainProblems': len(solver_problems),
        'maxTrainProblemSize': max(len(p.objects) for p in solver_problems),
        'maxProblemSize': max(len(p.objects) for p in problems),
        'memUsage': mem_usage,
        #'numFeatures': len(policy.features),
        'maxFeatureComplexity': last_complexity,
        #'numConstraints': max(len(policy.state_constraints), len(policy.constraints)),
        'cost': policy.cost[0],
        'failureReason': failure_reason,
        **stats,
    }

    log.info('Total wall time: {:.2f}s'.format(total_wall_time))
    log.info('Best policy solver CPU time: {:.2f}s'.format(best_solve_cpu_time))
    log.info('Best policy solver wall time: {:.2f}s'.format(best_solve_wall_time))
    log.info('Total solver CPU time: {:.2f}s'.format(total_solve_cpu_time))
    log.info('Total CPU time: {:.2f}s'.format(total_cpu_time))
    log.info('Total memory usage: {:.2f}MB'.format(mem_usage))
    if args.stats:
        lock = FileLock(args.stats + '.lock')
        with lock:
            file_exists = os.path.isfile(args.stats)
            with open(args.stats, 'a') as f:
                writer = csv.DictWriter(f, fieldnames=stats.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(stats)
    if len(considered_problems) == len(problems):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
