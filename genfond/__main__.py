import argparse
from genfond.feature_generator import FeaturePool
from genfond.solver import Solver
from genfond.policy import PolicyType
from genfond.rule_policy import Policy
from genfond.datalog_policy import DatalogPolicy
from genfond.generate_policy import generate_policy
from genfond.execute_policy import execute_policy
from genfond.config_handler import ConfigHandler, DEFAULT_TYPE_CONFIGS
from genfond.problem_iterator import ProblemIterator, Result, MAX_COST
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

log = logging.getLogger('genfond')


def solve(
    domain,
    problems,
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
    log.debug(f'ASP instance:\n{asp_instance}')
    if config.get('dump_clingo_program', None):
        with open(config['dump_clingo_program'], 'w') as f:
            f.write(asp_instance)
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
    solver = Solver(asp_instance,
                    config['num_threads'],
                    max_cost=max_cost,
                    min_feature_complexity=complexity if enforce_highest_complexity else None,
                    solve_prog=config['solve_prog'])
    if not solver.solve():
        log.info('No solution found')
        return None
    solution = solver.solution
    stats |= {
        'clingoAtoms': solver.statistics['problem']['lp']['atoms'],
        'clingoRules': solver.statistics['problem']['lp']['rules'],
        'clingoCpuTime': solver.statistics['summary']['times']['cpu'],
    }
    log.debug(f'Solution: {solution}')
    log.debug(f'f_selected: {solution.get("f_selected", [])}')
    log.debug(f'f_distinguished: {solution.get("f_distinguished", [])}')
    try:
        policy = generate_policy(solution, policy_type=PolicyType[config['policy_type']])
    except KeyError as e:
        log.error(f'Error during policy generation: {e}')
        raise
    return policy, stats


def pnames(problems):
    return ", ".join([p.name for p in problems])


def signal_handler(sig, frame):
    log.info(f'Received {signal.Signals(sig).name}, exiting ...')
    os.kill(os.getpid(), signal.SIGTERM)


def solve_iteratively(domain, problems, config):
    policy = None
    queue = problems.copy()
    failure_reason = ''
    total_solve_cpu_time = 0
    best_solve_cpu_time = 0
    best_solve_wall_time = 0
    stats = dict()
    queue.sort(key=lambda p: len(p.objects))
    problem_iterator = ProblemIterator(queue, config['min_complexity'], config['max_complexity'],
                                       config['use_unrestricted_features'])
    for solver_problems, i, all_generators, max_cost in problem_iterator:
        try:
            log.info(f'Starting solver for {pnames(solver_problems)} with max complexity {i}')
            solve_wall_time_start = time.perf_counter()
            solve_cpu_time_start = time.process_time()
            solution = solve(
                domain,
                solver_problems,
                config=config,
                complexity=i,
                all_generators=all_generators,
                max_cost=max_cost if max_cost < MAX_COST else None,
                enforce_highest_complexity=True,
            )
        except (RuntimeError, MemoryError) as e:
            if 'Id out of range' in str(e):
                stats['failureReason'] = 'id'
                problem_iterator.set_last_result(Result.OUT_OF_RESOURCES)

            elif isinstance(e, MemoryError):
                stats['failureReason'] = 'memory'
                problem_iterator.set_last_result(Result.OUT_OF_RESOURCES)

            else:
                problem_iterator.set_last_result(Result.UNKNOWN)
                stats['failureReason'] = str(e)
            log.warning(f'Error during policy generation for {pnames(solver_problems)} with max complexity {i}: {e}')
            continue
        finally:
            solve_wall_time = time.perf_counter() - solve_wall_time_start
            solve_cpu_time = time.process_time() - solve_cpu_time_start
            log.info('Solver wall time: {:.2f}s'.format(solve_wall_time))
            log.info('Solver CPU time: {:.2f}s'.format(solve_cpu_time))
            total_solve_cpu_time += solve_cpu_time
        if solution:
            new_policy, solve_stats = solution
            log.info(f'Found policy with cost {new_policy.cost} for'
                     f' {pnames(solver_problems)} with max complexity {i}')
            log.info(f'New policy: {new_policy}')
            log.info('Verifying new policy on solved problems')
            try:
                for problem in tqdm.tqdm(solver_problems, disable=None):
                    execute_policy(domain, problem, new_policy, config)
            except RuntimeError:
                log.critical('New policy does not solve {}'.format(problem.name))
                if config['dump_failed_policies']:
                    h = hash(new_policy)
                    with open(f'failed_policy-{h}.pickle', 'wb') as f:
                        pickle.dump(new_policy, f)
                    log.critical(f'Dumped failed policy to failed_policy-{h}.pickle')
                if not config['continue_after_error']:
                    sys.exit(1)
                continue
            policy = new_policy
            stats = solve_stats
            stats['maxFeatureComplexity'] = i
            best_solve_wall_time = solve_wall_time
            best_solve_cpu_time = solve_cpu_time
            problem_iterator.set_last_result(Result.SUCCESS, cost=new_policy.cost[0])
            policy = new_policy
            log.info(f'Testing policy on unsolved problems {config["policy_iterations"]} times ...')
            for problem in tqdm.tqdm(problem_iterator.get_unsolved_problems(), disable=None):
                try:
                    log.debug(f'Testing policy on {problem.name} {config["policy_iterations"]} times ...')
                    # Execute policy policy_iterations times
                    for _ in range(config['policy_iterations']):
                        plan = execute_policy(domain, problem, policy, config)
                    log.debug(f'Policy already solves {problem.name} (plan length {len(plan)})')
                    problem_iterator.set_solved(problem)
                except RuntimeError:
                    log.info('Policy does not solve {}'.format(problem.name))
                    break
        else:
            log.error('No policy found for {} with max complexity {}'.format(
                ", ".join([p.name for p in solver_problems]), i))
            problem_iterator.set_last_result(Result.NO_SOLUTION)
            stats['failureReason'] = 'maxcomplexity'
    stats.update({
        'trainProblems': len(problem_iterator.active_problems),
        'maxTrainProblemSize': max(len(p.objects) for p in problem_iterator.active_problems) if policy else 0,
        'bestSolveCpuTime': best_solve_cpu_time,
        'bestSolveWallTime': best_solve_wall_time,
        'totalSolveCpuTime': total_solve_cpu_time,
    })
    return policy, [p for p in problems if problem_iterator.solved[p.name]], stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('domain_file')
    parser.add_argument('problem_file', nargs='*')
    parser.add_argument('--one-shot', action='store_true', help='solve all problems at once')
    parser.add_argument('--name', help='Name of the problem set (default: domain name)')
    parser.add_argument('--output', '-o', help='Output file for the resulting policy (as pickle dump)')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--stats', help='file to dump stats to')
    parser.add_argument('--config', type=argparse.FileType('r'), help='config file for parameters')
    parser.add_argument('--dump-config', help='dump effective config to file')
    parser.add_argument('--dump-clingo-program', help='dump clingo program to file')
    parser.add_argument('--type', choices=DEFAULT_TYPE_CONFIGS.keys(), help='generate policies of the given type')
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
    signal.signal(signal.SIGINT, signal_handler)
    config = ConfigHandler(args.config, args.type, vars(args))
    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            f.write(config.dump())
    for component, loglevel in config['log'].items():
        component = f'genfond.{component}'
        log.info(f'Setting log level for {component} to {loglevel}')
        logging.getLogger(component).setLevel(loglevel)
    if config['max_memory']:
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (config['max_memory'] * 1024 * 1024, hard))
    total_wall_time_start = time.perf_counter()
    total_cpu_time_start = time.process_time()
    log.info('Parsing domain ...')
    domain = pddl.parse_domain(args.domain_file)
    log.info('Parsing problems ...')
    problems = []
    for f in tqdm.tqdm(args.problem_file, disable=None):
        problems.append(pddl.parse_problem(f))
    name = args.name if args.name else domain.name
    log.info('Starting policy generation for domain {}'.format(name))
    log.info(f'Generating policies of type {args.type}')
    stats = {
        'domain': name,
        'constraintType': args.type,
    }
    if args.one_shot:
        solve_cpu_time_start = time.process_time()
        solution = solve(domain,
                         problems,
                         config=config,
                         complexity=config['max_complexity'],
                         all_generators=False,
                         dump_clingo_program=args.dump_clingo_program)
        solve_cpu_time = time.process_time() - solve_cpu_time_start
        log.info(f'CPU time: {solve_cpu_time:.2f}s')
        mem_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024
        log.info('Memory usage: {:.2f}MB'.format(mem_usage))
        if solution:
            policy, solve_stats = solution
            stats.update(solve_stats)
        else:
            log.error('No policy found')
            sys.exit(1)
        log.info(f'Policy:\n{policy}')
        if args.output:
            with open(args.output, 'wb') as f:
                pickle.dump(policy, f)
        sys.exit(0)
    policy, succs, solve_stats = solve_iteratively(domain, problems, config)
    stats.update(solve_stats)
    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(policy, f)
    log.info('Verifying policy ...')
    for problem in tqdm.tqdm([p for p in problems if p not in succs], disable=None):
        try:
            for _ in tqdm.trange(config['policy_iterations'], leave=False, disable=None):
                execute_policy(domain, problem, policy, config)
            succs.append(problem)
        except RuntimeError:
            log.error('Policy does not solve {}'.format(problem.name))
    log.info('Policy solves {} out of {} problems, unsolved: {}'.format(
        len(succs), len(problems), pnames([p for p in problems if p not in succs])))
    log.info('Final policy: {}'.format(policy))
    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(policy, f)
    total_wall_time = time.perf_counter() - total_wall_time_start
    total_cpu_time = time.process_time() - total_cpu_time_start
    mem_usage = (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024
    stats.update({
        'totalWallTime': total_wall_time,
        'totalCpuTime': total_cpu_time,
        'problems': len(problems),
        'solved': len(succs),
        'maxProblemSize': max(len(p.objects) for p in problems) if succs else 0,
        'memUsage': mem_usage,
        #'numFeatures': len(policy.features),
        #'numConstraints': max(len(policy.state_constraints), len(policy.constraints)),
        'cost': policy.cost[0] if policy else 0,
    })

    log.info('Total wall time: {:.2f}s'.format(total_wall_time))
    log.info('Best policy solver CPU time: {:.2f}s'.format(stats['bestSolveCpuTime']))
    log.info('Best policy solver wall time: {:.2f}s'.format(stats['bestSolveWallTime']))
    log.info('Total solver CPU time: {:.2f}s'.format(stats['totalSolveCpuTime']))
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
    if len(succs) == len(problems):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
