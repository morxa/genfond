from .problem_iterator import ProblemIterator, Result, MAX_COST
from .feature_generator import FeaturePool
from .solver import Solver
from .policy import PolicyType
from .execute_policy import execute_policy
from .execute_datalog_policy import NoActionError, CycleError
from .generate_policy import generate_policy
import logging
import time
import pickle
import tqdm
import sys
import statistics

log = logging.getLogger('genfond.solve')


def solve(
    domain,
    problems,
    config,
    complexity,
    max_cost,
    max_prune_cost,
    all_generators=True,
    enforce_highest_complexity=False,
    selected_states=None,
):
    stats = dict()
    log.debug('Generating feature pool ...')
    feature_pool = FeaturePool(
        domain,
        problems,
        config=config,
        max_complexity=complexity,
        all_generators=all_generators,
        selected_states=selected_states,
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
    log.info('Solving {} with {} features, {} concepts, {} roles ({}); up to {}complexity {},{}{} and {} = {} states'.
             format(", ".join([p.name for p in problems]), len(feature_pool.features), len(feature_pool.concepts),
                    len(feature_pool.roles), 'unrestricted' if all_generators else 'restricted',
                    "enforced " if enforce_highest_complexity else "", complexity,
                    f' max cost {max_cost},' if max_cost < MAX_COST else '',
                    f' max prune cost {max_prune_cost},' if max_prune_cost < MAX_COST else '',
                    " + ".join([str(s) for s in state_counts]), sum(state_counts)))
    solver = Solver(asp_instance,
                    config['num_threads'],
                    max_cost=max_cost,
                    max_prune_cost=max_prune_cost,
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


def solve_iteratively(domain, problems, config):
    policy = None
    problems.sort(key=lambda p: len(p.objects))
    total_solve_cpu_time = 0
    best_solve_cpu_time = 0
    best_solve_wall_time = 0
    stats = dict()
    problem_iterator = ProblemIterator(problems, config['min_complexity'], config['max_complexity'],
                                       config['use_unrestricted_features'])
    for solver_problems, i, all_generators, max_cost, max_prune_cost, selected_states in problem_iterator:
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
                max_cost=max_cost,
                max_prune_cost=max_prune_cost,
                enforce_highest_complexity=True,
                selected_states=selected_states,
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
            # log.info('Verifying new policy on solved problems')
            # try:
            #     for problem in tqdm.tqdm(solver_problems, disable=None):
            #         execute_policy(domain, problem, new_policy, config)
            # except RuntimeError:
            #     log.info('New policy does not solve {}'.format(problem.name))
            #     if config['dump_failed_policies']:
            #         h = hash(new_policy)
            #         with open(f'failed_policy-{h}.pickle', 'wb') as f:
            #             pickle.dump(new_policy, f)
            #         log.critical(f'Dumped failed policy to failed_policy-{h}.pickle')
            #     if not config['continue_after_error']:
            #         sys.exit(1)
            #     continue
            policy = new_policy
            stats = solve_stats
            stats['maxFeatureComplexity'] = i
            best_solve_wall_time = solve_wall_time
            best_solve_cpu_time = solve_cpu_time
            problem_iterator.set_last_result(Result.SUCCESS, cost=new_policy.cost)
            policy = new_policy
            log.info(f'Testing policy on unsolved problems {config["policy_iterations"]} times ...')
            for problem in tqdm.tqdm(problems, disable=None):
                log.info(f'Testing policy on {problem.name} {config["policy_iterations"]} times ...')
                plans = []
                traces = []
                solved = True
                for _ in range(config['policy_iterations']):
                    try:
                        plan, trace = execute_policy(domain, problem, policy, config)
                        plans.append(plan)
                        traces.append(trace)
                    except NoActionError as e:
                        log.info(f'Policy does not solve {problem.name}, no action in reachable state')
                        solved = False
                        for state in e.trace.keys():
                            problem_iterator.set_new_state(problem.name, state)
                        problem_iterator.set_new_state(problem.name, e.state)
                    except CycleError as e:
                        log.info(f'Policy does not solve {problem.name}, found cycle of length {len(e.cycle)}')
                        solved = False
                        for state in e.trace.keys():
                            problem_iterator.set_new_state(problem.name, state)
                    except RuntimeError:
                        log.info('Policy does not solve {}'.format(problem.name))
                        solved = False
                if solved:
                    plan_lengths = [len(plan) for plan in plans]
                    log.debug(f'Policy already solves {problem.name}'
                              f'(plan length {statistics.mean(plan_lengths)} +- {statistics.stdev(plan_lengths)})')
                    problem_iterator.set_solved(problem)
                else:
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
