import logging
import pickle
import statistics
import sys
import time
from typing import Any, Collection, Mapping, MutableMapping, Optional

import tqdm
from pddl.core import Domain, Plan, Problem
from tqdm.contrib.logging import logging_redirect_tqdm

from .datalog_policy import DatalogPolicy
from .execute_datalog_policy import CycleError, NoActionError
from .execute_policy import execute_policy
from .feature_generator import FeaturePool
from .generate_policy import generate_policy
from .policy import PolicyType
from .problem_iterator import MAX_COST, ProblemIterator, Result
from .rule_policy import Policy
from .solver import Solver
from .state_space_generator import State, check_formula, random_walk
from .topk_planner import compute_plans

log = logging.getLogger("genfond.iterative_solver")


def solve(
    domain: Domain,
    problems: Collection[Problem],
    config: Mapping,
    complexity: int,
    max_cost: int,
    max_prune_cost: int,
    all_generators: bool = True,
    enforce_highest_complexity: bool = False,
    selected_states: Optional[dict[str, Collection[State]]] = None,
    plans: Optional[MutableMapping[str, Collection[Plan]]] = None,
) -> Optional[tuple[DatalogPolicy | Policy, dict[str, Any]]]:
    stats: dict[str, Any] = dict()
    log.debug("Generating feature pool ...")
    feature_pool = FeaturePool(
        domain,
        problems,
        config=config,
        max_complexity=complexity,
        all_generators=all_generators,
        selected_states=selected_states,
        plans=plans,
    )
    stats["featurePoolSize"] = len(feature_pool.features)
    log.debug("Generating ASP instance ...")
    asp_instance = feature_pool.to_clingo()
    log.debug(f"ASP instance:\n{asp_instance}")
    if config.get("dump_clingo_program", None):
        with open(config["dump_clingo_program"], "w") as f:
            f.write(asp_instance)
    state_counts = [len(sg.nodes) for sg in feature_pool.state_graphs.values()]
    edge_counts = [len(node.children) for sg in feature_pool.state_graphs.values() for node in sg.nodes.values()]
    stats["numStates"] = sum(state_counts)
    stats["numTransitions"] = sum(edge_counts)
    if max_cost and enforce_highest_complexity and max_cost < complexity:
        log.info(
            f"No solution possible for {pnames(problems)}"
            f" with enforced max complexity {complexity} and max cost {max_cost}"
        )
        return None
    log.info(
        "Solving {} with {} features, {} concepts, {} roles ({}); up to {}complexity {},{}{} and {} = {} states".format(
            ", ".join([p.name for p in problems]),
            len(feature_pool.features),
            len(feature_pool.concepts),
            len(feature_pool.roles),
            "unrestricted" if all_generators else "restricted",
            "enforced " if enforce_highest_complexity else "",
            complexity,
            f" max cost {max_cost}," if max_cost < MAX_COST else "",
            f" max prune cost {max_prune_cost}," if max_prune_cost < MAX_COST else "",
            " + ".join([str(s) for s in state_counts]),
            sum(state_counts),
        )
    )
    solver = Solver(
        asp_instance,
        config["num_threads"],
        max_cost=max_cost,
        max_prune_cost=max_prune_cost,
        min_feature_complexity=complexity if enforce_highest_complexity else None,
        solve_prog=config["solve_prog"],
    )
    if not solver.solve():
        log.info("No solution found")
        return None
    solution = solver.solution
    stats |= {
        "clingoAtoms": solver.statistics["problem"]["lp"]["atoms"],
        "clingoRules": solver.statistics["problem"]["lp"]["rules"],
        "clingoCpuTime": solver.statistics["summary"]["times"]["cpu"],
    }
    log.debug(f"Solution: {solution}")
    log.debug(f'f_selected: {solution.get("f_selected", [])}')
    log.debug(f'f_distinguished: {solution.get("f_distinguished", [])}')
    try:
        policy = generate_policy(solution, policy_type=PolicyType[config["policy_type"]])
    except KeyError as e:
        log.error(f"Error during policy generation: {e}")
        raise
    return policy, stats


def pnames(problems: Collection[Problem]) -> str:
    return ", ".join([p.name for p in problems])


def solve_iteratively(
    domain: Domain, problems: list[Problem], config: Mapping, problem_iterator: Optional[ProblemIterator] = None
) -> tuple[Optional[Policy | DatalogPolicy], list[Problem], dict[str, str | int | float]]:
    policy = None
    problems.sort(key=lambda p: len(p.objects))
    stats: dict[str, str | int | float] = dict()
    if problem_iterator is None:
        problem_iterator = ProblemIterator(problems, config)
    example_plans: dict[str, Collection[Plan]] = dict()
    if config["use_random_walks"]:
        for problem in problems:
            if not any(
                check_formula(state, problem.goal) for state in problem_iterator.selected_states.get(problem.name, [])
            ):
                log.info(f"No goal state in selected states for {problem.name}, starting random walk")
                walk_states = random_walk(
                    domain,
                    problem,
                    problem_iterator.selected_states.get(problem.name, {problem.init}),
                )
                log.info(f"Random walk found {len(walk_states)} states")
                for state in walk_states:
                    problem_iterator.set_new_state(problem.name, state)
                continue
    for iter_kwargs in problem_iterator:
        result, policy = solve_step(
            **iter_kwargs, domain=domain, stats=stats, config=config, example_plans=example_plans
        )
        problem_iterator.set_last_result(result, cost=policy.cost if policy else None)
        if result != Result.SUCCESS:
            continue
        log.info(f'Testing policy on unsolved problems {config["policy_iterations"]} times ...')
        with logging_redirect_tqdm():
            for problem in tqdm.tqdm(problems, disable=None):
                log.info(f'Testing policy on {problem.name} {config["policy_iterations"]} times ...')
                plans = []
                solved = True
                for _ in range(config["policy_iterations"]):
                    try:
                        plan = execute_policy(domain, problem, policy, config)
                        plans.append(plan)
                    except NoActionError as e:
                        log.info(f"Policy does not solve {problem.name}, no action in reachable state")
                        solved = False
                        problem_iterator.set_solved(problem, False)
                        for state in e.trace.keys():
                            problem_iterator.set_new_state(problem.name, state)
                        problem_iterator.set_new_state(problem.name, e.state)
                    except CycleError as e:
                        log.info(f"Policy does not solve {problem.name}, found cycle of length {len(e.cycle)}")
                        solved = False
                        problem_iterator.set_solved(problem, False)
                        for state in e.trace.keys():
                            problem_iterator.set_new_state(problem.name, state)
                    except RuntimeError:
                        log.info("Policy does not solve {}".format(problem.name))
                        solved = False
                        problem_iterator.set_solved(problem, False)
                if solved:
                    plan_lengths = [len(plan) for plan in plans]
                    log.info(
                        f"Policy already solves {problem.name}"
                        f" (plan length {statistics.mean(plan_lengths)} ± {statistics.stdev(plan_lengths):.2f})"
                    )
                    problem_iterator.set_solved(problem)
                else:
                    break
        if solved and config["stop_after_first_solution"]:
            log.info(f"Policy solves all problems")
            break
        else:
            log.error(
                "No policy found for {} with max complexity {}".format(
                    ", ".join([p.name for p in iter_kwargs["active_problems"]]), iter_kwargs["complexity"]
                )
            )
            problem_iterator.set_last_result(Result.NO_SOLUTION)
            stats["failureReason"] = "maxcomplexity"
    stats.update(
        {
            "trainProblems": len(problem_iterator.active_problems),
            "maxTrainProblemSize": (max(len(p.objects) for p in problem_iterator.active_problems) if policy else 0),
        }
    )
    return policy, [p for p in problems if problem_iterator.solved[p.name]], stats


def solve_step(
    domain: Domain,
    config: Mapping,
    stats: MutableMapping[str, Any],
    example_plans: MutableMapping[str, Collection[Plan]],
    active_problems: Collection[Problem],
    complexity: int,
    all_features: bool,
    max_cost: int,
    max_prune_cost: int,
    selected_states: Optional[dict[str, Collection[Any]]],
) -> tuple[Result, Optional[Policy | DatalogPolicy]]:
    for problem in active_problems:
        if config["use_example_plans"] and problem.name not in example_plans:
            num_plans = len(problem.objects)
            # num_plans = config["number_of_plans"]
            log.info("Computing %d example plans for %s ...", num_plans, problem.name)
            example_plans[problem.name] = compute_plans(str(domain), str(problem), number_of_plans=num_plans)

            log.info(
                "Plan lengths for %s: %s",
                problem.name,
                [len(plan.actions) for plan in example_plans[problem.name]],
            )
            log.debug("Plans:\n%s", "\n\n".join([str(plan) for plan in example_plans[problem.name]]))
    try:
        log.info(f"Starting solver for {pnames(active_problems)} with max complexity {complexity}")
        solve_wall_time_start = time.perf_counter()
        solve_cpu_time_start = time.process_time()
        solution = solve(
            domain,
            active_problems,
            config=config,
            complexity=complexity,
            all_generators=all_features,
            max_cost=max_cost,
            max_prune_cost=max_prune_cost,
            enforce_highest_complexity=True,
            selected_states=selected_states,
            plans=example_plans,
        )
    except (RuntimeError, MemoryError) as e:
        log.warning(
            f"Error during policy generation for {pnames(active_problems)} with max complexity {complexity}: {e}"
        )
        if "Id out of range" in str(e):
            stats["failureReason"] = "id"
            return Result.OUT_OF_RESOURCES, None
        elif isinstance(e, MemoryError):
            stats["failureReason"] = "memory"
            return Result.OUT_OF_RESOURCES, None
        else:
            stats["failureReason"] = str(e)
            return Result.UNKNOWN, None
    finally:
        stats["lastSolveWallTime"] = time.perf_counter() - solve_wall_time_start
        stats["lastSolveCpuTime"] = time.process_time() - solve_cpu_time_start
        log.info("Solver wall time: {:.2f}s".format(stats["lastSolveWallTime"]))
        log.info("Solver CPU time: {:.2f}s".format(stats["lastSolveCpuTime"]))
        stats["totalSolveCpuTime"] = stats.get("totalSolveCpuTime", 0) + stats["lastSolveCpuTime"]
    if solution:
        policy, solve_stats = solution
        stats.update(solve_stats)
        log.info(
            f"Found policy with cost {policy.cost} for" f" {pnames(active_problems)} with max complexity {complexity}"
        )
        log.info(f"New policy: {policy}")
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
        stats["maxFeatureComplexity"] = complexity
        stats["bestSolveWallTime"] = stats["lastSolveWallTime"]
        stats["bestSolveCpuTime"] = stats["lastSolveCpuTime"]
        return Result.SUCCESS, policy
    return Result.NO_SOLUTION, None
