import logging
import pickle
import statistics
import sys
import time
import traceback
from typing import Any, Collection, Mapping, Optional

import tqdm
from pddl.core import Domain, Problem
from tqdm.contrib.logging import logging_redirect_tqdm

from .datalog_policy import DatalogPolicy
from .execute_datalog_policy import CycleError, NoActionError
from .execute_policy import execute_policy_wl
from .feature_generator_wlplan import WlPlanFeaturePool
from .generate_policy import generate_policy
from .policy import PolicyType
from .problem_iterator import MAX_COST, ProblemIterator, Result
from .rule_policy import Policy
from .solver import Solver
from .state_space_generator import State, check_formula, random_walk

log = logging.getLogger("genfond.iterative_solver_wlplan")


def solve_wl(
    domain: Domain,
    problems: Collection[Problem],
    config: Mapping,
    complexity: int,
    max_cost: int,
    max_prune_cost: int,
    selected_states: Optional[dict[str, set[State]]] = None,
) -> Optional[tuple[DatalogPolicy | Policy, dict[str, Any]]]:
    stats: dict[str, Any] = dict()
    log.debug("Generating feature pool ...")
    feature_pool = WlPlanFeaturePool(
        domain,
        problems,
        config=config,
        max_complexity=complexity,
        selected_states=selected_states,
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
    log.info(f"Solving {', '.join([p.name for p in problems])} with {len(feature_pool.features)=} {max_cost=} {max_prune_cost=}, {complexity=}, {sum(state_counts)=}")
    solver = Solver(
        asp_instance,
        config["num_threads"],
        max_cost=max_cost,
        max_prune_cost=max_prune_cost,
        min_feature_complexity=None,
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
        policy = generate_policy(solution, policy_type=PolicyType[config["policy_type"]], save_file=feature_pool.get_save_file())
    except KeyError as e:
        log.error(f"Error during policy generation: {e}")
        raise
    return policy, stats


def pnames(problems: Collection[Problem]) -> str:
    return ", ".join([p.name for p in problems])


def solve_iteratively_wl(
    domain: Domain, problems: list[Problem], config: Mapping
) -> tuple[Optional[Policy | DatalogPolicy], list[Problem], dict[str, str | int | float]]:
    policy = None
    problems.sort(key=lambda p: len(p.objects))
    total_solve_cpu_time = 0.0
    best_solve_cpu_time = 0.0
    best_solve_wall_time = 0.0
    stats: dict[str, str | int | float] = dict()
    problem_iterator = ProblemIterator(problems, config)
    for (
        solver_problems,
        i,
        all_generators,
        max_cost,
        max_prune_cost,
        selected_states,
    ) in problem_iterator:
        try:
            log.info(f"Starting solver for {pnames(solver_problems)} with max complexity {i}")
            solve_wall_time_start = time.perf_counter()
            solve_cpu_time_start = time.process_time()
            solution = solve_wl(
                domain,
                solver_problems,
                config=config,
                complexity=i,
                max_cost=max_cost,
                max_prune_cost=max_prune_cost,
                selected_states=selected_states,
            )
        except (RuntimeError, MemoryError) as e:
            if "Id out of range" in str(e):
                stats["failureReason"] = "id"
                problem_iterator.set_last_result(Result.OUT_OF_RESOURCES)

            elif isinstance(e, MemoryError):
                stats["failureReason"] = "memory"
                problem_iterator.set_last_result(Result.OUT_OF_RESOURCES)

            else:
                problem_iterator.set_last_result(Result.UNKNOWN)
                stats["failureReason"] = str(e)
            log.warning(f"Error during policy generation for {pnames(solver_problems)} with max complexity {i}: {e}")
            continue
        finally:
            solve_wall_time = time.perf_counter() - solve_wall_time_start
            solve_cpu_time = time.process_time() - solve_cpu_time_start
            log.info("Solver wall time: {:.2f}s".format(solve_wall_time))
            log.info("Solver CPU time: {:.2f}s".format(solve_cpu_time))
            total_solve_cpu_time += solve_cpu_time
        if solution:
            new_policy, solve_stats = solution
            log.info(
                f"Found policy with cost {new_policy.cost} for" f" {pnames(solver_problems)} with max complexity {i}"
            )
            log.info(f"New policy: {new_policy}")
            policy = new_policy
            stats = solve_stats
            stats["maxFeatureComplexity"] = i
            best_solve_wall_time = solve_wall_time
            best_solve_cpu_time = solve_cpu_time
            problem_iterator.set_last_result(Result.SUCCESS, cost=new_policy.cost)
            policy = new_policy
            log.info(f'Testing policy on unsolved problems {config["policy_iterations"]} times ...')
            with logging_redirect_tqdm():
                for problem in tqdm.tqdm(problems, disable=None):
                    log.info(f'Testing policy on {problem.name} {config["policy_iterations"]} times ...')
                    plans = []
                    solved = True
                    for _ in range(config["policy_iterations"]):
                        try:
                            plan = execute_policy_wl(domain, problem, policy, config)
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
                        except NotImplementedError as e:
                            traceback.print_exc()
                            raise e
                        except RuntimeError:
                            log.info("Policy does not solve {}".format(problem.name))
                            solved = False
                            problem_iterator.set_solved(problem, False)
                    if solved:
                        plan_lengths = [len(plan) for plan in plans]
                        log.info(
                            f"Policy already solves {problem.name}"
                            f" (plan length {statistics.mean(plan_lengths)} ± {statistics.stdev(plan_lengths)})"
                        )
                        problem_iterator.set_solved(problem)
                    else:
                        break
            if solved and config["stop_after_first_solution"]:
                log.info(f"Policy solves all problems")
                break
        else:
            log.error(
                "No policy found for {} with max complexity {}".format(", ".join([p.name for p in solver_problems]), i)
            )
            problem_iterator.set_last_result(Result.NO_SOLUTION)
            stats["failureReason"] = "maxcomplexity"
    stats.update(
        {
            "trainProblems": len(problem_iterator.active_problems),
            "maxTrainProblemSize": (max(len(p.objects) for p in problem_iterator.active_problems) if policy else 0),
            "bestSolveCpuTime": best_solve_cpu_time,
            "bestSolveWallTime": best_solve_wall_time,
            "totalSolveCpuTime": total_solve_cpu_time,
        }
    )
    return policy, [p for p in problems if problem_iterator.solved[p.name]], stats
