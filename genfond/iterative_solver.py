import logging
import pickle
import statistics
import sys
import time
from collections.abc import Iterator
from typing import Any, Callable, Collection, Mapping, MutableMapping, Optional

import tqdm
from pddl.core import Domain, Plan, Problem
from tqdm.contrib.logging import logging_redirect_tqdm

from .datalog_policy import DatalogPolicy
from .execute_datalog_policy import CycleError, NoActionError
from .execute_policy import execute_policy
from .feature_generator import FeaturePool
from .frontier import FrontierState, collect_frontier_states, expand_frontier
from .generate_policy import generate_policy
from .policy import PolicyType
from .problem_iterator import MAX_COST, OneShotProblemIterator, ProblemIterator, Result
from .rule_policy import Policy
from .solver import Solver
from .state_space_generator import State, check_formula
from .state_space_vis import dump_state_graphs

log = logging.getLogger("genfond.iterative_solver")

PlannerComputePlans = Callable[[str, str, dict[str, Any]], Iterator[Plan]]


def _get_example_plan_computer(config: Mapping[str, Any]) -> tuple[PlannerComputePlans, dict[str, Any], str]:
    planner_name = config["planner"]
    planner_config = dict(config["planners"][planner_name])

    match planner_name:
        case "topk_planner":
            from .topk_planner import compute_plans as planner_compute_plans
        case "siw":
            from .siw_planner import compute_plans as planner_compute_plans
        case _:
            raise ValueError(f"Unknown planner '{planner_name}'. Expected one of: topk_planner, siw")

    return planner_compute_plans, planner_config, planner_name


def max_prune_cost(config: Mapping[str, Any], max_cost: int, allow_frontier: bool = True) -> int:
    """How many frontier transitions a single model may use.

    The frontier exists for rounds that are otherwise unsolvable. A tightened `max_cost` means
    a policy already exists and this round is only trying to beat its cost, so the frontier is
    closed off there: expanding it would spend planner calls and state-space growth on shaving
    feature complexity rather than on gaining solvability.

    It is also closed once the expansion budget is spent (`allow_frontier`). A frontier model
    yields no policy, so leaving it available past that point makes every remaining round
    return one and pay for a pointless planner call before escalating anyway.
    """
    if not allow_frontier or max_cost < MAX_COST:
        return 0
    return config["max_frontier_transitions"] or MAX_COST


def solve(
    domain: Domain,
    problems: Collection[Problem],
    config: Mapping,
    complexity: int,
    max_cost: int,
    all_generators: bool = True,
    enforce_highest_complexity: bool = False,
    plans: Optional[MutableMapping[str, Collection[Plan]]] = None,
    dead_states: Optional[Mapping[str, set[State]]] = None,
    allow_frontier: bool = True,
    round_index: int = 0,
) -> Optional[tuple[DatalogPolicy | Policy, dict[str, Any], list[FrontierState]]]:
    stats: dict[str, Any] = dict()
    log.debug("Generating feature pool ...")
    feature_pool = FeaturePool(
        domain,
        problems,
        config=config,
        max_complexity=complexity,
        all_generators=all_generators,
        plans=plans,
        dead_states=dead_states,
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
        "Solving {}{} with {} features, {} concepts, {} roles ({}); up to {}complexity {},{} and {} = {} states".format(
            ", ".join([p.name for p in problems]),
            ((" (" + ", ".join([str(len(plans[problem.name])) for problem in problems]) + ")") if plans else ""),
            len(feature_pool.features),
            len(feature_pool.concepts),
            len(feature_pool.roles),
            "unrestricted" if all_generators else "restricted",
            "enforced " if enforce_highest_complexity else "",
            complexity,
            f" max cost {max_cost}," if max_cost < MAX_COST else "",
            " + ".join([str(s) for s in state_counts]),
            sum(state_counts),
        )
    )
    solver = Solver(
        asp_instance,
        config["num_threads"],
        max_cost=max_cost,
        max_prune_cost=max_prune_cost(config, max_cost, allow_frontier),
        min_feature_complexity=complexity if enforce_highest_complexity else None,
        solve_prog=config["solve_prog"],
    )
    satisfiable = solver.solve()
    solution = solver.solution if satisfiable else None
    if config["state_graph_dir"]:
        dump_state_graphs(
            feature_pool.state_graphs,
            feature_pool.problem_id_to_name,
            solution,
            config,
            round_index=round_index,
            complexity=complexity,
        )
    if not satisfiable:
        log.info("No solution found")
        return None
    assert solution is not None
    if "viol" in solution:
        log.warning(f"Found violations: {solution['viol']}")
    stats |= {
        "clingoAtoms": solver.statistics["problem"]["lp"]["atoms"],
        "clingoRules": solver.statistics["problem"]["lp"]["rules"],
        "clingoCpuTime": solver.statistics["summary"]["times"]["cpu"],
    }
    log.debug(f"Solution: {solution}")
    log.debug(f'f_selected: {solution.get("f_selected", [])}')
    log.debug(f'f_distinguished: {solution.get("f_distinguished", [])}')
    frontier_states = collect_frontier_states(feature_pool, solution)
    stats["numFrontierTransitions"] = len(frontier_states)
    try:
        policy = generate_policy(solution, policy_type=PolicyType[config["policy_type"]])
    except KeyError as e:
        log.error(f"Error during policy generation: {e}")
        raise
    return policy, stats, frontier_states


def pnames(problems: Collection[Problem]) -> str:
    return ", ".join([p.name for p in problems])


def solve_iteratively(
    domain: Domain, problems: list[Problem], config: Mapping, one_shot: bool = False
) -> tuple[Optional[Policy | DatalogPolicy], list[Problem], dict[str, str | int | float]]:
    policy = None
    problems.sort(key=lambda p: len(p.objects))
    stats: dict[str, str | int | float] = dict()
    example_plans: dict[str, Iterator[Plan]] = dict()
    planner_compute_plans: Optional[PlannerComputePlans] = None
    planner_config: dict[str, Any] = dict()
    if config["use_example_plans"]:
        planner_compute_plans, planner_config, planner_name = _get_example_plan_computer(config)
        log.info(f"Using planner '{planner_name}' to generate example plans")
        for problem in problems:
            example_plans[problem.name] = planner_compute_plans(
                str(domain),
                str(problem),
                dict(planner_config),
            )
    if config["frontier_expansion"] and not config["use_example_plans"]:
        # StateSpaceGraph only leaves states unexpanded when it is restricted by example
        # plans, so without them the flag is inert: no state is ever marked PRUNED.
        log.warning("frontier_expansion has no effect without use_example_plans")
    problem_iterator: ProblemIterator | OneShotProblemIterator
    if one_shot:
        problem_iterator = OneShotProblemIterator(problems, config, plans=example_plans)
    else:
        problem_iterator = ProblemIterator(problems, config, plans=example_plans)
    problems_by_name = {problem.name: problem for problem in problems}
    for round_index, iter_kwargs in enumerate(problem_iterator, start=1):
        result, new_policy, frontier_states = solve_step(
            **iter_kwargs,
            domain=domain,
            stats=stats,
            config=config,
            enforce_highest_complexity=not one_shot,
            round_index=round_index,
        )
        if result == Result.FRONTIER:
            # Expand the unexpanded states the model relied on, then retry the same
            # configuration. Must not fall through: the model is not a valid policy.
            assert planner_compute_plans, "A frontier can only arise from plan-restricted expansion"
            new_plans, dead_states = expand_frontier(
                domain,
                problems_by_name,
                frontier_states,
                planner_compute_plans,
                planner_config,
                config,
            )
            problem_iterator.record_frontier_expansion(new_plans, dead_states)
            problem_iterator.set_last_result(result)
            continue
        problem_iterator.set_last_result(result, cost=new_policy.cost if new_policy else None)
        if result != Result.SUCCESS:
            continue
        policy = new_policy
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
                    except CycleError as e:
                        log.info(f"Policy does not solve {problem.name}, found cycle of length {len(e.cycle)}")
                        solved = False
                        problem_iterator.set_solved(problem, False)
                    except RuntimeError:
                        log.info("Policy does not solve {}".format(problem.name))
                        solved = False
                        problem_iterator.set_solved(problem, False)
                if solved:
                    plan_lengths = [len(plan) for plan in plans]
                    log.info(
                        f"Policy already solves {problem.name} (plan length "
                        f"{statistics.mean(plan_lengths)} ± {statistics.stdev(plan_lengths):.2f})"
                        if len(plan_lengths) > 1
                        else f"{plan_lengths[0]}"
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
    enforce_highest_complexity: bool,
    all_features: bool,
    max_cost: int,
    dead_states: Optional[Mapping[str, set[State]]] = None,
    allow_frontier: bool = True,
    round_index: int = 0,
) -> tuple[Result, Optional[Policy | DatalogPolicy], list[FrontierState]]:
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
            enforce_highest_complexity=enforce_highest_complexity,
            plans=example_plans,
            dead_states=dead_states,
            allow_frontier=allow_frontier,
            round_index=round_index,
        )
    except (RuntimeError, MemoryError) as e:
        log.warning(
            f"Error during policy generation for {pnames(active_problems)} with max complexity {complexity}: {e}"
        )
        if "Id out of range" in str(e):
            stats["failureReason"] = "id"
            return Result.OUT_OF_RESOURCES, None, []
        elif isinstance(e, MemoryError):
            stats["failureReason"] = "memory"
            return Result.OUT_OF_RESOURCES, None, []
        else:
            stats["failureReason"] = str(e)
            return Result.UNKNOWN, None, []
    finally:
        stats["lastSolveWallTime"] = time.perf_counter() - solve_wall_time_start
        stats["lastSolveCpuTime"] = time.process_time() - solve_cpu_time_start
        log.info("Solver wall time: {:.2f}s".format(stats["lastSolveWallTime"]))
        log.info("Solver CPU time: {:.2f}s".format(stats["lastSolveCpuTime"]))
        stats["totalSolveCpuTime"] = stats.get("totalSolveCpuTime", 0) + stats["lastSolveCpuTime"]
    if solution:
        policy, solve_stats, frontier_states = solution
        stats.update(solve_stats)
        if frontier_states:
            # Not a policy: it assumes the frontier states it selects are solvable. The caller
            # must expand them and re-solve; only a zero-frontier model is acceptable.
            log.info(
                f"Model for {pnames(active_problems)} with max complexity {complexity} relies on"
                f" {len(frontier_states)} transition(s) into unexpanded states, expanding them"
            )
            return Result.FRONTIER, None, frontier_states
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
        return Result.SUCCESS, policy, []
    return Result.NO_SOLUTION, None, []
