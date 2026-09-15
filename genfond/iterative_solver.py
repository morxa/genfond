import gc
import logging
import pickle
import resource
import statistics
import sys
import time
from collections.abc import Iterator
from typing import Any, Callable, Collection, Mapping, MutableMapping, Optional

import tqdm
from pddl.core import Domain, Plan, Problem
from tqdm.contrib.logging import logging_redirect_tqdm

from .cost_utils import feature_cost
from .datalog_policy import DatalogPolicy
from .execute_datalog_policy import CycleError, NoActionError
from .execute_policy import execute_policy
from .feature_generator import FeaturePool
from .frontier import FrontierState, collect_frontier_states, expand_frontier
from .generate_policy import generate_policy
from .lazy_pairs import DEFAULT_BATCH, solve_with_lazy_pairs
from .policy import PolicyType
from .problem_iterator import MAX_COST, OneShotProblemIterator, ProblemIterator, Result
from .rule_policy import Policy
from .solver import Solver, SolveStatus
from .state_space_generator import State, check_formula

log = logging.getLogger("genfond.iterative_solver")


def _read_proc_status_kb(*fields: str) -> dict[str, Optional[int]]:
    """Read the given `VmXxx:` fields (in kB) from /proc/self/status.

    Returns None for a field that could not be read (e.g. on a non-Linux platform), so callers
    never have to special-case the whole call failing.
    """
    result: dict[str, Optional[int]] = {field: None for field in fields}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                key, _, rest = line.partition(":")
                if key in result:
                    result[key] = int(rest.split()[0])
    except OSError:
        pass
    return result


def log_memory(tag: str) -> None:
    """Log the process's memory footprint, for tracking retention across solve rounds.

    `ru_maxrss` is a high-water mark (never decreases within the process); `VmRSS` is the
    current resident set; `VmSize` is the current virtual address space, which is what
    `RLIMIT_AS` (`--max-memory`) actually caps -- it can stay high even after `VmRSS` drops,
    which is the signature of fragmentation rather than a live Python reference holding memory.
    """
    maxrss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    proc = _read_proc_status_kb("VmSize", "VmRSS")
    vmsize_mb = proc["VmSize"] / 1024 if proc["VmSize"] is not None else None
    vmrss_mb = proc["VmRSS"] / 1024 if proc["VmRSS"] is not None else None
    log.info(
        "Memory[%s]: maxrss=%.1fMB rss=%s vmsize=%s",
        tag,
        maxrss_mb,
        f"{vmrss_mb:.1f}MB" if vmrss_mb is not None else "?",
        f"{vmsize_mb:.1f}MB" if vmsize_mb is not None else "?",
    )


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
    stats: Optional[MutableMapping[str, Any]] = None,
) -> Optional[tuple[DatalogPolicy | Policy, MutableMapping[str, Any], list[FrontierState]]]:
    # The caller may hand in the run-level stats dict so that the keys describing *why* a round
    # produced no policy (solveStatus, solveOptimal) survive a `return None`; the returned
    # mapping is then that same object and `stats.update(...)` on it is a no-op.
    if stats is None:
        stats = dict()
    # Defaults for the paths that return before a solve happens at all. `None` is deliberately
    # not a SolveStatus name: those paths keep the pre-existing NO_SOLUTION handling.
    stats["solveStatus"] = None
    stats["solveOptimal"] = True
    if config.get("minimize_good_signatures", "none") != "none" and config["solve_prog"] != "solve_datalog_sig.lp":
        # good_sig/1 (and the two #program parts Solver grounds for it) only exist in
        # solve_datalog_sig.lp; grounding them against another solve_prog would fail inside
        # clingo with a much less legible error.
        raise ValueError(
            f"minimize_good_signatures={config['minimize_good_signatures']!r} needs solve_prog="
            f"'solve_datalog_sig.lp' (--type datalog-sig), got {config['solve_prog']!r}"
        )
    if config.get("minimize_selected_count", "none") != "none" and config["solve_prog"] != "solve_datalog_sig.lp":
        # The minimize_selected_count_{below,above} #program parts (see solve_datalog_sig.lp)
        # only exist there too, for the same reason.
        raise ValueError(
            f"minimize_selected_count={config['minimize_selected_count']!r} needs solve_prog="
            f"'solve_datalog_sig.lp' (--type datalog-sig), got {config['solve_prog']!r}"
        )
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
    # The full instance is available via dump_clingo_program; logging it made verbose run logs
    # grow to gigabytes (2.4 M dist/2 facts per round on blocks3ops). The summary line stays at
    # INFO (not DEBUG) so the effect of emit_object_facts on instance size is visible by default.
    log.info(f"ASP instance: {asp_instance.count(chr(10))} lines, {len(asp_instance) / 1e6:.1f} MB")
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
        opt_strategy=config["clingo_opt_strategy"],
        clingo_options=config["clingo_options"],
        time_limit=config["solve_time_limit"],
        minimize_good_signatures=config.get("minimize_good_signatures", "none"),
        minimize_selected_count=config.get("minimize_selected_count", "none"),
    )
    if config.get("lazy_pairs", False) and config.get("emit_action_signatures", False):
        # The instance carries no separation pairs; they are added batch by batch in response to
        # the models that violate them. The loop wraps the solve of this one round only, so
        # max_cost, the min_feature_complexity program and the frontier machinery are untouched.
        status = solve_with_lazy_pairs(
            solver, feature_pool.signatures, config.get("lazy_pairs_batch") or DEFAULT_BATCH, stats
        )
    else:
        solver.solve()
        status = solver.status
    stats["solveStatus"] = status.name
    # Whether the round *proved* its answer. A model whose cost was never proved optimal must
    # not be used to refute a complexity level; the iterator is told via
    # set_last_result(..., optimal=...).
    stats["solveOptimal"] = status in (SolveStatus.OPTIMAL, SolveStatus.UNSATISFIABLE)
    if not stats["solveOptimal"]:
        stats["nonOptimalRounds"] = stats.get("nonOptimalRounds", 0) + 1
    if status in (SolveStatus.UNSATISFIABLE, SolveStatus.UNKNOWN):
        log.info("No solution found" if status == SolveStatus.UNSATISFIABLE else "Solve was inconclusive (timed out)")
        return None
    solution = solver.solution
    if "viol" in solution:
        log.warning(f"Found violations: {solution['viol']}")
    # The quantity minimize_selected_count trades off against feature complexity. Logged at INFO
    # (not DEBUG), like the good-signature/rule count in generate_datalog_policy, so the effect
    # of that setting on a run is visible by default regardless of whether it is active.
    num_selected = (
        len(solution.get("c_selected", [])) + len(solution.get("f_selected", [])) + len(solution.get("r_selected", []))
    )
    stats["numSelectedElements"] = num_selected
    log.info(
        f"Selected {num_selected} element(s): {len(solution.get('c_selected', []))} concept(s), "
        f"{len(solution.get('f_selected', []))} feature(s), {len(solution.get('r_selected', []))} role(s)"
    )
    stats.update(
        {
            "clingoAtoms": solver.statistics["problem"]["lp"]["atoms"],
            "clingoRules": solver.statistics["problem"]["lp"]["rules"],
            "clingoCpuTime": solver.statistics["summary"]["times"]["cpu"],
        }
    )
    log.debug(f"Solution: {solution}")
    log.debug(f'f_selected: {solution.get("f_selected", [])}')
    log.debug(f'f_distinguished: {solution.get("f_distinguished", [])}')
    frontier_states = collect_frontier_states(feature_pool, solution)
    stats["numFrontierTransitions"] = len(frontier_states)
    try:
        policy = generate_policy(
            solution, policy_type=PolicyType[config["policy_type"]], signatures=feature_pool.signatures
        )
    except KeyError as e:
        log.error(f"Error during policy generation: {e}")
        raise
    return policy, stats, frontier_states


def pnames(problems: Collection[Problem]) -> str:
    return ", ".join([p.name for p in problems])


def _test_policy_on_problems(
    domain: Domain, problems: Collection[Problem], policy: Policy | DatalogPolicy, config: Mapping
) -> list[Problem]:
    """Execute `policy` on every problem in `problems`, `policy_iterations` times each.

    Unlike the main loop's testing block (which stops at the first problem it cannot solve,
    since `problems` is sorted smallest-first and a miss there usually means every larger
    problem misses too), this tests every one of them: `_final_cost_minimization_pass` compares
    candidate policies by how many problems they solve, so silently undercounting a later
    problem because an earlier one already failed would bias that comparison.
    """
    solved_problems = []
    for problem in problems:
        solved = True
        for _ in range(config["policy_iterations"]):
            try:
                execute_policy(domain, problem, policy, config)
            except (NoActionError, CycleError, RuntimeError):
                solved = False
                break
        if solved:
            solved_problems.append(problem)
    return solved_problems


def _final_cost_minimization_pass(
    domain: Domain,
    problems: list[Problem],
    problem_iterator: ProblemIterator,
    config: Mapping,
    stats: MutableMapping[str, Any],
    policy: Policy | DatalogPolicy,
) -> tuple[Policy | DatalogPolicy, list[Problem]]:
    """Look for a cheaper policy on the final training set, once the main loop has nothing left
    to add.

    `add_problem_after_success` (see its config comment and docs/no-cost-climb-results.md) skips
    the complexity climb after every success so the loop grows the training set instead. When a
    chain of successes solves every problem in a row -- which is exactly when the main loop's
    `stop_after_first_solution` break fires -- that climb never runs at all, and the final policy
    ends up priced at whatever complexity first solved the final training set rather than the
    cheapest complexity that does. A cheaper policy has been observed to generalize further (the
    c-base vs. c-combo rows in docs/experiments-log.md and their "Caution on reading these two
    rows" paragraph): the base arm's cost-10 policy from 9 training problems out-generalized the
    combo arm's cost-21 policy from 19.

    This runs that climb exactly once more, standalone, on the frozen final training set
    (`problem_iterator.active_problems`, with its final `active_plans`/`dead_states`): reset to
    `succ_complexity`, tighten `max_cost` to `cost - 1`, and increment complexity while
    `max_cost > complexity` and `complexity < max_complexity` -- the same condition as
    `ProblemIterator`'s `INC_COMPLEXITY` branch, run here on its own instead of interleaved with
    plan/problem escalation (`enforce_highest_complexity` and the frontier are both left off:
    this pass does not track refutations the way the main loop does, and the training set is
    already frozen, so there is nothing left for the frontier to expand). A round's ASP-level
    `Result.SUCCESS` only proves the model satisfies the constraints, the same way the main
    loop's own testing block treats it -- a training problem can still fail *execution* (e.g. a
    cycle) -- so a candidate is only kept if it still solves every training problem, and only
    when it beats the current best by (most problems solved overall, then lowest cost).
    `Result.OUT_OF_RESOURCES`/`TIMEOUT` end the pass gracefully with the best policy found so
    far, exactly like they end the main loop's own climb.
    """
    active_problems = problem_iterator.active_problems
    example_plans = problem_iterator.active_plans
    dead_states = problem_iterator.dead_states
    complexity = problem_iterator.succ_complexity
    active_problem_names = {p.name for p in active_problems}

    assert policy.cost is not None, "the main loop only sets `policy` from a Result.SUCCESS, which always has a cost"
    best_policy = policy
    best_solved = _test_policy_on_problems(domain, problems, policy, config)
    best_cost = feature_cost(
        policy.cost, config.get("minimize_good_signatures", "none"), config.get("minimize_selected_count", "none")
    )
    stats["finalPassCostBefore"] = best_cost
    max_cost = best_cost - 1
    rounds = 0
    log.info(f"Starting final cost minimization pass from complexity {complexity}, max cost {max_cost}")
    while max_cost > complexity and complexity < config["max_complexity"]:
        complexity += 1
        rounds += 1
        result, new_policy, _frontier_states = solve_step(
            domain=domain,
            config=config,
            stats=stats,
            example_plans=example_plans,
            active_problems=active_problems,
            complexity=complexity,
            all_features=config["use_unrestricted_features"],
            max_cost=max_cost,
            enforce_highest_complexity=False,
            dead_states=dead_states,
            allow_frontier=False,
        )
        if result in (Result.OUT_OF_RESOURCES, Result.TIMEOUT):
            break
        if result != Result.SUCCESS:
            # Nothing at this complexity beats max_cost; keep climbing, exactly like the
            # INC_COMPLEXITY branch, which does not stop on NO_SOLUTION either.
            continue
        assert new_policy is not None, "solve_step must return a policy on Result.SUCCESS"
        assert new_policy.cost is not None, "a Result.SUCCESS policy always has a cost"
        new_cost = feature_cost(
            new_policy.cost,
            config.get("minimize_good_signatures", "none"),
            config.get("minimize_selected_count", "none"),
        )
        # Tighten unconditionally on ASP-level success, before the execution check below --
        # this mirrors set_last_result, which does the same regardless of whether the main
        # loop's later testing block goes on to find an execution failure.
        max_cost = new_cost - 1
        solved = _test_policy_on_problems(domain, problems, new_policy, config)
        solved_names = {p.name for p in solved}
        if not (active_problem_names <= solved_names):
            log.info(f"Candidate policy at complexity {complexity} does not solve every training problem, discarding")
            continue
        if (len(solved), -new_cost) > (len(best_solved), -best_cost):
            best_policy, best_solved, best_cost = new_policy, solved, new_cost
    stats["finalPassRounds"] = rounds
    stats["finalPassCostAfter"] = best_cost
    log.info(
        f"Final cost minimization pass: {rounds} round(s), cost {stats['finalPassCostBefore']} ->"
        f" {best_cost}, solves {len(best_solved)}/{len(problems)} problems"
    )
    return best_policy, best_solved


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
    for iter_kwargs in problem_iterator:
        result, new_policy, frontier_states = solve_step(
            **iter_kwargs,
            domain=domain,
            stats=stats,
            config=config,
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
        problem_iterator.set_last_result(
            result,
            cost=new_policy.cost if new_policy else None,
            optimal=bool(stats.get("solveOptimal", True)),
        )
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
    solved_problems = [p for p in problems if problem_iterator.solved[p.name]]
    if config.get("final_cost_minimization", False) and config["add_problem_after_success"] and policy is not None:
        # Only meaningful together with add_problem_after_success: without it, the normal
        # ladder already climbs complexity whenever a round is not an immediate success, so
        # there is no gap for this pass to fill (see the config comment on
        # final_cost_minimization).
        policy, solved_problems = _final_cost_minimization_pass(
            domain, problems, problem_iterator, config, stats, policy
        )
    return policy, solved_problems, stats


def _release_round_memory() -> None:
    """Force prompt release of a round's memory instead of waiting on GC/allocator heuristics.

    Each round builds a `FeaturePool` (dlplan-backed state graphs and generated features) and a
    `Solver` (a clingo `Control`, which owns the grounded program and the ASP instance string)
    as locals of `solve()`. Both contain back-references -- state graph nodes point at their
    graph and vice versa, dlplan elements are cached by their factory -- so they form reference
    cycles that plain refcounting cannot free; they sit until Python's generational GC gets
    around to them. That GC is scheduled by allocation *count*, not size, so a handful of huge,
    cycle-holding rounds can go uncollected for a long time, and a `bad_alloc` (from clingo or,
    via pybind11's automatic `std::bad_alloc` -> `MemoryError` mapping, from dlplan) makes it
    worse: the exception's traceback keeps every frame between the raise and the `except` that
    caught it alive -- including `solve()`'s `feature_pool`, `asp_instance` and `solver` locals
    -- until the exception itself is collected. `except ... as e:` already deletes `e` (and so
    the traceback) when its suite ends, which is what turns that chain into a self-contained
    cycle with no external referrer; only a GC pass, not refcounting, reclaims a cycle like that.

    Measured on the repro in `docs/memory-release-results.md`: `gc.collect()` alone found
    thousands of unreachable objects and returned tens of MB of `VmSize` per round -- the
    quantity `RLIMIT_AS`/`--max-memory` actually caps, and so the one that determines whether
    the next round's allocation fits. `malloc_trim(0)` is a second, smaller lever on top: it
    returns freed heap pages to the OS, which helps `RSS` a lot (relevant since other jobs share
    this machine) but barely moved `VmSize` in that measurement, so it is not a substitute for
    the `gc.collect()` above.
    """
    collected = gc.collect()
    log.debug("gc.collect() freed %d unreachable object(s)", collected)
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        # No libc.so.6 (e.g. non-Linux): RSS just stays higher afterwards, no correctness issue.
        log.debug("malloc_trim(0) unavailable on this platform")


def solve_step(
    domain: Domain,
    config: Mapping,
    stats: MutableMapping[str, Any],
    example_plans: MutableMapping[str, Collection[Plan]],
    active_problems: Collection[Problem],
    complexity: int,
    all_features: bool,
    max_cost: int,
    # Whether `complexity - 1` has been refuted for this exact state space; the iterator owns
    # that bookkeeping because only it knows when the state space last changed.
    enforce_highest_complexity: bool = False,
    dead_states: Optional[Mapping[str, set[State]]] = None,
    allow_frontier: bool = True,
) -> tuple[Result, Optional[Policy | DatalogPolicy], list[FrontierState]]:
    log_memory(f"round start complexity={complexity}")
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
            stats=stats,
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
        log_memory(f"round end complexity={complexity}")
        # By this point `except ... as e:` has already deleted `e`, so `solve()`'s locals
        # (`feature_pool`, `asp_instance`, `solver`/`Control`) are unreachable except through
        # whatever reference cycles they formed among themselves -- exactly what `gc.collect()`
        # is for. Runs on every path (success, no-solution, frontier, and error alike): ordinary
        # rounds hold cyclic state-graph structures too, not just failed ones.
        _release_round_memory()
        log_memory(f"round end complexity={complexity} (after release)")
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
        stats["bestSolveOptimal"] = stats.get("solveOptimal", True)
        return Result.SUCCESS, policy, []
    if stats.get("solveStatus") == SolveStatus.UNKNOWN.name:
        # The solve ran out of its wall-clock budget without a model. That is not a refutation:
        # reporting NO_SOLUTION would let the iterator record the complexity level as refuted
        # and enable min_feature_complexity for every later round on this state space.
        log.info(
            f"Solve for {pnames(active_problems)} with max complexity {complexity} hit its time"
            " budget without a model; the round refutes nothing"
        )
        return Result.TIMEOUT, None, []
    return Result.NO_SOLUTION, None, []
