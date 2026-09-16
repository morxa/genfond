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
from .execute_datalog_policy import CycleError, ExecutionTimeout, NoActionError
from .execute_policy import execute_policy
from .feature_generator import FeaturePool
from .frontier import FrontierState, collect_frontier_states, expand_frontier
from .generate_policy import generate_policy
from .lazy_pairs import DEFAULT_BATCH, solve_with_lazy_pairs
from .policy import PolicyType
from .problem_iterator import MAX_COST, OneShotProblemIterator, PlanStateCoverage, ProblemIterator, Result
from .rule_policy import Policy
from .shutdown import stop_requested
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
    wall_deadline: Optional[float] = None,
    max_pool_size: Optional[int] = None,
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
    for key in ("fix_forced_labels", "plan_label_heuristic"):
        # forced_good/3, forced_bad/3, plan_action/3 and the plan_heuristic #program part only
        # exist in solve_datalog_sig.lp; asking for them elsewhere would either be silently
        # ignored (the facts) or fail inside clingo (the part).
        if config.get(key, False) and config["solve_prog"] != "solve_datalog_sig.lp":
            raise ValueError(
                f"{key}=True needs solve_prog='solve_datalog_sig.lp' (--type datalog-sig),"
                f" got {config['solve_prog']!r}"
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
    # `max_pool_size` (only ever passed by `_final_cost_minimization_pass`, from
    # `final_pass_max_pool`) bails out before grounding at all -- concepts/roles/features are
    # already known at this point, cheaply, from building `feature_pool`; it's grounding the ASP
    # instance from them (`to_clingo()` below, then `Solver.solve()`) that is what actually blows
    # up memory on a large pool. A skip here is reported like any other no-solution round (see
    # `solve_step`), so the pass's climb just keeps going -- capped independently by
    # `final_pass_max_levels`.
    pool_size = len(feature_pool.features) + len(feature_pool.concepts) + len(feature_pool.roles)
    if max_pool_size is not None and pool_size > max_pool_size:
        log.warning(
            "Pool size %d (features+concepts+roles) at complexity %d exceeds max_pool_size=%d;"
            " skipping this round without grounding",
            pool_size,
            complexity,
            max_pool_size,
        )
        stats["poolSizeSkipped"] = pool_size
        return None
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
    # Per-problem plan/state counts, so a runaway problem (one accumulating far more plans or
    # states than the rest, e.g. via frontier expansion) is visible in the log without having to
    # reconstruct it from individual round summaries after the fact.
    if plans:
        plan_counts = {problem.name: len(plans.get(problem.name, [])) for problem in problems}
        if plan_counts:
            stats["maxPlansPerProblem"] = max(plan_counts.values())
            stats["meanPlansPerProblem"] = statistics.mean(plan_counts.values())
            log.info(
                "Plans per problem: max=%d mean=%.1f (%s)",
                stats["maxPlansPerProblem"],
                stats["meanPlansPerProblem"],
                ", ".join(f"{name}={count}" for name, count in plan_counts.items()),
            )
    if state_counts:
        state_counts_by_name = {name: len(sg.nodes) for name, sg in feature_pool.state_graphs.items()}
        stats["maxStatesPerProblem"] = max(state_counts_by_name.values())
        stats["meanStatesPerProblem"] = statistics.mean(state_counts_by_name.values())
        log.info(
            "States per problem: max=%d mean=%.1f (%s)",
            stats["maxStatesPerProblem"],
            stats["meanStatesPerProblem"],
            ", ".join(f"{name}={count}" for name, count in state_counts_by_name.items()),
        )
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
        wall_deadline=wall_deadline,
        plan_label_heuristic=config.get("plan_label_heuristic", False),
    )
    forced = feature_pool.forced_labels
    if forced is not None:
        stats["forcedGoodActions"] = len(forced.good)
        stats["forcedBadActions"] = len(forced.bad)
        stats["forcedGoodSignatures"] = len(forced.good_signatures)
        stats["forcedBadSignatures"] = len(forced.bad_signatures)
        stats["forcedOccurrences"] = forced.num_occurrences
        stats["forcedInconsistent"] = forced.inconsistent
    if config.get("lazy_pairs", False) and config.get("emit_action_signatures", False):
        # The instance carries no separation pairs; they are added batch by batch in response to
        # the models that violate them. The loop wraps the solve of this one round only, so
        # max_cost, the min_feature_complexity program and the frontier machinery are untouched.
        status = solve_with_lazy_pairs(
            solver,
            feature_pool.signatures,
            config.get("lazy_pairs_batch") or DEFAULT_BATCH,
            stats,
            forced=forced,
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
    domain: Domain,
    problems: Collection[Problem],
    policy: Policy | DatalogPolicy,
    config: Mapping,
    max_consecutive_failures: Optional[int] = None,
    problem_iterator: Optional[ProblemIterator] = None,
    stats: Optional[MutableMapping[str, Any]] = None,
    show_progress: bool = False,
    wall_deadline: Optional[float] = None,
) -> list[Problem]:
    """Execute `policy` on problems in `problems`, `validation_iterations` times each (capped at
    `validation_time_limit` seconds per execution) -- the cheap, in-loop validation used both by
    `solve_iteratively`'s own testing block after every success and by
    `_final_cost_minimization_pass`'s candidate comparisons. Unlike the one-time final
    verification loop in `__main__`, which always tests every problem `policy_iterations` times
    with no time limit, this is meant to be run over and over on a large suite -- see
    docs/cheap-validation-results.md for the 8.5h an unbounded version of this spent testing
    candidates on a 141-problem suite.

    `problems` is tested in whatever order it is given, with no reordering here: both call sites
    pass `problem_iterator.active_problems`/the outer `problems` list, which `solve_iteratively`
    already sorts smallest-first once at the top, so a miss tends to predict misses on the
    (larger, harder) problems that follow.

    `max_consecutive_failures` (default None) stops testing once that many problems *in a row*
    fail, and the returned list then undercounts -- it is a lower bound on how many problems the
    policy actually solves, not the exact number. That is fine for `solve_iteratively`'s own
    round-to-round `keep_best_policy` comparison (both candidates it compares are counted the
    same way, and the exact number is recomputed once, unbounded, by `__main__`'s final
    verification) but NOT for `_final_cost_minimization_pass`: it compares candidates by exact
    solved count, so silently truncating a later candidate because an earlier problem already
    failed would bias that comparison -- `_final_cost_minimization_pass` therefore always calls
    this with the default `max_consecutive_failures=None` (test every problem), matching its
    pre-existing behaviour exactly.

    `problem_iterator`, when given, is told the outcome of every problem actually tested
    (`ProblemIterator.set_solved`); a problem never reached because of an early stop is left
    alone. `stats`, when given, accumulates `stats["validationTime"]` (wall time spent in this
    call) and increments `stats["validationEarlyStops"]` when `max_consecutive_failures` actually
    cut the test short.

    `wall_deadline` (an absolute `time.perf_counter()`-based deadline, the same kind
    `solve_iteratively`'s round loop and `_final_cost_minimization_pass` use) and
    `genfond.shutdown.stop_requested()` are checked before every problem *and* before every
    repeated iteration of the same problem, not just between calls to this function: four 12h
    SLURM jobs (barman, grid, reward, spanner) were killed without ever writing a stats row or
    policy file because a single call to this function -- one round's whole in-loop test -- ran
    for hours on its own, well past `--max-wall-time`, with nothing checking either flag until
    the *next* round's top-of-loop check that never arrived. When either fires, testing stops
    immediately (mid-problem if it fires between iterations) and `stats["validationStoppedBy"]`
    is set to `"wall_time"` or `"signal"`; the caller must check that and end the run gracefully
    the same way the round loop's own top-of-loop check does, exactly as if this had been caught
    between rounds instead of inside one -- see `solve_iteratively`'s post-call check. The
    problem being tested when this fires is not recorded either way (neither solved nor failed):
    its own outcome is unknown, only that there was no time left to find out.
    """
    iterations = config["validation_iterations"]
    time_limit = config.get("validation_time_limit")
    solved_problems: list[Problem] = []
    consecutive_failures = 0
    stopped_by: Optional[str] = None
    start = time.perf_counter()
    problem_list = list(problems)
    iterable = tqdm.tqdm(problem_list, disable=None) if show_progress else problem_list

    def _deadline_hit() -> Optional[str]:
        if wall_deadline is not None and time.perf_counter() >= wall_deadline:
            return "wall_time"
        if stop_requested():
            return "signal"
        return None

    with logging_redirect_tqdm():
        for problem in iterable:
            stopped_by = _deadline_hit()
            if stopped_by is not None:
                break
            log.info(f"Testing policy on {problem.name} {iterations} time(s) ...")
            plan_lengths = []
            solved = True
            for _ in range(iterations):
                stopped_by = _deadline_hit()
                if stopped_by is not None:
                    break
                try:
                    plan_lengths.append(len(execute_policy(domain, problem, policy, config, time_limit=time_limit)))
                except NoActionError:
                    log.info(f"Policy does not solve {problem.name}, no action in reachable state")
                    solved = False
                    break
                except CycleError as e:
                    log.info(f"Policy does not solve {problem.name}, found cycle of length {len(e.cycle)}")
                    solved = False
                    break
                except ExecutionTimeout:
                    log.info(f"Policy does not solve {problem.name}, execution exceeded validation_time_limit")
                    solved = False
                    break
                except RuntimeError:
                    log.info(f"Policy does not solve {problem.name}")
                    solved = False
                    break
            if stopped_by is not None:
                # This problem's own outcome is unknown (it may have been mid-way through its
                # repeated iterations) -- leave it unrecorded and stop without counting it either
                # way, same as a problem never reached at all.
                log.warning(
                    "In-loop validation: stopping before/during %s (%s); %d/%d problem(s) tested" " this call",
                    problem.name,
                    stopped_by,
                    len(solved_problems) + consecutive_failures,
                    len(problem_list),
                )
                break
            if problem_iterator is not None:
                problem_iterator.set_solved(problem, solved)
            if solved:
                log.info(
                    f"Policy already solves {problem.name} (plan length "
                    f"{statistics.mean(plan_lengths)} ± {statistics.stdev(plan_lengths):.2f})"
                    if len(plan_lengths) > 1
                    else f"Policy already solves {problem.name} (plan length {plan_lengths[0]})"
                )
                solved_problems.append(problem)
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                if max_consecutive_failures is not None and consecutive_failures >= max_consecutive_failures:
                    if stats is not None:
                        stats["validationEarlyStops"] = stats.get("validationEarlyStops", 0) + 1
                    log.info(
                        f"Stopping validation after {consecutive_failures} consecutive failure(s)"
                        " (validation_max_consecutive_failures); remaining problems are untested"
                        " this round"
                    )
                    break
    if stats is not None:
        stats["validationTime"] = stats.get("validationTime", 0.0) + (time.perf_counter() - start)
        if stopped_by is not None:
            stats["validationStoppedBy"] = stopped_by
    return solved_problems


def _final_cost_minimization_pass(
    domain: Domain,
    problems: list[Problem],
    problem_iterator: ProblemIterator,
    config: Mapping,
    stats: MutableMapping[str, Any],
    policy: Policy | DatalogPolicy,
    wall_deadline: Optional[float] = None,
) -> tuple[Policy | DatalogPolicy, list[Problem]]:
    """Look for a cheaper policy on the final training set, once the main loop has nothing left
    to add.

    `wall_deadline` is an absolute `time.perf_counter()`-based deadline, the same kind
    `solve_iteratively`'s main loop uses (see its module-level comment) -- but not necessarily
    the same *value*: it is always `wall_time_start + max_wall_time - wall_time_reserve`, i.e.
    this pass may run right up to the point that leaves `wall_time_reserve` seconds for
    `__main__`'s own (unbounded) verification pass after `solve_iteratively` returns, even when
    the main loop's own deadline was narrowed by `final_pass_budget` to leave this pass a
    guaranteed slice of the budget instead of whatever happens to be left over. None
    (`max_wall_time` unset) reproduces the previous unbounded pass exactly.

    A cluster run was killed by the SLURM time limit *inside* this pass: the main loop stopped
    gracefully at its own deadline, but the pass then ran its own climb rounds (each a fresh,
    increasingly expensive `solve_step` call) with nothing checking `wall_deadline` or
    `genfond.shutdown.stop_requested()`, so it ran past `wall_time_reserve` and the job was
    killed before the final evaluation, stats row and policy file were ever written -- exactly
    the failure mode `max_wall_time` exists to prevent in the main loop. This closes the same
    gap here: before starting the pass at all (skipping it outright, see below), and again
    before every round once it is running.

    If the main loop already stopped because it ran out of wall-clock budget
    (`stats["stoppedBy"] == "wall_time"`), starting this pass at all is only worthwhile when a
    healthy amount of the remaining budget is left for it -- a pass given only a few seconds
    cannot usefully climb even one complexity level, and it would rather not spend those seconds
    on a `solve_step` call it will likely have to cut off anyway. `final_pass_min_time` (config,
    default 600s) is that threshold, checked against `wall_deadline - now`. It is not applied
    when the main loop stopped for some other reason (every problem solved, or the ladder
    exhausted its own escalation branches): in that case any remaining time, however little, is
    better spent attempting a round than not.

    Called unconditionally after `solve_iteratively`'s main `for` loop ends, for any reason that
    lets it return normally: the `stop_after_first_solution` break (every problem solved), or
    the loop's `for` exhausting `problem_iterator` (`StopIteration` -- e.g. every escalation
    branch closed off after an `OUT_OF_RESOURCES`/`TIMEOUT`, or the ladder reached
    `max_complexity` with problems still unsolved, which is the common case on a suite too large
    to solve in full, where `stop_after_first_solution`'s break never fires at all). It is *not*
    reached after an unhandled exception, since that propagates out of the loop instead of
    letting it return. The only gate that matters here is `policy is not None`: some round
    succeeded at some point, regardless of how the run subsequently ended.

    `add_problem_after_success` (see its config comment and docs/no-cost-climb-results.md) skips
    the complexity climb after every success so the loop grows the training set instead. When a
    chain of successes solves every problem in a row -- or more generally, whenever the run ends
    with at least one success behind it -- that climb never runs at all, and the final policy
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

    `final_pass_max_levels` (config, default 2; null reproduces the previous unbounded climb)
    caps the number of complexity levels this pass tries above `succ_complexity`, independently
    of `max_cost`/`max_complexity`. A cluster run on blocks3ops climbed unchecked from
    `succ_complexity` all the way up to a pool of 1,945 features / 5,738 concepts before a single
    `solve_step` call exceeded any budget and the job died -- this pass is meant to look for a
    *cheaper* policy near the one the main loop already found, not to explore the whole ladder,
    so a small, bounded number of levels is the right default. `rounds` (also reported as
    `finalPassLevelsTried`) already counts exactly one complexity level per loop iteration, so
    bounding it bounds levels tried directly.

    `final_pass_max_pool` (config only, no CLI flag; default null = no limit) skips an individual
    round -- without grounding at all -- when the pool `FeaturePool` builds for it (features +
    concepts + roles) exceeds this many elements; see `solve()`'s check, placed right after the
    pool is built and before `to_clingo()`/`Solver.solve()`, which is where the memory actually
    blows up. A skipped round is reported exactly like `Result.NO_SOLUTION` (the pass keeps
    climbing), so `final_pass_max_levels` is what ultimately bounds the pass's worst case when
    every remaining level's pool is oversized.
    """
    if wall_deadline is not None:
        remaining = wall_deadline - time.perf_counter()
        skip_reason: Optional[str] = None
        if stop_requested():
            skip_reason = "stop_requested"
        elif remaining <= 0:
            skip_reason = "deadline_exhausted"
        elif stats.get("stoppedBy") == "wall_time":
            final_pass_min_time = config.get("final_pass_min_time")
            if final_pass_min_time is None:
                final_pass_min_time = 600
            if remaining < final_pass_min_time:
                skip_reason = "insufficient_wall_time"
        if skip_reason is not None:
            log.warning(
                "Skipping final cost minimization pass: %s (%.1fs remaining before deadline)",
                skip_reason,
                remaining,
            )
            stats["finalPassSkipped"] = skip_reason
            # problem_iterator.solved reflects the main loop's own testing, without
            # re-executing the policy -- exactly what solve_iteratively would return unchanged
            # if it never called this function at all, which is what a skip amounts to.
            already_solved = [p for p in problems if problem_iterator.solved[p.name]]
            return policy, already_solved

    active_problems = problem_iterator.active_problems
    example_plans = problem_iterator.active_plans
    dead_states = problem_iterator.dead_states
    complexity = problem_iterator.succ_complexity
    active_problem_names = {p.name for p in active_problems}

    assert policy.cost is not None, "the main loop only sets `policy` from a Result.SUCCESS, which always has a cost"
    best_policy = policy
    best_solved = _test_policy_on_problems(domain, problems, policy, config, stats=stats, wall_deadline=wall_deadline)
    best_cost = feature_cost(
        policy.cost, config.get("minimize_good_signatures", "none"), config.get("minimize_selected_count", "none")
    )
    stats["finalPassCostBefore"] = best_cost
    max_cost = best_cost - 1
    rounds = 0
    final_pass_max_levels = config.get("final_pass_max_levels")
    max_pool_size = config.get("final_pass_max_pool")
    log.info(f"Starting final cost minimization pass from complexity {complexity}, max cost {max_cost}")
    while (
        max_cost > complexity
        and complexity < config["max_complexity"]
        and (final_pass_max_levels is None or rounds < final_pass_max_levels)
    ):
        if wall_deadline is not None and (stop_requested() or time.perf_counter() >= wall_deadline):
            stats["finalPassStoppedBy"] = "signal" if stop_requested() else "wall_time"
            log.warning(
                "Final cost minimization pass: stopping before round %d (wall-clock deadline"
                " exhausted or stop requested)",
                rounds + 1,
            )
            break
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
            wall_deadline=wall_deadline,
            max_pool_size=max_pool_size,
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
        solved = _test_policy_on_problems(
            domain, problems, new_policy, config, stats=stats, wall_deadline=wall_deadline
        )
        solved_names = {p.name for p in solved}
        if not (active_problem_names <= solved_names):
            log.info(f"Candidate policy at complexity {complexity} does not solve every training problem, discarding")
            continue
        if (len(solved), -new_cost) > (len(best_solved), -best_cost):
            best_policy, best_solved, best_cost = new_policy, solved, new_cost
    stats["finalPassRounds"] = rounds
    # Same value as finalPassRounds by construction (one complexity level per loop iteration,
    # unconditionally) -- reported under its own name since it's what final_pass_max_levels
    # bounds, and a clearer stats-column name than reusing finalPassRounds for that purpose.
    stats["finalPassLevelsTried"] = rounds
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
    # `keep_best_policy` (default True) tracks the best policy seen across the whole run -- by
    # (problems solved on the full `problems` list, then lower feature cost) -- instead of just
    # returning whichever policy the loop happens to end on. A policy learned early from a small
    # training set can already generalize to nearly every problem; if the loop keeps adding a
    # stubborn holdout to the training set afterwards, later policies are patchworks fitted to a
    # larger, harder set and can solve far fewer problems overall, even though every one of them
    # was already tested on all problems inside the loop (`_test_policy_on_problems`-equivalent
    # block below). The old code discarded that early policy anyway, returning only the last one.
    # `keep_best_policy: false` restores that old behaviour exactly.
    keep_best_policy = config.get("keep_best_policy", True)
    best_policy: Optional[Policy | DatalogPolicy] = None
    best_solved: list[Problem] = []
    best_cost: Optional[int] = None
    best_round: Optional[str | int] = None
    round_num = 0
    # An absolute deadline (time.perf_counter()-based) for the whole run, `wall_time_reserve`
    # seconds short of `max_wall_time` so that budget is left over for the final verification
    # loop in __main__ after this function returns. `final_pass_deadline` is that deadline --
    # also the one `_final_cost_minimization_pass` itself must respect, see its docstring.
    # `wall_deadline` is the (possibly earlier) deadline handed to the *round loop* below and to
    # every round's Solver as `wall_deadline`, so a round already in flight when the budget runs
    # out is cut off at this instant too, rather than only being prevented from starting the next
    # one -- see docs/wall-budget-results.md. None (the default, `max_wall_time: null`)
    # reproduces the previous behaviour exactly: no round is ever cut short and this loop never
    # stops early.
    #
    # When the final cost-minimization pass is enabled (`final_cost_minimization` together with
    # `add_problem_after_success`, its only non-no-op combination), `wall_deadline` is narrowed
    # by `final_pass_budget` seconds so the round loop stops that much *earlier* than
    # `final_pass_deadline`, guaranteeing the pass a budget of its own instead of only whatever
    # happens to be left once the loop has already run right up to `final_pass_deadline` -- the
    # gap that let a cluster run's pass overrun the SLURM time limit, see
    # docs/final-climb-results.md and _final_cost_minimization_pass's docstring.
    wall_time_start = time.perf_counter()
    max_wall_time = config.get("max_wall_time")
    final_pass_deadline: Optional[float] = None
    wall_deadline: Optional[float] = None
    if max_wall_time is not None:
        final_pass_deadline = wall_time_start + max_wall_time - (config.get("wall_time_reserve") or 0)
        wall_deadline = final_pass_deadline
        if config.get("final_cost_minimization", False) and config.get("add_problem_after_success", False):
            wall_deadline = final_pass_deadline - (config.get("final_pass_budget") or 0)
    example_plans: dict[str, Iterator[Plan]] = dict()
    planner_compute_plans: Optional[PlannerComputePlans] = None
    planner_config: dict[str, Any] = dict()
    problems_by_name = {problem.name: problem for problem in problems}
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
    # Only meaningful once there are extra plans to dedupe (INC_PLANS / frontier expansion);
    # without use_example_plans the iterator never calls plan_coverage.add() at all.
    plan_coverage = PlanStateCoverage(domain, problems_by_name) if config["use_example_plans"] else None
    problem_iterator: ProblemIterator | OneShotProblemIterator
    if one_shot:
        problem_iterator = OneShotProblemIterator(problems, config, plans=example_plans, plan_coverage=plan_coverage)
    else:
        problem_iterator = ProblemIterator(problems, config, plans=example_plans, plan_coverage=plan_coverage)
    for iter_kwargs in problem_iterator:
        round_num += 1
        if wall_deadline is not None and time.perf_counter() >= wall_deadline:
            # `max_wall_time - (wall_deadline - wall_time_start)` is what's held back in total --
            # just `wall_time_reserve` normally, or `wall_time_reserve + final_pass_budget` when
            # the final pass narrowed `wall_deadline` (see the deadline computation above).
            # wall_deadline is only ever set below when max_wall_time is not None.
            assert max_wall_time is not None
            log.warning(
                "Wall-clock budget exhausted (%.1fs used of %.1fs, %.1fs held back); stopping"
                " without starting another round",
                time.perf_counter() - wall_time_start,
                max_wall_time,
                max_wall_time - (wall_deadline - wall_time_start),
            )
            stats["stoppedBy"] = "wall_time"
            stats["wallBudgetUsed"] = time.perf_counter() - wall_time_start
            break
        if stop_requested():
            # SIGINT/SIGTERM (see genfond.shutdown); a round already in flight was already cut
            # off by the same flag inside Solver.solve, so there is nothing left running here.
            log.warning("Stop requested; stopping without starting another round")
            stats["stoppedBy"] = "signal"
            stats["wallBudgetUsed"] = time.perf_counter() - wall_time_start
            break
        result, new_policy, frontier_states = solve_step(
            **iter_kwargs,
            domain=domain,
            stats=stats,
            config=config,
            wall_deadline=wall_deadline,
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
        assert new_policy is not None, "solve_step must return a policy on Result.SUCCESS"
        assert new_policy.cost is not None, "a Result.SUCCESS policy always has a cost"
        policy = new_policy
        log.info(f'Testing policy on unsolved problems {config["validation_iterations"]} time(s) ...')
        # keep_best_policy off reproduces the old behaviour exactly: stop at the very first
        # problem the round cannot solve (max_consecutive_failures=1), since `problems` is
        # sorted smallest-first and a miss there usually means every larger problem misses too.
        # keep_best_policy needs the coverage count from every round to compare candidates by,
        # so it uses validation_max_consecutive_failures instead (None = test everything,
        # reproducing the exact-count behaviour the old code had while keep_best_policy was on);
        # a round that stops early is inherently incomplete, so it can never count as "solves
        # all problems" below regardless of the cutoff value.
        round_solved = _test_policy_on_problems(
            domain,
            problems,
            policy,
            config,
            max_consecutive_failures=(
                1 if not keep_best_policy else config.get("validation_max_consecutive_failures")
            ),
            problem_iterator=problem_iterator,
            stats=stats,
            show_progress=True,
            wall_deadline=wall_deadline,
        )
        if stats.get("validationStoppedBy") is not None:
            # In-loop validation itself hit the wall-clock deadline or a pending stop signal
            # mid-test (see _test_policy_on_problems's wall_deadline handling) -- a single
            # round's testing block can itself run for hours on a large suite (see
            # docs/cheap-validation-results.md), so this is checked inside that block too, not
            # only between rounds. End the run here exactly like the top-of-loop
            # wall_deadline/stop_requested check would on the next iteration, so __main__'s
            # final verification, stats row and policy file are still written instead of losing
            # everything to a SIGKILL/SIGTERM that arrives before the next round even starts.
            log.warning(
                "In-loop validation hit its wall-clock deadline or a pending stop request;"
                " stopping without starting another round"
            )
            stats["stoppedBy"] = stats.pop("validationStoppedBy")
            stats["wallBudgetUsed"] = time.perf_counter() - wall_time_start
            break
        solved = len(round_solved) == len(problems)
        if keep_best_policy:
            assert policy.cost is not None, "a Result.SUCCESS policy always has a cost"
            round_cost = feature_cost(
                policy.cost,
                config.get("minimize_good_signatures", "none"),
                config.get("minimize_selected_count", "none"),
            )
            is_new_best = best_policy is None
            if not is_new_best:
                assert best_cost is not None
                is_new_best = (len(round_solved), -round_cost) > (len(best_solved), -best_cost)
            if is_new_best:
                best_policy, best_solved, best_cost, best_round = policy, round_solved, round_cost, round_num
            log.info(
                "Policy solves %d/%d (best so far %d/%d from round %s)",
                len(round_solved),
                len(problems),
                len(best_solved),
                len(problems),
                best_round,
            )
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
    # Normally seeded by solve_step's first round (`stats.get("totalSolveCpuTime", 0) + ...`);
    # __main__ reads it unconditionally (unlike bestSolve*, which are guarded by `if policy:`).
    # The wall-budget/stop-request break above can end this loop before solve_step ever runs a
    # single round, which is the one way that invariant used to hold unconditionally -- every
    # problem set is non-empty, so the loop's first solve_step call always ran before anything
    # could stop it.
    stats.setdefault("totalSolveCpuTime", 0)
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
        # final_cost_minimization). Reached whenever the loop above returns normally with a
        # policy in hand, whether that was via the stop_after_first_solution break or the for
        # loop exhausting problem_iterator (StopIteration, e.g. an unsolved-but-unaddable suite
        # that hits OUT_OF_RESOURCES/TIMEOUT or max_complexity) -- see the docstring of
        # _final_cost_minimization_pass. Not reached after an unhandled exception, which
        # propagates out of the loop instead.
        policy, solved_problems = _final_cost_minimization_pass(
            domain, problems, problem_iterator, config, stats, policy, wall_deadline=final_pass_deadline
        )
    if keep_best_policy and policy is not None:
        assert policy.cost is not None, "a policy this function returns always has a cost"
        final_cost = feature_cost(
            policy.cost, config.get("minimize_good_signatures", "none"), config.get("minimize_selected_count", "none")
        )
        is_new_best = best_policy is None
        if not is_new_best:
            assert best_cost is not None
            is_new_best = (len(solved_problems), -final_cost) > (len(best_solved), -best_cost)
        if is_new_best:
            best_policy, best_solved, best_cost, best_round = policy, solved_problems, final_cost, "final_pass"
        assert best_policy is not None and best_cost is not None and best_round is not None
        stats["lastSolved"] = len(solved_problems)
        stats["bestSolved"] = len(best_solved)
        stats["bestCost"] = best_cost
        stats["bestRound"] = best_round
        # Only swap in the best policy when it strictly solves more problems than the one this
        # run would otherwise return -- a tie is left alone (the returned policy is already the
        # cheapest one found for that coverage), so a cost-only difference never causes a swap.
        if len(best_solved) > len(solved_problems):
            log.warning(
                "Returning the best policy from round %s (%d/%d problems) instead of the final"
                " one (%d/%d problems)",
                best_round,
                len(best_solved),
                len(problems),
                len(solved_problems),
                len(problems),
            )
            policy, solved_problems = best_policy, best_solved
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
    wall_deadline: Optional[float] = None,
    max_pool_size: Optional[int] = None,
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
            wall_deadline=wall_deadline,
            max_pool_size=max_pool_size,
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
