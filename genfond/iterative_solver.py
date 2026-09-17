import gc
import logging
import pickle
import resource
import statistics
import sys
import time
import uuid
from collections.abc import Iterator
from typing import Any, Callable, Collection, Mapping, MutableMapping, Optional, Sequence

import tqdm
from pddl.action import Action
from pddl.core import Domain, Plan, Problem
from tqdm.contrib.logging import logging_redirect_tqdm

from .checkpoint import PROVISIONAL_COLUMN, append_stats_row, checkpoint_best_policy
from .cost_utils import feature_cost
from .datalog_policy import DatalogPolicy
from .execute_datalog_policy import CycleError, ExecutionTimeout, NoActionError
from .execute_policy import execute_policy
from .feature_generator import FeaturePool
from .frontier import FrontierState, collect_frontier_states, expand_frontier
from .generate_policy import generate_policy
from .lazy_pairs import DEFAULT_BATCH, solve_with_lazy_pairs
from .policy import PolicyType
from .problem_iterator import (
    MAX_COST,
    OneShotProblemIterator,
    PlanStateCoverage,
    ProblemIterator,
    Result,
    plan_from_actions,
)
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
    # The best policy's own trajectories on the problems it solved (H29,
    # `anchor_policy_labels`). Restricted to `problems` by the caller; see
    # `FeaturePool._emit_anchors`.
    anchor_plans: Optional[Mapping[str, Plan]] = None,
    allow_frontier: bool = True,
    stats: Optional[MutableMapping[str, Any]] = None,
    wall_deadline: Optional[float] = None,
    max_pool_size: Optional[int] = None,
    optimal_model_limit: int = 1,
) -> Optional[tuple[list[DatalogPolicy | Policy], MutableMapping[str, Any], list[FrontierState]]]:
    # The caller may hand in the run-level stats dict so that the keys describing *why* a round
    # produced no policy (solveStatus, solveOptimal) survive a `return None`; the returned
    # mapping is then that same object and `stats.update(...)` on it is a no-op.
    if stats is None:
        stats = dict()
    # Defaults for the paths that return before a solve happens at all. `None` is deliberately
    # not a SolveStatus name: those paths keep the pre-existing NO_SOLUTION handling.
    stats["solveStatus"] = None
    stats["solveOptimal"] = True
    # The per-round optimal-model keys describe *this* round; without clearing them a round that
    # enumerates nothing would report the previous round's numbers. The `*Total` counters below
    # are cumulative over the run and are deliberately not cleared.
    for key in (
        "optimalModelsEnumerated",
        "optimalModelsFeasible",
        "optimalModelChosenCoverage",
        "optimalModelChosenIndex",
        "optimalModelCoverages",
        # Per-round anchor numbers, for the same reason: a round that anchors nothing must not
        # report the previous round's. `anchorFallbacks` is cumulative and deliberately kept.
        "anchoredTransitions",
        "anchoredProblems",
        "anchorsDropped",
    ):
        stats.pop(key, None)
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
    if config.get("anchor_policy_labels", False) and config["solve_prog"] not in (
        "solve_datalog.lp",
        "solve_datalog_sig.lp",
    ):
        # anchor/3 and the `anchor` #program part only exist in the two datalog programs;
        # grounding the part elsewhere would fail inside clingo.
        raise ValueError(
            "anchor_policy_labels=True needs solve_prog='solve_datalog_sig.lp' or"
            f" 'solve_datalog.lp' (--type datalog-sig/datalog), got {config['solve_prog']!r}"
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
        anchor_plans=anchor_plans,
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
    forced = feature_pool.forced_labels
    if forced is not None:
        stats["forcedGoodActions"] = len(forced.good)
        stats["forcedBadActions"] = len(forced.bad)
        stats["forcedGoodSignatures"] = len(forced.good_signatures)
        stats["forcedBadSignatures"] = len(forced.bad_signatures)
        stats["forcedOccurrences"] = forced.num_occurrences
        stats["forcedInconsistent"] = forced.inconsistent

    def _run_solver(use_anchors: bool) -> tuple[Solver, SolveStatus]:
        """One clingo run over the instance, with or without the anchor constraint.

        Everything but the `anchor` #program part is identical between the two, so the fallback
        below re-solves the *same* round: same feature pool, same instance text (the anchor/3
        facts stay in it either way), same budget.
        """
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
            anchors=use_anchors,
        )
        if config.get("lazy_pairs", False) and config.get("emit_action_signatures", False):
            # The instance carries no separation pairs; they are added batch by batch in response
            # to the models that violate them. The loop wraps the solve of this one round only,
            # so max_cost, the min_feature_complexity program and the frontier machinery are
            # untouched.
            status = solve_with_lazy_pairs(
                solver,
                feature_pool.signatures,
                config.get("lazy_pairs_batch") or DEFAULT_BATCH,
                stats,
                forced=forced,
                optimal_model_limit=optimal_model_limit,
            )
        else:
            solver.solve()
            status = solver.status
            if optimal_model_limit > 1 and status == SolveStatus.OPTIMAL:
                # The lazy loop does this itself (it has to run the enumeration on its *final*
                # grounded program and re-check every model against the pairs it never grounded);
                # here the grounded program is the whole problem, so every enumerated model of the
                # optimal cost is feasible as it stands.
                solver.enumerate_optimal(optimal_model_limit)
        return solver, status

    anchored = bool(feature_pool.anchored_transitions)
    if anchored:
        stats["anchoredTransitions"] = len(feature_pool.anchored_transitions)
        stats["anchoredProblems"] = len(feature_pool.anchored_problems)
        stats["anchorsDropped"] = feature_pool.anchors_dropped
    solver, status = _run_solver(anchored)
    if anchored and status in (SolveStatus.UNSATISFIABLE, SolveStatus.UNKNOWN):
        # The anchors are a preference, never a refutation: an anchored occurrence may share a
        # signature class with one that has to be bad elsewhere, and with the quotient that makes
        # the round unsatisfiable even though a policy exists. So this round's *answer* must
        # always come from a solve without them -- which is also what keeps the refuted-complexity
        # bookkeeping in `ProblemIterator` sound, since only the unanchored solve is evidence
        # about what this complexity level can express. UNKNOWN (the solve budget ran out before
        # any model) is retried for the same reason: the anchored attempt proves nothing either
        # way, and a cheaper unanchored program may still find one.
        log.info("Anchored solve unsatisfiable; retrying without anchors")
        stats["anchorFallbacks"] = int(stats.get("anchorFallbacks", 0)) + 1
        solver, status = _run_solver(False)
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
    policy_type = PolicyType[config["policy_type"]]
    try:
        policy = generate_policy(solution, policy_type=policy_type, signatures=feature_pool.signatures)
    except KeyError as e:
        log.error(f"Error during policy generation: {e}")
        raise
    policies: list[DatalogPolicy | Policy] = [policy]
    if frontier_states:
        # Not a policy at all (see solve_step): the caller expands the frontier and re-solves,
        # so there is nothing for a tie-breaker to choose between.
        return policies, stats, frontier_states
    # Further optimal models of the same cost, if the round asked for them. Each becomes a
    # candidate policy; `solve_step` picks between them by coverage. A candidate that reaches
    # into the frontier is dropped rather than returned: only a zero-frontier model is a policy,
    # and the incumbent already gave us one, so there is no frontier expansion to trigger.
    if optimal_model_limit > 1:
        stats["optimalModelsEnumerated"] = solver.num_enumerated
        stats["optimalModelsEnumeratedTotal"] = stats.get("optimalModelsEnumeratedTotal", 0) + solver.num_enumerated
    for candidate in solver.candidates[1:]:
        if collect_frontier_states(feature_pool, candidate):
            log.info("Discarding an enumerated optimal model: it relies on frontier transitions")
            continue
        policies.append(generate_policy(candidate, policy_type=policy_type, signatures=feature_pool.signatures))
    if optimal_model_limit > 1:
        stats["optimalModelsFeasible"] = len(policies)
        stats["optimalModelsFeasibleTotal"] = stats.get("optimalModelsFeasibleTotal", 0) + len(policies)
    if len(policies) > 1:
        log.info("Round has %d candidate policies of cost %s", len(policies), solver.cost)
    return policies, stats, frontier_states


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
    outcomes: Optional[MutableMapping[str, bool]] = None,
    trajectories: Optional[MutableMapping[str, Plan]] = None,
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

    `outcomes`, when given, is filled with `problem name -> solved` for exactly the problems whose
    outcome was actually determined -- the same set `problem_iterator.set_solved` is called for.
    It lets a caller that must not record into the iterator yet (`solve_iteratively`'s per-
    candidate validation: the losing candidates' results must not count) replay the winner's
    results afterwards instead of re-running the whole test.

    `trajectories`, when given, is filled with `problem name -> Plan` for exactly the problems
    the policy *solved*, holding the action sequence the first solving execution took (the
    policy-conformant plans of `policy_conformant_plans`; see
    `ProblemIterator.record_policy_plans`). It is the only thing that asks `execute_policy` for
    its trajectory: left at None, `execute_policy` is called exactly as it always was, with no
    extra keyword -- which is also what keeps every existing `execute_policy` mock working.
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
            # The trajectory of the first execution of this problem. Execution is deterministic
            # given the seed for a datalog policy, but `validation_iterations > 1` still runs it
            # several times; the first one that got through is the one recorded.
            first_trajectory: Optional[list[Action]] = None
            for _ in range(iterations):
                stopped_by = _deadline_hit()
                if stopped_by is not None:
                    break
                try:
                    if trajectories is None:
                        plan_lengths.append(
                            len(execute_policy(domain, problem, policy, config, time_limit=time_limit))
                        )
                    else:
                        taken: list[Action] = []
                        plan_lengths.append(
                            len(
                                execute_policy(
                                    domain, problem, policy, config, time_limit=time_limit, out_actions=taken
                                )
                            )
                        )
                        if first_trajectory is None:
                            first_trajectory = taken
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
            if outcomes is not None:
                outcomes[problem.name] = solved
            if solved:
                log.info(
                    f"Policy already solves {problem.name} (plan length "
                    f"{statistics.mean(plan_lengths)} ± {statistics.stdev(plan_lengths):.2f})"
                    if len(plan_lengths) > 1
                    else f"Policy already solves {problem.name} (plan length {plan_lengths[0]})"
                )
                solved_problems.append(problem)
                consecutive_failures = 0
                if trajectories is not None and first_trajectory:
                    # An empty trajectory means the goal already held at `problem.init`; it is
                    # useless as an example plan (siw_planner drops those too).
                    trajectories[problem.name] = plan_from_actions(first_trajectory)
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


def _write_provisional_stats_row(config: Mapping, stats: Mapping[str, Any]) -> None:
    """Append a provisional row for the in-progress run, tagged with `stats["runId"]`.

    Called the moment `solve_iteratively`'s round loop notices a graceful stop (SIGTERM/SIGINT,
    or an exhausted `--max-wall-time` budget) -- well before the run would otherwise finish -- so
    a hard kill that follows (SLURM's SIGKILL after the SIGTERM warning) still leaves a row
    behind. `__main__.py` deletes this row (`checkpoint.remove_provisional_row`, matched by
    `runId`) right before it appends the real final row for the same run, so a run that does
    finish never leaves a duplicate. See `genfond/checkpoint.py` for what this can and cannot
    protect against, and `checkpoint_best_policy` (used at the two coverage-improvement call
    sites below) for the accompanying policy checkpoint.
    """
    if not config.get("checkpoint_best_policy", True):
        return
    stats_path = config.get("stats")
    if not stats_path:
        return
    row = dict(stats)
    row[PROVISIONAL_COLUMN] = 1
    append_stats_row(stats_path, row, create_if_missing=False)
    log.info(f"Wrote provisional stats row (runId {stats.get('runId')}) to {stats_path}")


def solve_iteratively(
    domain: Domain,
    problems: list[Problem],
    config: Mapping,
    one_shot: bool = False,
    extra_stats: Optional[Mapping[str, Any]] = None,
) -> tuple[Optional[Policy | DatalogPolicy], list[Problem], dict[str, str | int | float]]:
    policy = None
    problems.sort(key=lambda p: len(p.objects))
    stats: dict[str, str | int | float] = dict()
    # A per-run id (not just informational): `_write_provisional_stats_row` and
    # `checkpoint.remove_provisional_row` use it to tag/find this run's provisional row across
    # the stats CSV's other rows -- including other runs' own provisional/final rows, which may
    # be appended concurrently by a parallel SLURM array job sharing the same --stats path.
    stats["runId"] = uuid.uuid4().hex
    if extra_stats:
        stats.update(extra_stats)
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
    # The trajectories of `best_policy` itself, per problem name -- the plans H29 anchors on.
    # Kept separately from the iterator's `active_plans`, which mixes in the planner's plans and
    # the trajectories of every *other* candidate that ever solved something: anchoring those
    # would pin labels of policies the loop has already rejected. Replaced wholesale whenever a
    # new best policy is adopted, so it always describes exactly one policy.
    best_trajectories: dict[str, Plan] = dict()
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
    # Policy-conformant example plans (H27): whenever a candidate policy is validated and solves
    # a problem, the trajectory it took is recorded as an example plan for that problem, so the
    # plan-restricted state space of every later round contains it and the policy that produced
    # it stays feasible on the ASP instance. Only meaningful with `use_example_plans` -- without
    # it `StateSpaceGraph` is not plan-restricted in the first place, so no trajectory can ever
    # be missing from it. That gate is also what keeps the default `true` harmless for the
    # rule-based state/trans/d2l types, none of which use example plans.
    policy_conformant_plans = bool(config.get("policy_conformant_plans", False)) and bool(config["use_example_plans"])
    if policy_conformant_plans:
        log.info("Policy-conformant example plans are enabled")
    # Policy-anchored labels (H29): every transition of the best policy's own trajectories is
    # emitted as `anchor/3` and constrained to be good, so the next round's model has to *agree*
    # with that policy where it already worked instead of merely being allowed to. H27 keeps the
    # policy feasible; this keeps it preferred. Same `use_example_plans` gate: without a
    # plan-restricted state space there is no drift of the kind this addresses, and the rule-based
    # types would not even have the `anchor` program part.
    anchor_policy_labels = bool(config.get("anchor_policy_labels", False)) and bool(config["use_example_plans"])
    if anchor_policy_labels and not keep_best_policy:
        # There is no "best policy" to anchor on then -- only whichever policy the last round
        # happened to produce, which is exactly the drifting thing this mechanism corrects.
        log.warning("anchor_policy_labels needs keep_best_policy; no anchors will be emitted")
        anchor_policy_labels = False
    if anchor_policy_labels:
        log.info("Policy-anchored labels are enabled")
    # The trajectories are a by-product of validation, and both mechanisms consume them.
    track_trajectories = policy_conformant_plans or anchor_policy_labels
    if anchor_policy_labels and not policy_conformant_plans:
        # Anchoring a transition the plan-restricted state space does not contain is a no-op at
        # best: `_emit_anchors` drops every step that is not in the graph. H27 is what puts the
        # best policy's trajectory there in the first place.
        log.warning(
            "anchor_policy_labels without policy_conformant_plans: the best policy's trajectories"
            " are not example plans, so most anchored steps will be missing from the state space"
        )
    problem_iterator: ProblemIterator | OneShotProblemIterator
    if one_shot:
        problem_iterator = OneShotProblemIterator(problems, config, plans=example_plans, plan_coverage=plan_coverage)
    else:
        problem_iterator = ProblemIterator(problems, config, plans=example_plans, plan_coverage=plan_coverage)
    # One round's validated candidate policies, as (policy, solved problems, per-problem
    # outcomes). `solve_step` scores every equal-cost candidate through `_validate_candidate`
    # below and returns the winner; this is what lets the winner's result be reused here instead
    # of validating it a second time, and what keeps the losers' results out of
    # `problem_iterator` (only the round's chosen policy may mark a problem solved). With the
    # default `optimal_model_limit: 1` there is only ever one candidate, `solve_step` never calls
    # the validator at all, and this list stays empty -- the round then takes the same testing
    # path it always did.
    # The fourth element is the candidate's policy-conformant trajectories (empty when the
    # mechanism is off); only the *winning* candidate's are recorded as example plans, so a
    # losing sibling never grows the state space -- same rule as for `outcomes`.
    candidate_results: list[tuple[Policy | DatalogPolicy, list[Problem], dict[str, bool], dict[str, Plan]]] = []

    def _validate_candidate(candidate: Policy | DatalogPolicy) -> list[Problem]:
        outcomes: dict[str, bool] = dict()
        candidate_trajectories: dict[str, Plan] = dict()
        candidate_solved = _test_policy_on_problems(
            domain,
            problems,
            candidate,
            config,
            max_consecutive_failures=(
                1 if not keep_best_policy else config.get("validation_max_consecutive_failures")
            ),
            stats=stats,
            show_progress=True,
            wall_deadline=wall_deadline,
            outcomes=outcomes,
            trajectories=candidate_trajectories if track_trajectories else None,
        )
        candidate_results.append((candidate, candidate_solved, outcomes, candidate_trajectories))
        return candidate_solved

    for iter_kwargs in problem_iterator:
        round_num += 1
        candidate_results.clear()
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
            _write_provisional_stats_row(config, stats)
            break
        if stop_requested():
            # SIGINT/SIGTERM (see genfond.shutdown); a round already in flight was already cut
            # off by the same flag inside Solver.solve, so there is nothing left running here.
            log.warning("Stop requested; stopping without starting another round")
            stats["stoppedBy"] = "signal"
            stats["wallBudgetUsed"] = time.perf_counter() - wall_time_start
            _write_provisional_stats_row(config, stats)
            break
        # Anchors only make sense for problems whose states are actually in this round's
        # instance; a trajectory on a problem outside the training set has nothing to anchor to.
        anchor_plans = (
            {
                problem.name: best_trajectories[problem.name]
                for problem in iter_kwargs["active_problems"]
                if problem.name in best_trajectories
            }
            if anchor_policy_labels
            else None
        )
        result, new_policy, frontier_states = solve_step(
            **iter_kwargs,
            domain=domain,
            stats=stats,
            config=config,
            wall_deadline=wall_deadline,
            validate=_validate_candidate,
            anchor_plans=anchor_plans,
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
        # Already validated as one of this round's equal-cost candidates (see
        # `candidate_results`, filled by `solve_step` via `_validate_candidate`): reuse that
        # result and replay its per-problem outcomes into the iterator, which
        # `_validate_candidate` deliberately left alone while the candidates were still
        # competing. With `optimal_model_limit: 1` there are no candidates and the round takes
        # the `else` branch, exactly as before.
        round_trajectories: dict[str, Plan] = dict()
        cached = next(
            ((solved_c, out, traj) for cand, solved_c, out, traj in candidate_results if cand is policy), None
        )
        if cached is not None:
            round_solved, cached_outcomes, round_trajectories = cached
            for name, was_solved in cached_outcomes.items():
                problem_iterator.set_solved(problems_by_name[name], was_solved)
        else:
            log.info(f'Testing policy on unsolved problems {config["validation_iterations"]} time(s) ...')
            # keep_best_policy off reproduces the old behaviour exactly: stop at the very first
            # problem the round cannot solve (max_consecutive_failures=1), since `problems` is
            # sorted smallest-first and a miss there usually means every larger problem misses
            # too. keep_best_policy needs the coverage count from every round to compare
            # candidates by, so it uses validation_max_consecutive_failures instead (None = test
            # everything, reproducing the exact-count behaviour the old code had while
            # keep_best_policy was on); a round that stops early is inherently incomplete, so it
            # can never count as "solves all problems" below regardless of the cutoff value.
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
                trajectories=round_trajectories if track_trajectories else None,
            )
        if policy_conformant_plans and round_trajectories:
            # Record before the wall-deadline check below: the plans are a by-product of work
            # already done, and adding them costs nothing even if the run ends here. They do not
            # trigger a round of their own -- they are simply part of the next round's state
            # space, and `record_policy_plans` invalidates the refuted complexity levels the
            # same way INC_PLANS and frontier expansion do.
            added, problems_affected = problem_iterator.record_policy_plans(round_trajectories)
            stats["policyPlansAdded"] = int(stats.get("policyPlansAdded", 0)) + added
            log.info("Added %d policy-conformant plan(s) for %d problem(s)", added, problems_affected)
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
            _write_provisional_stats_row(config, stats)
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
                # Only a strict increase in coverage over the *previous* best is checkpointed
                # here -- a tie broken purely by feature cost does not change what "the best
                # policy so far" solves, so re-pickling it would not change what a kill loses.
                improved_coverage = best_policy is None or len(round_solved) > len(best_solved)
                best_policy, best_solved, best_cost, best_round = policy, round_solved, round_cost, round_num
                # The anchors follow the best policy, so they are replaced wholesale here rather
                # than accumulated: a rejected policy's trajectories must not keep pinning labels.
                best_trajectories = dict(round_trajectories)
                stats["bestSolved"] = len(best_solved)
                stats["bestCost"] = best_cost
                stats["bestRound"] = best_round
                if improved_coverage:
                    checkpoint_best_policy(best_policy, len(best_solved), len(problems), config)
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
            improved_coverage = best_policy is None or len(solved_problems) > len(best_solved)
            best_policy, best_solved, best_cost, best_round = policy, solved_problems, final_cost, "final_pass"
            if improved_coverage:
                checkpoint_best_policy(best_policy, len(best_solved), len(problems), config)
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


def _choose_candidate(
    policies: Sequence[Policy | DatalogPolicy],
    validate: Callable[[Policy | DatalogPolicy], Collection[Problem]],
    stats: MutableMapping[str, Any],
) -> Policy | DatalogPolicy:
    """Validate every equal-cost candidate and return the one that solves the most problems.

    The candidates all have the same feature cost and all satisfy the round's constraints, so
    nothing in the ASP model distinguishes them; coverage on the actual problem set does. Ties
    are broken by rule count (a smaller policy has the better shot at generalizing further --
    see H18 in docs/experiments-log.md, where a 7-rule policy out-covered 26-40-rule ones) and
    then by clingo's own order, so a tie reproduces the single-model path's choice exactly.

    `validate` is expected to be cheap in-loop validation (`validation_iterations`,
    `validation_time_limit`, `validation_max_consecutive_failures`); it is called once per
    candidate, which is what this hypothesis trades for the tie-break. It may also stop early
    when the run's wall-clock budget runs out, in which case its counts are truncated for every
    remaining candidate alike and the caller ends the run right after this returns.
    """
    scored: list[tuple[int, int, int]] = []
    for index, candidate in enumerate(policies):
        coverage = len(validate(candidate))
        scored.append((coverage, -len(candidate.rules), -index))
        log.info(
            "Candidate %d/%d: %d rule(s), solves %d problem(s)",
            index + 1,
            len(policies),
            len(candidate.rules),
            coverage,
        )
    best = max(range(len(policies)), key=lambda index: scored[index])
    stats["optimalModelChosenCoverage"] = scored[best][0]
    stats["optimalModelChosenIndex"] = best
    stats["optimalModelCoverages"] = [coverage for coverage, _, _ in scored]
    stats["optimalModelTieRounds"] = stats.get("optimalModelTieRounds", 0) + 1
    # The number that says whether the hypothesis paid for itself: how many rounds ended on a
    # model other than the one clingo happened to return, i.e. how often the tie-break actually
    # changed the run.
    stats["optimalModelSwitches"] = stats.get("optimalModelSwitches", 0) + (1 if best else 0)
    log.info(
        "Chose candidate %d/%d (%d problem(s) solved) out of coverages %s",
        best + 1,
        len(policies),
        scored[best][0],
        stats["optimalModelCoverages"],
    )
    return policies[best]


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
    # The trajectories of the best policy so far, on the problems it solved and that are in this
    # round's training set (`anchor_policy_labels`; see `FeaturePool._emit_anchors`). The round
    # is solved twice when they turn out to be contradictory -- see the fallback in `solve`.
    anchor_plans: Optional[Mapping[str, Plan]] = None,
    allow_frontier: bool = True,
    wall_deadline: Optional[float] = None,
    max_pool_size: Optional[int] = None,
    # Scores one candidate policy and returns the problems it solves. Only passed by
    # `solve_iteratively` (the final cost-minimization pass runs its own comparison), and only
    # then is more than one optimal model asked for at all: without a validator there is nothing
    # to choose between siblings with, so the extra clingo enumeration would be pure cost.
    validate: Optional[Callable[[Policy | DatalogPolicy], Collection[Problem]]] = None,
) -> tuple[Result, Optional[Policy | DatalogPolicy], list[FrontierState]]:
    """Run one solve round; on success return the policy the round settled on.

    With `optimal_model_limit > 1` and a `validate` callback the round may have several optimal
    models of the *same* cost to choose from -- clingo returns whichever one it happened to prove
    optimal, and on blocks3ops that choice between `c_equal_closure(on, on_g)` and
    `c_equal(on, on_g)` is worth 95/95 versus 28/95 with no static tie-breaker between them (see
    docs/opt-enum-results.md). Each candidate is validated and the one with the highest coverage
    wins; ties go to the smaller policy (fewer rules), then to the model clingo returned first,
    which is exactly what the single-model path would have used.
    """
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
            anchor_plans=anchor_plans,
            allow_frontier=allow_frontier,
            stats=stats,
            wall_deadline=wall_deadline,
            max_pool_size=max_pool_size,
            optimal_model_limit=(config.get("optimal_model_limit") or 1) if validate is not None else 1,
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
        policies, solve_stats, frontier_states = solution
        policy = policies[0]
        stats.update(solve_stats)
        if frontier_states:
            # Not a policy: it assumes the frontier states it selects are solvable. The caller
            # must expand them and re-solve; only a zero-frontier model is acceptable.
            log.info(
                f"Model for {pnames(active_problems)} with max complexity {complexity} relies on"
                f" {len(frontier_states)} transition(s) into unexpanded states, expanding them"
            )
            return Result.FRONTIER, None, frontier_states
        if len(policies) > 1 and validate is not None:
            policy = _choose_candidate(policies, validate, stats)
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
