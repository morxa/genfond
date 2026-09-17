import enum
import logging
import os
import os.path
import time
from typing import Any, Mapping, Optional, Sequence

import clingo

from .cost_utils import feature_cost, prune_cost
from .problem_iterator import MAX_COST
from .shutdown import stop_requested

log = logging.getLogger(__name__)

# How often the async wait loop in Solver.solve checks for a deadline or a stop request. Short
# enough that a SIGTERM (genfond.bash gives SLURM jobs a 10-minute warning via
# --signal=B:TERM@600) or an exhausted --max-wall-time budget takes effect promptly, long enough
# that polling overhead is negligible next to an actual clingo solve.
POLL_INTERVAL = 1.0


class SolveStatus(enum.Enum):
    """Outcome of a single clingo solve.

    `OPTIMAL` and `UNSATISFIABLE` are proofs; `SATISFIABLE` and `UNKNOWN` are what a solve that
    ran out of its wall-clock budget can leave behind (a best-so-far model, or nothing at all).
    Only a `solve_time_limit` can produce the latter two, so a run without one sees exactly the
    two outcomes this code saw before.
    """

    OPTIMAL = 0
    SATISFIABLE = 1
    UNSATISFIABLE = 2
    UNKNOWN = 3


def convert_arg(symbol: clingo.Symbol) -> str | int:
    if symbol.type == clingo.SymbolType.Number:
        return symbol.number
    elif symbol.type == clingo.SymbolType.String:
        return symbol.string
    elif symbol.type == clingo.SymbolType.Function:
        return str(symbol)
    else:
        raise ValueError(f"Unknown symbol type: {symbol.type} for {symbol}")


def extract_solution(model: clingo.Model) -> dict:
    """The shown atoms of `model` as the `{predicate: set of argument tuples}` dict the rest of
    the code calls a "solution", plus its `cost`.

    Factored out of `Solver.on_model` so that `Solver.enumerate_optimal` can build the same dict
    for every model it enumerates without any of them becoming *the* incumbent solution.
    """
    solution: dict = dict()
    for symbol in model.symbols(shown=True):
        args = [convert_arg(arg) for arg in symbol.arguments]
        if len(args) == 0:
            solution.setdefault(symbol.name, set()).add(True)
        elif len(args) == 1:
            solution.setdefault(symbol.name, set()).add(args[0])
        else:
            solution.setdefault(symbol.name, set()).add(tuple([convert_arg(arg) for arg in symbol.arguments]))
    solution["cost"] = model.cost
    return solution


def solution_key(solution: Mapping[str, Any]) -> tuple:
    """A hashable identity for a solution dict, for deduplicating enumerated models.

    `repr` of each element rather than the element itself: a predicate's argument set mixes ints,
    strings and tuples, which do not sort against each other.
    """
    return tuple(
        (name, tuple(sorted(repr(value) for value in values)))
        for name, values in sorted(solution.items())
        if name != "cost"
    )


class Solver:

    def __init__(
        self,
        asp_code: str,
        num_threads: Optional[int] = None,
        solve_prog: str = "solve.lp",
        max_cost: int = MAX_COST,
        max_prune_cost: int = MAX_COST,
        min_feature_complexity: Optional[int] = None,
        opt_strategy: str = "bb",
        clingo_options: Optional[Sequence[str]] = None,
        time_limit: Optional[float] = None,
        minimize_good_signatures: str = "none",
        minimize_selected_count: str = "none",
        wall_deadline: Optional[float] = None,
        plan_label_heuristic: bool = False,
        anchors: bool = False,
    ):
        self.asp_code = asp_code
        self.minimize_good_signatures = minimize_good_signatures
        self.minimize_selected_count = minimize_selected_count
        self.opt_strategy = opt_strategy or "bb"
        self.time_limit = time_limit
        # An absolute `time.perf_counter()`-based deadline (see --max-wall-time /
        # genfond.shutdown), independent of and composable with `time_limit`: the effective
        # per-solve budget is whichever of the two runs out first (computed fresh in `solve()`,
        # since the same Control is solved repeatedly in the lazy-pairs loop and the remaining
        # wall budget shrinks between those calls even though `wall_deadline` itself does not).
        self.wall_deadline = wall_deadline
        options = list(clingo_options or [])
        if plan_label_heuristic:
            # #heuristic directives are inert unless clingo's domain heuristic is switched on.
            # A decision heuristic biases the search only; the set of models and the optimum are
            # unchanged, so this is safe to add whenever the part is grounded.
            options.append("--heuristic=Domain")
        if self.opt_strategy != "bb":
            # "bb" is clingo's own default, so leaving the option off reproduces the previous
            # command line exactly; anything else is passed through verbatim (clingo accepts
            # "usc", but also qualified forms such as "usc,oll,disjoint" or "bb,dec").
            options.append(f"--opt-strategy={self.opt_strategy}")
        self.control = clingo.Control(options)
        self.control.load(os.path.join(os.path.dirname(__file__), solve_prog))
        self.control.add("instances", [], asp_code)
        parts: list[tuple[str, list[clingo.Symbol]]] = [("base", []), ("instances", [])]
        if max_cost < MAX_COST:
            parts.append(("limit_feature_cost", [clingo.Number(max_cost)]))
        if max_prune_cost < MAX_COST:
            parts.append(("limit_prune_cost", [clingo.Number(max_prune_cost)]))
        if min_feature_complexity:
            parts.append(("min_feature_complexity", [clingo.Number(min_feature_complexity)]))
        if minimize_good_signatures == "below":
            parts.append(("minimize_good_sigs_below", []))
        elif minimize_good_signatures == "above":
            parts.append(("minimize_good_sigs_above", []))
        elif minimize_good_signatures != "none":
            raise ValueError(f"Unknown minimize_good_signatures: {minimize_good_signatures!r}")
        if minimize_selected_count == "below":
            parts.append(("minimize_selected_count_below", []))
        elif minimize_selected_count == "above":
            parts.append(("minimize_selected_count_above", []))
        elif minimize_selected_count != "none":
            raise ValueError(f"Unknown minimize_selected_count: {minimize_selected_count!r}")
        if plan_label_heuristic:
            parts.append(("plan_heuristic", []))
        if anchors:
            # `:- anchor(I,S,A), not good_action(I,S,A).` (solve_datalog*.lp). The anchor/3 facts
            # are part of the instance either way; leaving this part out is what makes the same
            # instance solvable again without them -- the fallback re-solve of an anchored round
            # builds a second Solver over the very same `asp_code` with anchors=False.
            parts.append(("anchor", []))
        self.control.ground(parts)
        assert isinstance(self.control.configuration.solve, clingo.Configuration)
        self.control.configuration.solve.parallel_mode = num_threads or os.cpu_count()
        self.solution: dict = dict()
        # Further optimal models of the same cost as `solution`, collected by
        # `enumerate_optimal()`; empty until it runs (so `candidates or [solution]` is the one
        # place callers have to look).
        self.candidates: list[dict] = []
        # How many distinct optimal models `enumerate_optimal()` actually found; `candidates`
        # may be filtered down afterwards (the lazy-pairs loop drops the infeasible ones), so
        # the count is kept separately for the stats.
        self.num_enumerated = 0
        self.cost: list[int] = []
        self.statistics: dict = dict()
        self.status = SolveStatus.UNKNOWN
        # Whether the last solve *proved* its answer: an optimal model, or unsatisfiability.
        self.optimal = False
        self.timed_out = False
        self.elapsed = 0.0

    @property
    def feature_complexity(self) -> int:
        """`self.cost`'s feature/concept/role complexity component; see `feature_cost()`."""
        return feature_cost(self.cost, self.minimize_good_signatures, self.minimize_selected_count)

    @property
    def prune_count(self) -> int:
        """`self.cost`'s frontier-transition component; see `prune_cost()`.

        0 both when the model uses no frontier transition and when the instance has no reachable
        `pruned/2` state at all, in which case the level is missing from the vector entirely.
        """
        return prune_cost(self.cost, self.minimize_good_signatures, self.minimize_selected_count)

    def on_model(self, model: clingo.Model) -> None:
        if not self.solution:
            log.info("Found first solution")
        self.solution = extract_solution(model)
        self.cost = model.cost

    def add_pairs(self, batch: int, facts: str) -> None:
        """Ground one more batch of lazy separation pairs into the running control.

        The facts go into a part of their own and the rules that consume them into a fresh
        instance of ``#program pairs(k)`` (see solve_datalog_sig.lp), so the rules ground against
        this batch only. Both are grounded in one call, facts first.
        """
        name = f"pair_facts_{batch}"
        self.control.add(name, [], facts)
        self.control.ground([(name, []), ("pairs", [clingo.Number(batch)])])

    def solve(self, bound: Optional[Sequence[int]] = None) -> bool:
        """Solve the grounded program; return whether a model was found.

        With `time_limit` and/or `wall_deadline` set the solve is anytime: it is cancelled once
        the effective budget (the sooner of the two) is up, keeping the best model clingo
        reported so far. `self.status` then distinguishes the four outcomes and `self.optimal`
        says whether the answer was proved. A cancelled solve that never produced a model is
        `UNKNOWN`, *not* unsatisfiable -- the caller must not read a refutation into it.

        The solve always runs asynchronously, polling for completion in `POLL_INTERVAL` steps,
        even with no budget at all: that is the only way a `genfond.shutdown.stop_requested()`
        signal (SIGINT/SIGTERM) can reach a solve already in flight, since a plain blocking
        `control.solve()` call would not return control to Python -- where a signal handler
        actually runs -- until clingo itself finishes. With no budget and no stop request this
        polling loop only ever exits once the solve completes on its own, so it settles the same
        question a single blocking call would, just observed in small steps.
        """
        # Constraints added since the previous solve can only raise the optimum, so a bound
        # carried over from it would be unsound. clingo does not keep one, but say so anyway.
        assert isinstance(self.control.configuration.solve, clingo.Configuration)
        self.control.configuration.solve.opt_mode = (
            "opt" if bound is None else "opt," + ",".join(str(int(level)) for level in bound)
        )
        # A previous solve's model must not be mistaken for this one's best-so-far: in the lazy
        # loop the same Control is solved repeatedly, and a solve that is cut off before its
        # first model has to report "no model", not the model of the round before.
        self.solution = dict()
        self.cost = []
        start = time.perf_counter()
        # The effective deadline for *this* solve: the sooner of the configured time_limit and
        # the remaining wall-clock budget. Recomputed from wall_deadline (an absolute instant)
        # on every call, not just once at construction time, since the lazy-pairs loop solves
        # the same Control repeatedly and the remaining budget shrinks between those calls even
        # though wall_deadline itself is fixed.
        stop_at: Optional[float] = None if self.time_limit is None else start + self.time_limit
        if self.wall_deadline is not None:
            stop_at = self.wall_deadline if stop_at is None else min(stop_at, self.wall_deadline)
        # Computed from `start`, not a fresh `time.perf_counter()`, so the first wait() below is
        # the with-block's first statement, exactly like the plain `handle.wait(self.time_limit)`
        # this replaces: an unbounded (solve_time_limit=0) hard-optimisation instance can return
        # its first (non-optimal) model within microseconds, so any extra work inserted before
        # that first wait() measurably changes how often it wins the race -- see
        # docs/wall-budget-results.md.
        step = POLL_INTERVAL if stop_at is None else max(0.0, min(POLL_INTERVAL, stop_at - start))
        finished = False
        with self.control.solve(on_model=self.on_model, async_=True) as handle:
            while True:
                finished = handle.wait(step)
                if finished:
                    break
                if (stop_at is not None and time.perf_counter() >= stop_at) or stop_requested():
                    break
                now = time.perf_counter()
                step = POLL_INTERVAL if stop_at is None else max(0.0, min(POLL_INTERVAL, stop_at - now))
            if not finished:
                handle.cancel()
            res = handle.get()
        self.timed_out = not finished
        self.elapsed = time.perf_counter() - start
        self.statistics = self.control.statistics
        if self.solution:
            # A solve that ran to completion settles the question, whether or not clingo reports
            # the search space as exhausted: with a #minimize it proved the optimum, and without
            # one it stops at the first model, which is then trivially optimal. Only a cancelled
            # solve leaves a model that is merely the best seen so far.
            proved = res.exhausted or not self.timed_out
            self.status = SolveStatus.OPTIMAL if proved else SolveStatus.SATISFIABLE
        elif res.satisfiable is False:
            self.status = SolveStatus.UNSATISFIABLE
        else:
            # No model and no proof of unsatisfiability: the budget ran out first, or a stop was
            # requested, before any model was found. Without a budget and without a stop request
            # this cannot happen, and the `assert res.satisfiable is not None` below still
            # reports it exactly as it did before -- no new failure mode on that path.
            self.status = SolveStatus.UNKNOWN
        self.optimal = self.status in (SolveStatus.OPTIMAL, SolveStatus.UNSATISFIABLE)
        log.info(
            "clingo solve [opt-strategy=%s%s%s]: %s in %.2fs%s, cost %s",
            self.opt_strategy,
            f", limit={self.time_limit:g}s" if self.time_limit is not None else "",
            f", warm bound={list(bound)}" if bound is not None else "",
            self.status.name,
            self.elapsed,
            " (timed out)" if self.timed_out else "",
            self.cost if self.cost else "-",
        )
        if finished:
            # The async wait resolved on its own -- not cut off by a deadline or a stop request
            # -- so clingo settled the question one way or the other, exactly as a plain
            # blocking solve() always did (the case this reproduces when stop_at is None and no
            # stop was ever requested).
            assert res.satisfiable is not None
            return res.satisfiable
        return bool(self.solution)

    def enumerate_optimal(self, limit: int) -> list[dict]:
        """Collect up to `limit` optimal models of the already-solved program.

        The caller must have run `solve()` to a *proven* optimum first: the enumeration only
        makes sense relative to a known optimal cost, and clingo's `optN` mode needs to redo the
        optimisation phase on the same `Control` anyway (it is fast there, since the solve that
        proved the optimum left its learnt clauses behind).

        Why this exists: on blocks3ops two models of cost 3 with one selected element each --
        `c_equal_closure(on, on_g)` and `c_equal(on, on_g)` -- decide the whole run (95/95 vs.
        28/95), and no static tie-breaker tells them apart. The loop already validates every
        candidate policy on all problems, so coverage is the natural tie-breaker; it just never
        had more than one candidate per round to apply it to.

        `limit <= 1` is a no-op returning just the incumbent, so the default configuration never
        runs a second solve at all. Otherwise `opt_mode` is switched to `optN`, which first
        re-establishes the optimum and then enumerates models *of that cost*; clingo reports
        those with `optimality_proven` set (the one re-reported model of the optimisation phase
        has it clear), and `configuration.solve.models = limit` counts exactly the enumerated
        ones. A program with no `#minimize` at all has no optimisation phase and no
        `optimality_proven` flag -- every model is trivially optimal there, so those are taken as
        they come.

        The enumeration as a whole gets the same budget a single solve gets (`time_limit`, and
        whatever is left of `wall_deadline`); running out of it simply returns fewer candidates.
        The incumbent is always first in the returned list, so a caller that cannot afford to
        look past it -- or an enumeration that timed out before its first model -- still sees the
        exact model the single-model path would have produced.
        """
        self.candidates = [self.solution] if self.solution else []
        self.num_enumerated = len(self.candidates)
        if limit <= 1 or not self.solution or not self.optimal:
            return self.candidates
        assert isinstance(self.control.configuration.solve, clingo.Configuration)
        known_cost = list(self.cost)
        models: list[dict] = []

        def on_model(model: clingo.Model) -> None:
            if list(model.cost) != known_cost:
                # Cannot normally happen (optN only enumerates at the optimum, and the one
                # optimisation-phase model it re-reports is the incumbent), but a model of a
                # different cost is not a tie and must never become a candidate.
                return
            if model.cost and not model.optimality_proven:
                return
            models.append(extract_solution(model))

        previous_models = self.control.configuration.solve.models
        self.control.configuration.solve.opt_mode = "optN"
        self.control.configuration.solve.models = limit
        start = time.perf_counter()
        stop_at: Optional[float] = None if self.time_limit is None else start + self.time_limit
        if self.wall_deadline is not None:
            stop_at = self.wall_deadline if stop_at is None else min(stop_at, self.wall_deadline)
        # The polling loop of `solve()`, repeated rather than shared: that one is written so that
        # its first `wait()` is the very first statement inside the `with`, because an unbounded
        # hard-optimisation instance can produce its first model within microseconds and any work
        # inserted before that wait measurably changes the race (see `solve()`'s comment and
        # docs/wall-budget-results.md). Refactoring both onto one helper would put that property
        # at the mercy of a later edit here.
        step = POLL_INTERVAL if stop_at is None else max(0.0, min(POLL_INTERVAL, stop_at - start))
        finished = False
        try:
            with self.control.solve(on_model=on_model, async_=True) as handle:
                while True:
                    finished = handle.wait(step)
                    if finished:
                        break
                    if (stop_at is not None and time.perf_counter() >= stop_at) or stop_requested():
                        break
                    now = time.perf_counter()
                    step = POLL_INTERVAL if stop_at is None else max(0.0, min(POLL_INTERVAL, stop_at - now))
                if not finished:
                    handle.cancel()
        finally:
            # A later `solve()` on this Control (the lazy-pairs loop solves the same one over and
            # over) must see the configuration it always saw; `solve()` resets opt_mode itself,
            # but nothing there resets the model count.
            self.control.configuration.solve.opt_mode = "opt"
            self.control.configuration.solve.models = previous_models
        elapsed = time.perf_counter() - start
        seen = {solution_key(self.solution)}
        for solution in models:
            key = solution_key(solution)
            if key in seen:
                continue
            seen.add(key)
            self.candidates.append(solution)
        self.num_enumerated = len(self.candidates)
        log.info(
            "Enumerated %d optimal model(s) of cost %s in %.2fs (limit %d%s)",
            len(self.candidates),
            known_cost,
            elapsed,
            limit,
            "" if finished else ", cut off",
        )
        return self.candidates
