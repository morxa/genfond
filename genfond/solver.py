import enum
import logging
import os
import os.path
import time
from typing import Optional, Sequence

import clingo

from .cost_utils import feature_cost
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
        self.control.ground(parts)
        assert isinstance(self.control.configuration.solve, clingo.Configuration)
        self.control.configuration.solve.parallel_mode = num_threads or os.cpu_count()
        self.solution: dict = dict()
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

    def on_model(self, model: clingo.Model) -> None:
        if not self.solution:
            log.info("Found first solution")
        self.solution = dict()
        for symbol in model.symbols(shown=True):
            args = [convert_arg(arg) for arg in symbol.arguments]
            if len(args) == 0:
                self.solution.setdefault(symbol.name, set()).add(True)
            elif len(args) == 1:
                self.solution.setdefault(symbol.name, set()).add(args[0])
            else:
                self.solution.setdefault(symbol.name, set()).add(tuple([convert_arg(arg) for arg in symbol.arguments]))
        self.solution["cost"] = model.cost
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

    def solve(self) -> bool:
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
        self.control.configuration.solve.opt_mode = "opt"
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
            "clingo solve [opt-strategy=%s%s]: %s in %.2fs%s, cost %s",
            self.opt_strategy,
            f", limit={self.time_limit:g}s" if self.time_limit is not None else "",
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
