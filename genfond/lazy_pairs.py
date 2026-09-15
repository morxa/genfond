"""Counterexample-guided (lazy) pairwise separation for the signature-quotiented encoding.

`--type datalog-sig` states the separation constraint as a hitting set per unordered pair of
signature classes with the same action name. Emitting all of them up front is what now dominates
the instance: `Sum |D|` over the pairs is 689k `dist/2` facts at blocks3ops p005-1 and 55M at
p006-1, where the run dies building the instance text. Almost none of those pairs constrain the
optimum -- any selection that separates the hard pairs separates them too.

So the pairs are added on demand instead, in the classic lazy-constraint loop:

1. Ground the base program and the instance *without* any `sig_pair`/`dist` facts, and solve to
   optimality. This is a relaxation: every model of the full problem is a model of it.
2. Read the selection and the good/bad labelling off the model, and ask `LazyPairs` which pairs
   that model violates. None means the relaxed optimum is feasible for the full problem, hence
   optimal for it as well, and the loop is done.
3. Otherwise ground one more batch of violated pairs into the same `clingo.Control` (a fresh
   `#program pairs(k)` instance, see solve_datalog_sig.lp) and solve again.

The relaxation only grows, so the loop terminates: each iteration adds at least one pair that
the previous model violated and the pair set is finite. An UNSAT relaxation proves the full
problem UNSAT. Only pairs actually needed to justify the final cost are ever grounded.

Under a `solve_time_limit` (see `Solver.solve`) step 1 may be cut off. The loop keeps working on
the best model clingo had: the stopping rule is "this model violates no pair", which is a
statement about the model, not about how it was found, so whatever the loop returns is still
feasible for the full problem and still a valid policy. What a cut-off solve costs is the
optimality argument of step 2 -- that only goes through for a relaxed *optimum* -- so the loop
reports `SolveStatus.SATISFIABLE` rather than `OPTIMAL` in that case. A solve cut off before its
first model yields `UNKNOWN`, which is not a refutation of anything.
"""

import logging
import time
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from .action_signatures import ActionSignature, LazyPairs
from .forced_labels import ForcedLabels
from .solver import Solver, SolveStatus

log = logging.getLogger("genfond.lazy_pairs")

DEFAULT_BATCH = 5000


def _selected(solution: Mapping[str, Any], key: str) -> set[str]:
    return {str(element).strip('"') for element in solution.get(key, set())}


def _seed_forced_pairs(solver: Solver, pairs: LazyPairs, forced: ForcedLabels, batch_size: int) -> tuple[int, int]:
    """Ground the pairs between a forced-good and a forced-bad class before the first solve.

    These pairs constrain *every* model, because both of their classes carry the same label in
    every model (see `forced_labels`), so they are exactly the part of the separation layer that
    does not have to wait for a counterexample. Without them the first relaxed solve sees no
    separation constraint at all and returns the cheapest possible selection, which then has to
    be walked back one batch of violated pairs at a time.

    Note that seeding is sound whatever the forced-label analysis concludes: a `sig_pair`/`dist`
    fact is part of the full eager encoding regardless, and its constraint only fires on a model
    that actually labels K1 good and K2 bad. Only the `forced_good`/`forced_bad` facts in the
    instance itself rely on the analysis being right.

    Returns (violated pairs found, pairs grounded).
    """
    if not forced.good_signatures or not forced.bad_signatures:
        return 0, 0
    # An empty selection separates nothing, so every good x bad pair of a class group counts as
    # violated and the batch is the `batch_size` smallest distinguishing sets among them.
    violated, batch = pairs.violated_pairs(
        [], [], [], good=forced.good_signatures, bad=forced.bad_signatures, limit=batch_size
    )
    if batch:
        solver.add_pairs(0, pairs.facts(0, batch))
    log.info(
        f"Lazy pairs: seeded batch 0 from the forced labels -- {violated} pair(s) between a forced"
        f" good and a forced bad class, {len(batch)} grounded, {pairs.num_dist_facts} dist fact(s)"
    )
    return violated, len(batch)


def solve_with_lazy_pairs(
    solver: Solver,
    signatures: Sequence[ActionSignature],
    batch_size: int = DEFAULT_BATCH,
    stats: Optional[MutableMapping[str, Any]] = None,
    forced: Optional[ForcedLabels] = None,
) -> SolveStatus:
    """Solve, adding violated separation pairs until the model satisfies them all.

    Returns the status of the round. `OPTIMAL` means `solver.solution` holds a genuinely
    optimal model of the full problem, `SATISFIABLE` that it holds a feasible one whose cost
    was never proved optimal (only possible under a `solve_time_limit`), `UNSATISFIABLE` that
    the full problem has no model at all, and `UNKNOWN` that the budget ran out before any
    model was found -- which refutes nothing.

    With a time limit an intermediate solve may be cut off. Two cases:

    * It has a model. The model is a model of the current relaxation, so the counterexample
      scan applies to it unchanged and the loop proceeds exactly as before; it is simply a
      weaker (more expensive) point to cut from. Because the loop only stops once a model
      violates *no* pair, whatever it returns is feasible for the full problem -- i.e. a valid
      policy -- regardless of how many solves were cut off on the way. What is lost is
      optimality: a relaxed optimum that is feasible is a full optimum, but a merely feasible
      relaxed model says nothing about the full optimum, so the final status is `OPTIMAL` only
      when the *last* solve proved its own cost optimal.
    * It has no model. Then nothing is known: it is not a refutation, since the relaxation may
      well be satisfiable and the solver just did not get there. The loop reports `UNKNOWN`.
    """
    pairs = LazyPairs(signatures)
    total_pairs = pairs.index.num_pairs()
    if forced is not None:
        seeded_violated, seeded = _seed_forced_pairs(solver, pairs, forced, batch_size)
        if stats is not None:
            stats["lazyPairsSeeded"] = seeded
            stats["lazyPairsSeededViolated"] = seeded_violated
    for iteration in range(1, len(signatures) ** 2 + 2):
        start = time.perf_counter()
        solver.solve()
        elapsed = time.perf_counter() - start
        if solver.status == SolveStatus.UNSATISFIABLE:
            # Every model of the full problem is a model of the relaxation, so an unsatisfiable
            # relaxation settles the full problem too.
            log.info(f"Lazy pairs: iteration {iteration} unsatisfiable after {pairs.num_pairs_emitted} pair(s)")
            _record(stats, iteration, pairs, total_pairs, optimal=True)
            return SolveStatus.UNSATISFIABLE
        if solver.status == SolveStatus.UNKNOWN:
            # Cut off before the first model of the relaxation: no refutation, no labelling to
            # scan, nothing to add. The round is inconclusive.
            log.info(
                f"Lazy pairs: iteration {iteration} ran out of its time budget before any model,"
                f" after {pairs.num_pairs_emitted} pair(s); round is inconclusive"
            )
            _record(stats, iteration, pairs, total_pairs, optimal=False)
            return SolveStatus.UNKNOWN
        solution = solver.solution
        violated, batch = pairs.violated_pairs(
            _selected(solution, "f_selected"),
            _selected(solution, "c_selected"),
            _selected(solution, "r_selected"),
            good={identifier for identifier, _ in solution.get("sig_action", set())},
            bad=solution.get("bad_sig", set()),
            limit=batch_size,
        )
        facts = pairs.facts(iteration, batch) if batch else ""
        log.info(
            f"Lazy pairs: iteration {iteration}, cost {solver.cost}, {violated} violated pair(s) of"
            f" {total_pairs}, adding {len(batch)}, {pairs.num_pairs_emitted} pair(s) and"
            f" {pairs.num_dist_facts} dist fact(s) grounded, solved in {elapsed:.2f}s"
            f'{"" if solver.optimal else " (not proven optimal)"}'
        )
        if not violated:
            log.info(
                f"Lazy pairs: done after {iteration} iteration(s), cost {solver.cost},"
                f' {"proven optimal" if solver.optimal else "NOT proven optimal (time budget)"}'
            )
            _record(stats, iteration, pairs, total_pairs, optimal=solver.optimal)
            return SolveStatus.OPTIMAL if solver.optimal else SolveStatus.SATISFIABLE
        assert batch, "a violated pair must be addable, otherwise the loop cannot make progress"
        solver.add_pairs(iteration, facts)
    raise RuntimeError("Lazy pair loop did not terminate")


def _record(
    stats: Optional[MutableMapping[str, Any]],
    iterations: int,
    pairs: LazyPairs,
    total_pairs: int,
    optimal: bool,
) -> None:
    if stats is None:
        return
    stats["lazyPairIterations"] = iterations
    stats["lazyPairsOptimal"] = optimal
    stats["lazyPairsGrounded"] = pairs.num_pairs_emitted
    stats["lazyPairsTotal"] = total_pairs
    stats["lazyDistFacts"] = pairs.num_dist_facts
