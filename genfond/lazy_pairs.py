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
"""

import logging
import time
from typing import Any, Mapping, MutableMapping, Optional, Sequence

from .action_signatures import ActionSignature, LazyPairs
from .solver import Solver

log = logging.getLogger("genfond.lazy_pairs")

DEFAULT_BATCH = 5000


def _selected(solution: Mapping[str, Any], key: str) -> set[str]:
    return {str(element).strip('"') for element in solution.get(key, set())}


def solve_with_lazy_pairs(
    solver: Solver,
    signatures: Sequence[ActionSignature],
    batch_size: int = DEFAULT_BATCH,
    stats: Optional[MutableMapping[str, Any]] = None,
) -> bool:
    """Solve to optimality, adding violated separation pairs until the model satisfies them all.

    Returns whether the (full) problem is satisfiable; on success `solver.solution` holds the
    final, genuinely optimal model, so policy extraction runs on it unchanged.
    """
    pairs = LazyPairs(signatures)
    total_pairs = pairs.index.num_pairs()
    for iteration in range(1, len(signatures) ** 2 + 2):
        start = time.perf_counter()
        satisfiable = solver.solve()
        elapsed = time.perf_counter() - start
        if not satisfiable:
            # Every model of the full problem is a model of the relaxation, so an unsatisfiable
            # relaxation settles the full problem too.
            log.info(f"Lazy pairs: iteration {iteration} unsatisfiable after {pairs.num_pairs_emitted} pair(s)")
            _record(stats, iteration, pairs, total_pairs)
            return False
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
        )
        if not violated:
            _record(stats, iteration, pairs, total_pairs)
            return True
        assert batch, "a violated pair must be addable, otherwise the loop cannot make progress"
        solver.add_pairs(iteration, facts)
    raise RuntimeError("Lazy pair loop did not terminate")


def _record(stats: Optional[MutableMapping[str, Any]], iterations: int, pairs: LazyPairs, total_pairs: int) -> None:
    if stats is None:
        return
    stats["lazyPairIterations"] = iterations
    stats["lazyPairsGrounded"] = pairs.num_pairs_emitted
    stats["lazyPairsTotal"] = total_pairs
    stats["lazyDistFacts"] = pairs.num_dist_facts
