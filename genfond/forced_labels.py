"""Labels that every model of the signature encoding agrees on, computed before the search.

`solve_datalog_sig.lp` has two choice layers: which transitions are good
(``1 { good_trans(I,S,A,S2) : trans(I,S,A,S2) } :- alive(I,S), not goal(I,S).``) and which
elements are selected. The separation constraints sit behind both -- their bodies mention
``good_sig``/``bad_sig``, which are derived from the first choice -- so nothing about pair
separation is decided before search starts. Till's ASP rule of thumb is to move choices as far
back as possible, and a label that is the same in *every* model is not a choice at all.

This module computes, purely from the graph layer of the instance, the set of (instance, state,
action) occurrences that are good in every model and the set that is bad in every model, plus the
signature classes those force. `feature_generator` emits them as ``forced_good/3`` and
``forced_bad/3``; `solve_datalog_sig.lp` drops the forced-bad transitions from the choice rule
and constrains the forced-good ones.

Soundness -- each step is a consequence of the program itself, so no model is lost:

1. **Non-candidate occurrences are bad.** ``:- good_trans(I,S,A,_), trans(I,S,A,S2), not
   alive(I,S2), not pruned(I,S2).`` forbids selecting an action with an outcome that is neither
   alive nor pruned (a dead end, or a state the instance dropped). So if any outcome of A in S is
   not alive and not pruned, ``good_trans(I,S,A,_)`` is false in every model, hence
   ``good_action(I,S,A)`` is false in every model and, since S is alive and non-goal,
   ``bad_sig(K)`` holds in every model for A's class K.
2. **A bad class makes all its occurrences bad.** ``:- good_sig(K), bad_sig(K).`` So once
   ``bad_sig(K)`` holds in every model, ``good_sig(K)`` is false in every model; with
   ``good_sig(K) :- good_action(I,S,A), asig(I,S,A,K).`` every occurrence of K at an alive
   non-goal state has ``good_action`` false in every model -- i.e. it is forced bad too, wherever
   it occurs. This is the step that crosses states.
3. **A state whose surviving candidates all share one class forces that class good.** The choice
   rule requires at least one good transition per alive non-goal state. If every action at S that
   is not already forced bad has class K, then whichever one the model picks derives
   ``good_sig(K)``, so ``good_sig(K)`` holds in every model. (|remaining| = 1 is the special
   case.)
4. **A good class makes all its occurrences good.** ``good_sig(K)`` in every model means
   ``bad_sig(K)`` is false in every model (same constraint as step 2), and the body of the
   ``bad_sig`` rule then forces ``good_action(I,S,A)`` true at every alive non-goal occurrence of
   K.

Steps 2-4 feed each other, so the whole thing is run to a fixpoint. It is monotone (labels are
only ever added) over a finite set, so it terminates.

If the fixpoint ever labels one class both good and bad, or forces a non-candidate occurrence
good, the instance has no model at all. That is reported as `inconsistent`; the caller still
emits the facts, and the ASP program then reports UNSAT by itself (an empty choice, or the
``:- good_sig(K), bad_sig(K)`` constraint), so the refutation never rests on this analysis alone.
"""

import logging
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

log = logging.getLogger("genfond.forced_labels")


class ActionOccurrence(NamedTuple):
    """One ``asig(I, S, A, K)`` fact at an alive, non-goal state, plus whether it is selectable.

    `candidate` is false exactly when some outcome of A in S is neither ``alive`` nor ``pruned``,
    which is what makes ``good_trans`` unselectable for it (step 1 above).
    """

    instance: int
    state: int
    action: str
    signature: int
    candidate: bool


@dataclass
class ForcedLabels:
    """The occurrences and classes whose label is the same in every model."""

    good: set[ActionOccurrence] = field(default_factory=set)
    bad: set[ActionOccurrence] = field(default_factory=set)
    good_signatures: set[int] = field(default_factory=set)
    bad_signatures: set[int] = field(default_factory=set)
    # The fixpoint derived a contradiction: the instance is unsatisfiable.
    inconsistent: bool = False
    # Totals the fractions are reported against.
    num_occurrences: int = 0
    num_signatures: int = 0

    def summary(self) -> str:
        def share(part: int, whole: int) -> str:
            return f"{part}/{whole} ({100.0 * part / whole:.1f}%)" if whole else f"{part}/0"

        return (
            f"forced {share(len(self.good), self.num_occurrences)} occurrence(s) good and "
            f"{share(len(self.bad), self.num_occurrences)} bad; "
            f"{share(len(self.good_signatures), self.num_signatures)} signature class(es) good and "
            f"{share(len(self.bad_signatures), self.num_signatures)} bad"
            f'{"; INCONSISTENT (instance is unsatisfiable)" if self.inconsistent else ""}'
        )


def compute_forced_labels(occurrences: Iterable[ActionOccurrence]) -> ForcedLabels:
    """Run the fixpoint of the four rules in the module docstring.

    `occurrences` must contain exactly the ``asig/4`` facts of alive, non-goal states -- those are
    the only ones the ``good_trans`` choice and the ``good_sig``/``bad_sig`` rules can see.
    """
    occurrences = list(occurrences)
    result = ForcedLabels(num_occurrences=len(occurrences))
    by_signature: dict[int, list[ActionOccurrence]] = dict()
    # Per state: the occurrences not yet known bad, and how many of them each class still has.
    remaining: dict[tuple[int, int], set[ActionOccurrence]] = dict()
    remaining_signatures: dict[tuple[int, int], Counter[int]] = dict()
    for occurrence in occurrences:
        by_signature.setdefault(occurrence.signature, []).append(occurrence)
        key = (occurrence.instance, occurrence.state)
        remaining.setdefault(key, set()).add(occurrence)
        remaining_signatures.setdefault(key, Counter())[occurrence.signature] += 1
    result.num_signatures = len(by_signature)

    bad_queue: list[int] = []
    good_queue: list[int] = []

    def mark_bad_signature(signature: int) -> None:
        if signature not in result.bad_signatures:
            result.bad_signatures.add(signature)
            bad_queue.append(signature)

    def mark_good_signature(signature: int) -> None:
        if signature not in result.good_signatures:
            result.good_signatures.add(signature)
            good_queue.append(signature)

    def mark_bad(occurrence: ActionOccurrence) -> None:
        if occurrence in result.bad:
            return
        result.bad.add(occurrence)
        mark_bad_signature(occurrence.signature)
        key = (occurrence.instance, occurrence.state)
        left = remaining[key]
        left.discard(occurrence)
        counts = remaining_signatures[key]
        counts[occurrence.signature] -= 1
        if counts[occurrence.signature] == 0:
            del counts[occurrence.signature]
        if not left:
            # The choice rule demands one good transition here and nothing is left to pick.
            result.inconsistent = True
        elif len(counts) == 1:
            mark_good_signature(next(iter(counts)))

    def mark_good(occurrence: ActionOccurrence) -> None:
        if occurrence in result.good:
            return
        result.good.add(occurrence)
        if not occurrence.candidate:
            result.inconsistent = True
        mark_good_signature(occurrence.signature)

    for occurrence in occurrences:
        if not occurrence.candidate:
            mark_bad(occurrence)
    # A state whose classes already collapse to one before anything is marked bad (step 3 with an
    # empty bad set) forces that class good as well.
    for key, counts in remaining_signatures.items():
        if len(counts) == 1 and remaining[key]:
            mark_good_signature(next(iter(counts)))
    while bad_queue or good_queue:
        while bad_queue:
            for occurrence in by_signature[bad_queue.pop()]:
                mark_bad(occurrence)
        while good_queue:
            for occurrence in by_signature[good_queue.pop()]:
                mark_good(occurrence)
    if result.good_signatures & result.bad_signatures:
        result.inconsistent = True
    return result
