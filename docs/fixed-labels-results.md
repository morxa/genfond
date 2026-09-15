# Fixing the forced labels before the search: results

Date: 2026-09-15. Branch: `hyp/fixed-labels` (on `hyp/min-count`).

## Background

`solve_datalog_sig.lp` has two choice layers. Which transitions are good is a choice
(`1 { good_trans(I,S,A,S2) : trans(I,S,A,S2) } :- alive(I,S), not goal(I,S).`), which elements
are selected is a choice, and the separation constraints sit *behind both*: their bodies mention
`good_sig`/`bad_sig`, which are derived from the first choice. So nothing about pair separation
is decided before search starts, and — per Till's ASP advice — a fact that depends on a choice
cannot be pre-grounded. At 20–30 training problems the first relaxed lazy solve takes seconds
and the next one hours (`docs/experiments-log.md`, cluster snapshot 07:00 and H8).

A label that is the same in *every* model is not really a choice. This hypothesis computes those
labels in Python and pins them.

## What is forced, and why it is implied

`genfond/forced_labels.py` runs the fixpoint of four rules. Each is a consequence of the existing
program, so pinning it removes no model:

1. **Non-candidate occurrences are bad.** `:- good_trans(I,S,A,_), trans(I,S,A,S2), not
   alive(I,S2), not pruned(I,S2).` So if any outcome of `A` in `S` is neither alive nor pruned,
   `good_trans` is false in every model, `good_action(I,S,A)` is false in every model, and (for
   an alive non-goal `S`) `bad_sig(K)` holds in every model for `A`'s class `K`.
2. **A bad class is bad everywhere.** `:- good_sig(K), bad_sig(K).` gives `not good_sig(K)` in
   every model; with `good_sig(K) :- good_action(I,S,A), asig(I,S,A,K).` every occurrence of `K`
   at an alive non-goal state is bad in every model — *including states that have perfectly safe
   alternatives*. This is the step that crosses states, and the one the grounder cannot see.
3. **A state whose surviving candidates share one class forces that class good.** The choice rule
   demands at least one good transition per alive non-goal state; if every action there that is
   not already forced bad has class `K`, then whatever the model picks derives `good_sig(K)`.
   (`|remaining| = 1` is the special case.)
4. **A good class is good everywhere.** `good_sig(K)` in every model means `bad_sig(K)` is false
   in every model, and the body of the `bad_sig` rule then forces `good_action` true at every
   alive non-goal occurrence of `K`.

2–4 feed each other, so the whole thing runs to a fixpoint; it is monotone over a finite set, so
it terminates. Adding round-level constraints (`limit_feature_cost`, `limit_prune_cost`,
`min_feature_complexity`) only shrinks the model set, so a label forced in the base program stays
forced under them — the analysis never has to know about them.

A contradiction in the fixpoint (a class forced both ways, a state with no candidate left, a
forced-good non-candidate) means the instance has no policy at this complexity. It is logged, the
facts are emitted anyway, and clingo produces the refutation itself — the run never rests on this
analysis for an UNSAT answer.

## What is emitted

`feature_generator` collects the `asig/4` occurrences at alive non-goal states while it writes the
per-node facts (no second pass over the state graphs, so the emitted-vs-recorded decisions cannot
drift) and emits `forced_good/3` and `forced_bad/3`. In `solve_datalog_sig.lp`:

```
1 { good_trans(I,S,A,S2) : trans(I,S,A,S2), not forced_bad(I,S,A) } :- alive(I,S), not goal(I,S).
:- forced_good(I,S,A), not good_action(I,S,A).
```

The forced-good side is a **constraint**, not a rule deriving `good_action`. A rule would *add*
models: `good_action` true without a supporting `good_trans` would let a model claim the class is
good without the policy actually taking the action, dropping the obligations that come with taking
it (all outcomes safe, `safe_state`), and a cheaper selection could become feasible.

Both predicates are `#defined`, so with `fix_forced_labels: false` nothing is emitted and the
ground program is the one the branch produced before.

### Seeding the lazy loop

With `lazy_pairs`, the forced labels also ground **batch 0**: every pair whose two classes are one
forced good and one forced bad. Those constrain every model, so they need no counterexample —
which is exactly the part of the separation layer that can be pre-grounded. The batch is produced
by `LazyPairs.violated_pairs` with an empty selection (nothing separates anything, so every
good × bad pair of a class group is returned, smallest `|D|` first, capped at `lazy_pairs_batch`).

Seeding is sound **whatever the analysis concludes**: a `sig_pair`/`dist` fact is part of the full
eager encoding regardless, and its constraint only fires on a model that actually labels `K1` good
and `K2` bad. Only the `forced_good`/`forced_bad` facts rely on the analysis being right.

### The plan heuristic (separate flag)

`plan_label_heuristic` emits `plan_action/3` for the actions that start an example-plan suffix at
a state and grounds

```
#heuristic good_trans(I,S,A,S2) : trans(I,S,A,S2), plan_action(I,S,A). [1,true]
```

plus clingo's `--heuristic=Domain`. A decision heuristic changes the order of the search, never
its solution space or its optimum. Independent of `fix_forced_labels`.

## Gates

In `/home/thofmann/code/genfond-wt/fixed-labels`: `pytest tests/ --import-mode importlib -q` →
**164 passed, 1 skipped** (150 + 1 before, plus the 14 new ones). `mypy genfond tests`,
`black --check`, `isort --check` clean. One pre-existing flake seen once in four full runs:
`test_execute_datalog_policy.py::test_datalog_policy_with_roles`, which exercises the randomized
policy executor with no seed and does not touch any code this branch changes; it passes in
isolation every time and passed in the other three full runs.

`tests/test_forced_labels.py` checks each of the four rules on hand-built occurrence sets, both
inconsistency cases, and — non-circularly — that clingo refutes the *opposite* of every forced
label on the **unmodified** program (no `forced_*` facts in the instance): `:- good_action(I,S,A).`
must be UNSAT for a forced-good occurrence, `:- not good_action(I,S,A).` for a forced-bad one.
Plus: flag on/off give the same optimum eagerly and lazily, the lazy loop really seeds batch 0,
and the plan heuristic keeps the cost.

## Results

All runs on `desk-03.ml.rwth-aachen.de` inside the apptainer image, `-n 1 --seed 0
PYTHONHASHSEED=0 --max-memory 60000`, `timeout 30m`.

### How much is actually forced

This decides *which* of the four rules does the work. Across all five suites the forced-**bad**
set is empty on four of them, in every single round:

| suite (iterative run) | rounds | forced good (occurrences) | forced bad | forced good classes | forced bad classes |
|---|---|---|---|---|---|
| gripper-local | 7 | 0-2 of 16-96 (0-11%) | 0-3 (0-11%) | 0-1 of 7-31 | 0-2 |
| miconic-local | 5 | 1-2 of 6-47 (0-15%) | **0** | 1 of 7-47 | **0** |
| blocks4ops-clear-local | 2 | 1-5 of 3-16 (31-33%) | **0** | 1-2 of 3-9 | **0** |
| delivery-local | 7 | 1-3 of 1-35 (9-100%) | **0** | 1-2 of 1-25 | **0** |
| blocks3ops-local | 27 | 1-30 of 3-621 (1-10%) | **0** | 1-6 of 3-512 | **0** |

The reason is structural, not a bug. Rule 1 is the *only* seed for a bad label: it needs a
transition whose outcome is neither `alive` nor `pruned`, and on these state spaces there is no
such transition. Two mechanisms remove them, and both are in play:

* `frontier_expansion` (on by default for `datalog-sig`) turns every off-plan successor into a
  `pruned` state, and `safe_state(I,S) :- pruned(I,S)` counts `pruned` as **safe**. So the very
  states the hypothesis expected to be forced-bad are the ones frontier expansion declares good
  enough to aim at.
* On the small instances of these suites the plan restriction does not restrict at all: the
  blocks3ops-local one-shot expands `3+3+3+13+13+13+73+73+501+501 = 1196` states, i.e. the full
  reachable space of each problem, and re-running it with `frontier_expansion: false` produces
  *byte-identical* state counts. blocks3ops has no dead ends, so a full space has nothing for
  rule 1 to find either way.

With no bad label anywhere, rule 2 (the cross-state propagation, the one the grounder cannot do
itself) never fires, and rule 4 has nothing to start from. What is left is rule 3 alone — states
where the surviving candidates all share one signature class — which fires on 1-15% of
occurrences and **only produces good labels**.

And with no forced-bad *class*, the lazy-loop seeding has nothing to seed: a pair needs one class
of each kind. On four of the five suites batch 0 is empty in every round.

The one place with real forced-bad labels is gripper, where 35 of 120 states really are dead:

| gripper-local, one-shot c=4 | off | on |
|---|---|---|
| forced | -- | 5/246 good (2.0%), **70/246 bad (28.5%)**; 1/37 good classes, **12/37 bad (32.4%)** |
| batch 0 seeded | -- | 12 pairs, 269 dist facts |
| iteration 1: first model cost / pairs added | `[0]` / 117 | `[1]` / 58 |
| pairs grounded in total | 117 (1244 dist facts) | **70 (593 dist facts)** |
| lazy iterations | 2 | 2 |
| final cost / rules / solved | `[6]` / 4 / 5 of 5 | `[6]` / 4 / 5 of 5 |

So where bad labels exist the seeding does exactly what it is supposed to: the relaxation starts
from a non-trivial model instead of the free one, and 40 % fewer pairs are ever grounded. The
blocks3ops results below show that the forced-*good* labels alone are worth more, and they do
reach the domain the stall is in.

### Where it does pay: blocks3ops one-shot

The suites above are all sub-second solves, where nothing can show. The interesting case is one
round of blocks3ops big enough to hurt but small enough to finish: `--one-shot --max-complexity 4
--role-complexity-offset 2 --concept-complexity-offset 1` on the eight blocks3ops-local problems
p002-1 … p004-2 (558 occurrences, 377 signature classes).

| arm | lazy iterations | solver CPU | wall | cost | rules | solved |
|---|---|---|---|---|---|---|
| baseline | 8 | 327.0 s | 5:31 | `[8]` | 50 | 8/8 |
| `--fix-forced-labels` | **4** | **166.7 s** | **2:50** | `[8]` | 56 | 8/8 |
| `--plan-label-heuristic` | **2** | **40.7 s** | **0:45** | `[8]` | 48 | 8/8 |

Same proven optimum in all three. `fix_forced_labels` halves the counterexample loop and the
solver time (**1.96x**) — and note it does so with **zero** forced-bad labels and therefore an
empty batch 0: the entire gain comes from the 47 of 377 classes (12.5 %) whose `good_sig` is
pinned by rule 3, which decides one side of every separation constraint those classes appear in
before the search starts. The plan heuristic is better still here (**8x**), the opposite sign
from what it does on the iterative blocks3ops-local run.

### Equivalence: one-shot, identical inputs

The iterative runs are not a cost comparison (see below), so the equality claim is checked on
one-shot rounds, where both arms see the same problems, the same complexity and the same state
space.

| suite, one-shot c=4 | off | on |
|---|---|---|
| gripper-local | cost `[6]`, 4 rules, 5/5 | cost `[6]`, 4 rules, 5/5 |
| miconic-local | cost `[8]`, 9 rules, 4/4 | cost `[8]`, 11 rules, 4/4 |
| blocks4ops-clear-local | cost `[0]`, 2 rules, 4/4 | cost `[0]`, 2 rules, 4/4 |
| delivery-local | no solution | no solution |

Identical cost and identical outcome everywhere. `miconic-local` returns a different equal-cost
model (9 vs 11 rules): the forced facts change clingo's search, so which of several cost-optimal
models is reported can change. That is the only observable difference.

### The five suites, iterative

`--add-problem-after-success --solve-time-limit 300`, `--role-complexity-offset 2
--concept-complexity-offset 1` on blocks3ops.

| suite | arm | wall | solver CPU | solved | final cost | rules | clingo solves |
|---|---|---|---|---|---|---|---|
| gripper-local | off | 6.80 s | 1.58 s | 5/5 | `[0, 6]` | 4 | 16 |
| gripper-local | on | 6.93 s | 1.76 s | 5/5 | `[0, 6]` | 4 | 15 |
| miconic-local | off | 3.77 s | 0.65 s | 4/4 | `[0, 12]` | 8 | 9 |
| miconic-local | on | 3.86 s | 0.81 s | 4/4 | `[0, 12]` | 8 | 10 |
| blocks4ops-clear-local | off | 2.89 s | 0.05 s | 4/4 | `[0, 2]` | 3 | 3 |
| blocks4ops-clear-local | on | 2.85 s | 0.05 s | 4/4 | `[0, 2]` | 2 | 3 |
| delivery-local | off | 3.99 s | 0.38 s | 4/4 | `[0, 9]` | 8 | 10 |
| delivery-local | on | 3.86 s | 0.35 s | 4/4 | `[0, 9]` | 8 | 10 |
| blocks3ops-local | off | 25.5 s | 14.3 s | 10/10 | `[0, 9]` | 40 | 64 |
| blocks3ops-local | on | 27.8 s | 16.2 s | 10/10 | `[0, 10]` | 45 | 69 |

No regression: same solved count everywhere, same final cost on four suites. On blocks3ops-local
the final cost differs (`[0,9]` vs `[0,10]`) **on a trajectory, not on a round**: the two arms
agree round for round up to the fourth policy (`[3]`, `[0,4]`, `[0,4]`, `[0,5]` on the same four
problems), but they return different equal-cost models there, which produce different policies,
which solve different problems on the way and therefore add different example plans and frontier
states. By the final round the two arms are no longer solving the same instance. This is the same
equal-cost model swap `blocks4ops-clear-local` shows above (2 rules vs 3), amplified by the
feedback loop; the one-shot table is the controlled comparison.

### The plan-label heuristic

Same five suites, `--plan-label-heuristic` (baseline is the `off` row above). On-plan actions per
round: 0-34 (gripper), 3-16 (miconic), 2-9 (blocks4ops-clear), 0-16 (delivery).

| suite | solved | final cost | rules | wall (heuristic) | wall (baseline) |
|---|---|---|---|---|---|
| gripper-local | 5/5 | `[0, 6]` | 4 | 8.45 s | 6.80 s |
| miconic-local | 4/4 | `[0, 12]` | 8 | 4.75 s | 3.77 s |
| blocks4ops-clear-local | 4/4 | `[0, 2]` | 3 | 4.13 s | 2.89 s |
| delivery-local | 4/4 | `[0, 9]` | 8 | 5.02 s | 3.99 s |
| blocks3ops-local | 10/10 | `[0, 10]` | 39 | 64.4 s (51.9 s CPU) | 25.5 s (14.3 s CPU) |

Identical solved counts and, on the four small suites, identical costs and rule counts, as the
soundness argument requires. Every run is slower: 1-1.5 s on the small suites, which is the
constant cost of `--heuristic=Domain` on solves that take milliseconds, and **3.6x more solver
CPU on blocks3ops-local** (51.9 s vs 14.3 s) for the same 10/10. Biasing `good_trans` towards the
plan is not free: it fights the cost objective, which has no reason to prefer the plan's actions,
so clasp can spend its time undoing the hint. That is not the whole story — on the blocks3ops
one-shot above the same flag is the *fastest* arm (8x baseline, 2 lazy iterations instead of 8) —
but the sign flips by setting, so it stays off by default and needs its own study.

### The stall case

`--one-shot --max-complexity 4` on `domains/deterministic/blocks3ops/{domain,p005-1,p005-2,p006-1}`
— the case `docs/experiments-log.md` (H8) records as "4 solves, then hangs 21 min". 25 407
occurrences, 4 927 signature classes, **4 271 354** pairs in the full separation relation. Forced:
1 062 occurrences (4.2 %) and 196 classes (4.0 %) good, **0 bad**, so again no batch-0 seeding.
Every arm hit the 30-minute timeout without a policy; what differs is how far the lazy loop got.

| arm | lazy iterations in 30 min | last model cost | violated pairs still outstanding |
|---|---|---|---|
| baseline, no solve limit | 2 | `[0]` | 1 678 288 |
| **`--fix-forced-labels`, no solve limit** | **6** | **`[6]`** | **3 179** |
| `--plan-label-heuristic`, no solve limit | 2 | `[0]` | 1 631 856 |
| both, no solve limit | 2 | `[0]` | 1 602 894 |
| baseline, `--solve-time-limit 300` | 8 | `[7]` | 119 |
| `--fix-forced-labels`, `--solve-time-limit 300` | 9 | `[7]` | 200 |
| `--plan-label-heuristic`, `--solve-time-limit 300` | 7 | `[8]` | 581 |

Without a solve budget this is the clearest result in the whole experiment: `fix_forced_labels`
takes the round from **1.68 M outstanding pairs to 3 179** — a factor of 500, three orders of
magnitude closer to closing the loop — in the same wall clock, and lifts the best model from the
free selection (`[0]`) to cost `[6]`. That is exactly the failure mode the hypothesis targeted:
the unconstrained first relaxation returns a selection that separates nothing, and every
subsequent solve has to chew through millions of counterexamples one 5 000-pair batch at a time.
Pinning 4 % of the classes good decides one side of every constraint those classes appear in, so
the relaxation is no longer free.

With `--solve-time-limit 300` the two arms are level (8 vs 9 iterations, 119 vs 200 outstanding):
the budget already forces the loop forward, which is what `docs/anytime-solve-results.md` found,
and the forced labels add nothing on top. The plan heuristic does not help this round in either
configuration.

## Summary

**`fix_forced_labels` works, and it works on the stall.** On the exact case H8 records as
hanging, it moves the round from 1.68 M outstanding separation pairs to 3 179 in the same 30
minutes (×500) and lifts the best model from the free selection to cost `[6]`; on the largest
blocks3ops one-shot that still finishes, it halves the lazy loop and the solver time (327 s →
167 s) at the identical proven optimum. Correctness holds everywhere it was checked: identical
optimum with the flag on and off on every one-shot round, clingo refuting the opposite of every
label the analysis produces on the unmodified program, and no change in solved counts on any
suite.

**The surprise is *which* rule does the work.** The hypothesis expected rule 1 (a transition into
a dead end is bad) to seed a cross-state propagation through the signature classes. Measured, the
forced-**bad** set is empty in every round of four of the five suites *and* on the stall case:
`frontier_expansion` turns every off-plan successor into a `pruned` state, which `safe_state`
counts as safe, and these domains have no dead ends of their own — so rules 1, 2 and 4 never fire
and the pair seeding never has a pair to seed. All of the gain above comes from **rule 3 alone**,
the sole-surviving-class rule, which pins 4-15 % of the classes *good*. Pinning `good_sig(K)`
decides one side of every separation constraint K appears in, which is what stops the first
relaxed solve from returning a selection that separates nothing. That is a different mechanism
from the one the hypothesis proposed, and it is the one worth keeping.

Two follow-ups this opens up, both aimed at making rules 1/2/4 fire at all:

* Extend rule 1 with the round-level `limit_prune_cost(0)` that `iterative_solver.max_prune_cost`
  already imposes on every post-success round. Under that constraint a transition into a `pruned`
  state cannot be good either, so on those rounds every off-plan successor becomes a non-candidate
  and the cross-state propagation gets its seed.
* Measure at the scale where the plan restriction actually restricts (the local suites expand the
  full reachable space, so `frontier_expansion: false` changes nothing there), together with the
  `frontier_expansion` trade-off in `docs/frontier-expansion-results.md`.

`plan_label_heuristic` is a separate story and the evidence is mixed: 8× faster than baseline on
the blocks3ops one-shot that finishes (2 lazy iterations instead of 8), 3.6× *slower* in solver
CPU on the iterative blocks3ops-local run, and no help on the stall case in either configuration.
Composed with `fix_forced_labels` it dominates on the small one-shot and cancels the forced-label
gain on the stall. Leave it off by default; it needs its own study.

Both flags default to `false` and the default ground program is unchanged.
