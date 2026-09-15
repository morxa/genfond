# Occam bias over good signatures: results

Date: 2026-09-15. Branch: `hyp/goal-suffix` -> `hyp/min-rules`.

## Background

`--type datalog-sig` (`solve_datalog_sig.lp`) must select >= 1 good transition per alive
non-goal state, but may select more. Every extra good signature it labels beyond that minimum
turns into an extra datalog rule (a rule is the projection of a good signature onto the
selected elements), and a rule that only exists to fit one training-state peculiarity is exactly
the kind of thing that fails to generalize. Observed on blocks3ops: the plain learner returns a
36-rule policy (blocks3ops-local, feature cost 7) that solves 0/12 held-out problems, while a
hand-picked tiny feature pool gave a 13-rule policy (cost 6) that solves 5/12.

This adds `minimize_good_signatures: none | below | above` (config default.yaml, CLI
`--minimize-good-signatures`) to `solve_datalog_sig.lp`: `below` adds
`#minimize { 1@-1, K : good_sig(K) }` (decided *after* feature cost -- a pure tie-break), `above`
adds `#minimize { 1@1, K : good_sig(K) }` (decided *before* feature cost, so fewer signatures can
cost more features). Both sit strictly below the frontier-transition count (`@2`), so a frontier
transition is never preferred just to save a rule.

## The cost-vector wrinkle

`model.cost` is ordered highest-priority-first, and a level that never grounds is dropped
entirely (not zeroed) -- both already true before this change (the `@2` frontier level). Every
reader of the vector used to assume the feature/concept/role complexity was `cost[-1]`.

With `above` (`@1`, between frontier `@2` and feature `@0`) that is still true: feature
complexity is the lowest priority present, so it stays last. With `below` (`@-1`, *under* feature
cost) it necessarily is not: `below` exists specifically to make the good-signature count the
tie-break decided *after* feature cost, which is by definition the new lowest priority, so it
becomes the new `cost[-1]` and feature complexity moves to `cost[-2]`.

Fixed by extracting a shared `feature_cost(cost, minimize_good_signatures)` (`genfond/cost_utils.py`,
pulled out of `solver.py` to avoid a circular import with `problem_iterator.py`) and routing every
reader through it: `Solver.feature_complexity`, `ProblemIterator.set_last_result` (which uses the
extracted value for `max_cost`, i.e. what actually bounds `limit_feature_cost` next round -- this
one is a correctness fix, not just cosmetic, for `below`), and `__main__.py`'s stats `"cost"` key.
`lazy_pairs.py` and the `"Found policy with cost {policy.cost}"` log line print the raw vector and
were left alone. `minimize_good_signatures != "none"` is rejected outside `solve_datalog_sig.lp`
(`solve_prog`) with a clear `ValueError` instead of an opaque clingo grounding error. New tests in
`tests/test_minimize_good_signatures.py` pin a hand-built tied instance (two states, disjoint
alternative actions, one action's signature class shared between them) where `none` may pick
either equal-cost combination, and `below`/`above` must both pick the one with fewer signatures;
they also check the exact cost-vector layout and the `solve_prog` guard.

## Gates

In `/home/thofmann/code/genfond-wt/min-rules`: `pytest tests/ --import-mode importlib -q` ->
**136 passed, 1 skipped** (131 passed, 1 skipped, 1 known-failing before this branch --
`test_both_encodings_find_the_same_policy[gripper]`, repaired as task A, see below). `mypy`,
`black --check`, `isort --check` all clean.

### Task A: repaired tied test

`test_both_encodings_find_the_same_policy[gripper]` failed on `hyp/goal-suffix` (pre-existing,
logged in `docs/goal-suffix-results.md`): with `c_equal` concepts now generated, the plain and
signature-quotiented encodings find two different equal-cost (6) optimal policies for gripper (5
rules vs 4), not the same rule set. Rewrote the test to assert equal `cost[-1]` (already the
weaker claim `test_both_encodings_agree_on_feature_cost` made, now stated directly in this test
too) and that both generated policies actually execute-solve the problem
(`execute_datalog_policy`), rather than requiring byte-identical rules -- that is genuinely no
longer guaranteed once there are multiple equal-cost optima, and was never the claim the two
encodings' equivalence makes.

## blocks3ops-local -> blocks3ops-heldout (12 problems)

`--type datalog-sig -n 1 --seed 0 --max-memory 6000 --add-problem-after-success
--minimize-good-signatures {none,below,above}`, then `scripts/eval_policy.py --seed 0 -i 3` on
blocks3ops-heldout.

| setting | active training problems | wall | solver CPU | feature cost | rules | good sigs (bad) | local 10/10 | held-out |
|---|---|---|---|---|---|---|---|---|
| none  | 4 (002-1,003-1,004-2,005-2)             | 15.09s | 10.86s | 7 | 36 | 72 (260) | yes | 0/12 |
| below | 5 (002-1,003-1,003-3,004-2,005-2)       | 11.25s |  8.20s | 5 | 26 | 30 (111) | yes | 0/12 |
| above | 6 (002-1,003-1,004-1,004-2,005-2,003-3) |  5.04s |  2.76s | 6 | 20 | 23 (90) | yes | 0/12 |

\* the "Generated N rule(s)" log line and the final rule count agree exactly with `good_sig` in
every run (a direct cross-check that the cost-vector extraction is reading the right level): none
36/72, below 26/30, above 20/23.

Both `below` and `above` substantially shrink the policy (36 -> 26 -> 20 rules, 72 -> 30 -> 23
good signatures) relative to `none`, confirming the mechanism does what it says: fewer good
signatures, fewer rules. **Held-out generalization stayed at 0/12 in all three settings on this
run**, though -- the Occam bias alone did not reproduce the 5/12 the task description reports for
a hand-picked *feature pool*. The three runs also trained on different problem subsets (each
setting's search trajectory diverges after the first added problem, since which round succeeds
first differs under a different objective), so this is not a controlled like-for-like ablation of
rule count alone; a hand-restricted feature pool is a different intervention (it removes
c_equal/roles the solver would otherwise reach for) than a bias over signature count with the
full pool still available. Reported as observed, not fixed, per house rule.

**Concepts/roles used by the final policy** (from `print_policy.py`, families only -- exact
argument lists vary per rule):

| setting | concepts | roles |
|---|---|---|
| none  | `c_primitive(clear_g,0)`, `c_primitive(ontable,0)`, `c_primitive(ontable_g,0)`, `c_equal(r_primitive(on,0,1),r_primitive(on_g,0,1))`, and `c_not(...)` of all four | `r_primitive(on_g,0,1)`, `r_not(r_primitive(on_g,0,1))` |
| below | `c_primitive(clear_g,0)`, `c_primitive(ontable_g,0)`, and `c_not(...)` of those plus `c_not(ontable,0)` | `r_primitive(on_g,0,1)`, `r_not(r_primitive(on_g,0,1))` |
| above | `c_primitive(clear_g,0)`, `c_primitive(ontable,0)`, `c_primitive(ontable_g,0)`, and `c_not(...)` of all three | `r_primitive(on_g,0,1)`, `r_not(...)`, `r_transitive_closure(r_primitive(on_g,0,1))`, `r_not(r_transitive_closure(...))` |

`none` is the only setting that reaches for `c_equal` (comparing `on` against the goal-role
`on_g`); `below` and `above` both do without it. `above` is the only one that reaches for
`r_transitive_closure`, trading a more expensive role for fewer rules.

## Regression suites: below/above vs. baseline (none)

`--type datalog-sig -n 1 --seed 0 --max-memory 6000` (no `--add-problem-after-success`, full
training set from the start), one run per suite per setting. Baseline (`none`, not rerun here,
taken from the task's stated numbers): gripper-local 5/5 c6, miconic-local 4/4 c12,
blocks4ops-clear-local 4/4 c2, delivery-local 4/4 c9.

| suite | below: solved | below: feature cost | below: good sigs | above: solved | above: feature cost | above: good sigs |
|---|---|---|---|---|---|---|
| gripper-local           | 5/5 | 6  | 10 | 5/5 | 6  | 10 |
| miconic-local            | 4/4 | 12 | 15 | 4/4 | 12 | 15 |
| blocks4ops-clear-local  | 4/4 | 2  | 5  | 4/4 | 3  | 4  |
| delivery-local           | 4/4 | 9  | 11 | 4/4 | 9  | 11 |

`below` reproduces the baseline feature cost exactly on all four suites, as designed (it only
breaks ties after feature cost is already optimal). `above` matches baseline feature cost on
three of four; on blocks4ops-clear-local it accepts a strictly higher feature cost (3 instead of
2) to shave one good signature (4 instead of 5) -- the intended, and here the only observed,
above/below trade-off. No suite lost a single solved problem in either setting.

## Timing: is `above` slower?

No slowdown observed. On blocks3ops-local, `above` was the *fastest* of the three settings
(5.04s wall / 2.76s solver CPU, vs. 15.09s / 10.86s for `none` and 11.25s / 8.20s for `below`),
and the four regression suites all solved in well under a second of solver time regardless of
setting. This is consistent with `above`'s search landing on a smaller active-problem set before
`add-problem-after-success` needed another round (6 problems reached by the point of success vs.
4 for `none`), not evidence that minimizing good-signature count first is intrinsically cheaper to
solve -- the three blocks3ops-local runs are different search trajectories, not a controlled
apples-to-apples timing ablation (see above). No case in this validation showed `above` taking
materially longer than `none`.
