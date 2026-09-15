# Occam bias over selected element count: results

Date: 2026-09-15. Branch: `hyp/min-rules` -> `hyp/min-count`.

## Background

The hand-picked pool referenced by the task (`c_equal(on,on_g)`, `ontable`, `ontable_g`, roles
`on`/`on_g`; 5 selected elements, 13 rules) generalizes to 5/12 held-out blocks3ops instances
(8-30 blocks). The learned policies from `docs/min-rules-results.md` (`minimize_good_signatures`,
which minimizes the *rule* count directly) select more elements at lower total complexity and
generalize at 0/12. A datalog rule is the projection of one good signature onto the selected
elements, so the number of distinct object classes a policy can distinguish grows exponentially
in the number of selected elements -- minimizing total complexity alone favors several cheap,
narrow elements over one general one, which is the opposite bias from what the hand-picked pool
exhibits.

This adds `minimize_selected_count: none | below | above` (config `default.yaml`, CLI
`--minimize-selected-count`, `solve_datalog_sig.lp` only) in the same style as
`minimize_good_signatures`: `#minimize { 1@P, E : c_selected(E) }` plus the same statement for
`f_selected` and `r_selected` at the same priority `P`, so together they form a single
`#minimize` level over the total number of selected concepts+features+roles (not their
complexity). `below` (`@-2`) is a tie-break decided after feature complexity; `above` (`@1`) is
decided before it. Both settings compose with `minimize_good_signatures` (`below`/`above` at
`@-1`/`@2`); the frontier-transition count was bumped from `@2` to `@3` to stay the highest
priority in every combination.

## The cost-vector layout, generalized

`cost_utils.feature_cost` used to special-case a single `below` flag (moving feature complexity
from `cost[-1]` to `cost[-2]`). It now takes both `minimize_good_signatures` and
`minimize_selected_count` and computes the number of active `below` groups from the config
(`_below_levels`), indexing `cost[-(below+1)]`. Both groups are guaranteed to ground whenever
their `#program` part is grounded at all (`good_sig/1` grounds for any round with an alive
non-goal state; `c_selected(name)` is an unconditional fact, not a choice), so the offset is a
pure function of the config, never of what the model happened to select. Every caller
(`Solver.feature_complexity`, `problem_iterator.set_last_result`, `__main__.py`'s stats `"cost"`
key) now threads both values through. `iterative_solver.solve` rejects
`minimize_selected_count != "none"` outside `solve_prog == "solve_datalog_sig.lp"` with the same
`ValueError` pattern as `minimize_good_signatures`, and now logs `"Selected N element(s): ..."`
for every solve (not just when the setting is active), so the effect is visible in any run's log.

## Gates

In `/home/thofmann/code/genfond-wt/min-count`: `pytest tests/ --import-mode importlib -q` ->
**150 passed, 1 skipped** (identical to `hyp/min-rules`'s baseline; the previously-fixed
`test_both_encodings_find_the_same_policy[gripper]` stays fixed). `mypy genfond tests`, `black
--check genfond tests`, `isort --check genfond tests` all clean.

### Hand-built ASP test

`tests/test_minimize_selected_count.py`'s `SEPARATE` instance forces two independent separation
decisions (via dead-state alternatives that cannot legally be selected as good, so which
signature is good/bad is not a choice): hitting distinguishing set D1 needs concept `x1`
(complexity 1) or `y` (complexity 3); hitting D2 needs `x2` (complexity 1) or `y`. `{x1, x2}`
costs complexity 2 (2 selected elements); `{y}` alone costs complexity 3 but only 1 selected
element (it hits both). With `"none"` or `"below"` (no tie to break -- 2 is uniquely cheapest),
the solver picks `{x1, x2}`, cost `[2]`/`[2, 3]`. With `"above"` it picks `{y}` alone, cost
`[2, 3]` (count leads, complexity trails) -- one complexity-3 element beats two complexity-1
elements because it needs fewer of them. A `@pytest.mark.parametrize` over all 3x3
`minimize_good_signatures` x `minimize_selected_count` combinations (reusing the existing
good-signature `TIE` instance on disjoint state ids in the same grounded program) checks
`Solver.feature_complexity` recovers the correct complexity (2 or 3) in every one of the nine
combinations, and that the two axes never interfere with each other's choice.

## blocks3ops-local -> blocks3ops-heldout (12 problems)

`--type datalog-sig -n 1 --seed 0 --max-memory 6000 --add-problem-after-success
--minimize-selected-count {none,above}` ran locally to completion; `above --minimize-good-signatures
above` and `below --minimize-good-signatures above` were moved to the remote workstation
(`desk-03.ml.rwth-aachen.de`, `apptainer`, `--max-memory 60000`, `timeout 30m`) per the
coordinator's instruction to stop local benchmark runs, and **both timed out** -- see below.

| setting | active training problems | wall | solver CPU | feature cost | selected elements | rules | good sigs (bad) | held-out |
|---|---|---|---|---|---|---|---|---|
| none (baseline)        | 4 (002-1,003-1,004-2,005-2) | 15.17s | 10.89s | 7 | 6 (5 concept, 0 feature, 1 role)  | 36 | 72 (260) | 0/12 |
| above                  | 4 (002-1,003-1,004-2,005-2) |  6.27s |  3.46s | 6 | 5 (4 concept, 0 feature, 1 role)  | 27 | 30 (107) | 0/12 |
| above + sigs=above     | -- | **timed out (30 min)** | -- | -- | -- | -- | -- | not run |
| below + sigs=above     | -- | **timed out (30 min)** | -- | -- | -- | -- | -- | not run |

`above` alone shrinks the policy relative to baseline (36 -> 27 rules, 5 concepts+1 role instead
of 5 concepts... actually 6->5 selected elements) and drops `c_equal` entirely: baseline's
concepts are `c_equal(on,on_g)` (+ negation), `c_primitive(clear_g)`, `c_primitive(ontable)`,
`c_primitive(ontable_g)`, role `on_g` (+ negation); `above`'s are `c_primitive(clear_g)`,
`c_primitive(ontable)`, `c_not(c_primitive(ontable_g))`, `c_not(c_not(c_primitive(ontable_g)))`
(a double-negation reaching the same distinguishing power one complexity level cheaper than
`c_equal`), role `on_g` (+ negation). **Held-out generalization stayed at 0/12 for both `none`
and `above`** -- minimizing selected-element count alone, with the full feature pool still
available, did not reproduce the hand-picked pool's 5/12 either, matching the pattern already
observed for `minimize_good_signatures` in `docs/min-rules-results.md`.

### `above + sigs=above` and `below + sigs=above` time out

Both combined-bias runs reached the *same* round that the baseline (`none`) round after it
(complexity 4, 5 active training problems, 90 concepts / 76 roles / 514 signature classes, 276
states for `above+above`; complexity 3, 4 problems, 256 states for `below+above`), found a first
feasible model within a second, and then made no further progress for the entire 30-minute
budget -- confirmed not to be node contention (`uptime` showed load average ~1 on the 96-core
node, no other CPU-heavy processes, both immediately before and reflected in the fact the
processes were cleanly reaped by `timeout` with nothing else consuming CPU). This is a genuine
solver-time blowup, not resource contention: combining two "above" Occam biases means clingo's
branch-and-bound has to jointly optimize *three* stacked priority levels (good-signature count,
then selected-element count, then feature complexity) instead of one or two, and proving
optimality at each level requires re-proving no lower-priority improvement can help -- on
`docs/min-rules-results.md`'s own numbers, `minimize_good_signatures=above` *alone* solved a
*larger* round (6 active problems) in 5.04s; it is the composition of two "above" objectives on
this domain that is expensive, not either one alone (`minimize_selected_count=above` alone above
took 6.27s). Reported as observed, not fixed, per house rule (research code: report, don't fix).
Given this, `above` alone (not composed with `minimize_good_signatures`) is the setting used for
the regression-suite validation below.

## Regression suites: `minimize_selected_count=above` vs. baseline

Run remotely (`desk-03.ml.rwth-aachen.de`, apptainer, `-n 1 --seed 0 --max-memory 60000`, no
`--add-problem-after-success`, full training set from the start, `timeout 30m`, all four
completed in under 5 seconds wall time each).

| suite | solved | feature cost | selected elements | baseline (none) |
|---|---|---|---|---|
| gripper-local           | 5/5 | 6  | 3 (2 concept, 1 feature, 0 role) | 5/5, cost 6 |
| miconic-local            | 4/4 | 12 | 4 (4 concept, 0 feature, 0 role) | 4/4, cost 12 |
| blocks4ops-clear-local  | 4/4 | 2  | 2 (1 concept, 1 feature, 0 role) | 4/4, cost 2 |
| delivery-local           | 4/4 | 9  | 4 (4 concept, 0 feature, 0 role) | 4/4, cost 9 |

No regressions: identical solved counts and identical feature costs to the pre-existing baseline
on all four suites (same numbers reported in `docs/min-rules-results.md`). Unlike
`minimize_good_signatures=above` (which paid one extra complexity point on
`blocks4ops-clear-local` to save one good signature), `minimize_selected_count=above` did not
need to trade complexity for count on any of these four domains -- the complexity-minimal
solution already happened to be count-minimal too.

## Summary

`minimize_selected_count` behaves as designed: `above` measurably shrinks both the selected-
element count and the rule count on blocks3ops-local (6->5 elements, 36->27 rules) without
costing anything on the four regression suites, and the hand-built ASP test confirms it makes
exactly the qualitative trade the hypothesis describes (one expensive general element over two
cheap narrow ones). It did not, on its own, reproduce the hand-picked pool's 5/12 held-out
generalization -- consistent with `minimize_good_signatures`'s own finding that an Occam bias
over the *objective* is not a substitute for restricting the *feature pool* the solver searches
over in the first place. Composing it with `minimize_good_signatures=above` is not currently
practical on blocks3ops-local: both directions tested timed out at 30 minutes on the exact round
size that either bias alone handles in seconds, so the composition needs either a real time
budget investigation or an alternative solving strategy (e.g. `--opt-strategy usc`) before it can
be evaluated for generalization at all.
