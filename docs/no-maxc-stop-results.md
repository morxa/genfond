# Max complexity is not the end of the run (H33): implementation

Date: 2026-09-18. Branch: `hyp/no-maxc-stop` (parent: `hyp/resample-on-stall`, commit `e1584fb`).

**Status: implemented and unit-tested. No benchmark numbers yet** — this document describes the
mechanism and the decisions it rests on, not a measured result. A/B it per
`docs/experiments-log.md` before drawing any conclusion about coverage.

## The observation

logistics_dp, 47 problems, `--type datalog-sig`, 4 h graceful wall budget:

* the loop reached `max_complexity` on its current training set after **28 rounds / 84 min**,
  with **15/47** solved, and terminated with `failureReason=maxcomplexity`;
* **2.6 h of the budget went unused**;
* slower variants that never reached the top of the sweep reached **23/47** in the same budget.

Reaching `max_complexity` ends one complexity sweep over one training set. It is not evidence
that the run has nothing left to try — and since H31 made rounds fast, it is the *common* ending
rather than a rare one.

## The mechanism

`continue_after_max_complexity` (default **true**). When `ProblemIterator.__next__` would raise
`StopIteration` while the sweep sits at `max_complexity`, unsolved problems remain and the wall
deadline is not reached, it instead continues, in this order:

1. **Resample** the example plans (H32's `resample_planner_plans`), if `resample_on_stall` is on
   and fewer than `resample_max` resamples have been spent. This is first because it is the only
   escalation that changes the **state space**, and therefore the only one that can make a policy
   possible that the exhausted sweep could not express at any complexity. A resample that
   replaces nothing (every stream dry) is not a continuation and falls through to 2.
2. **Add the next unsolved problem** outside the training set, through the normal
   `_add_next_problem` bookkeeping. All accumulated plans stay (policy-conformant, policy-prefix
   and planner plans alike).
3. **Neither available** → `StopIteration`, and the run ends with the `maxcomplexity` reason it
   always had.

Either continuation **restarts the complexity sweep at `min_complexity`** and logs one INFO line:

```
Max complexity 2 reached with 2 unsolved problem(s); continuing by adding problem gripper-1-2
Max complexity 3 reached with 1 unsolved problem(s); continuing by resample (2 plan(s) for 1 problem(s))
```

Stats: `maxComplexityContinuations` (plus H32's `resamples` / `resampledPlans`, which the
continuation's resample feeds through the same counters).

### Why restarting the sweep at `min_complexity` is sound

It can only cost time, never a policy. The feature pool at complexity `c` contains every feature
of complexity `< c`, so every policy reachable from the level the sweep had climbed to is also
reachable on the way back up from `min_complexity` — and preferred, since the `#minimize` is over
feature cost. The restart cannot exclude anything.

What it buys is **size**. The low-complexity instances are far smaller (roles alone contribute
`n·m²` grounded values per state), and a continuation happens precisely because the large
instances at the top of the sweep produced no policy — on the domains this is aimed at, several
of them because they could not be grounded at all (`Id out of range`, or memory). The same
argument `reset_complexity_on_state_space_change` rests on, applied at the one point where the
alternative is not "a cheaper route to the same round" but "no round at all".

### Refutation bookkeeping

Following `AGENTS.md` → *Refuted complexity levels* exactly; `_restart_sweep` deliberately does
**not** touch refutations itself, each caller having already done what its own change requires:

| continuation | refutations | why |
|---|---|---|
| resample | cleared (`resample_planner_plans` → `_invalidate_refutations`) | the plan set changed, so the state space changed; a refutation is a claim about one ASP instance and does not survive it |
| add a problem | `refuted_complexity = succ_complexity - 1` kept (`_add_next_problem`, unless `plans_added_since_success`) | adding an instance is monotone: a selection solving the larger set solves every subset, so "nothing below `succ_complexity` works" carries over |

`sweep_target` is raised to the level the sweep had reached, the same bookkeeping
`_invalidate_refutations` does for `reset_complexity_on_state_space_change`, and for the same
reason: without it the restarted sweep would add an example plan at `min_complexity` on the very
next round (`last_step == INC_COMPLEXITY` still holds from the climb) and the restart and the
addition would alternate at one fixed level instead of the sweep ever climbing again.
`last_step` goes back to `START`.

### `max_cost` is never loosened

The resample leaves `max_cost` alone by design (H32). The add-a-problem branch *does* reset it to
`MAX_COST` — but on this path that is a no-op, and provably so: reaching the continuation with a
problem still to add means the last `elif` of `__next__` failed on `active_problems_solved`, and
`active_problems_solved` is only ever cleared by `_add_next_problem`, which sets `max_cost =
MAX_COST` in the same breath (as does INC_PLANS). A training set that has not been solved since
the last problem joined it therefore already carries `MAX_COST`. Tested both ways.

### Termination

Each continuation either spends one of the finitely many resamples (`resample_max`) or moves one
problem into the training set. Both are bounded, so the run cannot circle here; when both are
exhausted the iterator stops exactly as before.

## Where the code and the hypothesis disagreed

Worth recording:

1. **`failureReason=maxcomplexity` is not written where the run stops.** It is set at the end of
   every round that *produced* a policy which did not solve every problem
   (`iterative_solver.py`, the `else` of `if solved and stop_after_first_solution`), and simply
   happens to be the last value left standing when the iterator later runs out. A run whose every
   round is `NO_SOLUTION` never sets it at all — before or after this change. The stop itself is
   a bare `StopIteration` with no reason attached, which is why the new stat is a counter on the
   iterator rather than a new `failureReason` value.
2. **Step 2 cannot help a *cleanly* refuted sweep.** If every round of the exhausted sweep was a
   full-pool `NO_SOLUTION`, then by the same monotonicity that lets `_add_next_problem` keep its
   bound, each of those levels is refuted for the enlarged training set too — the re-climb is
   guaranteed to fail again. Step 2 pays off only where the sweep was *not* cleanly refuted:
   rounds that hit `TIMEOUT`, `OUT_OF_RESOURCES`/`Id out of range`, or ran over the restricted
   generators. Those are exactly the rounds large instances produce, and exactly the case the
   observation is about (28 rounds, 15/47 solved, so successes and failures mixed), which is why
   step 2 is worth trying rather than stopping — but the ordering matters: the resample, which
   changes the state space and so escapes the monotonicity argument entirely, goes first.
3. **A resample that replaces nothing still costs its budget.** `_do_resample` increments
   `resamples_done` before it knows whether any plan was replaced, exactly as H32's stall trigger
   did. Kept deliberately: it stops a run whose plan streams are dry from re-attempting the same
   futile planner calls at every continuation. The *continuation* is not counted in that case —
   it falls through to step 2.
4. **The iterator needed a way to ask about things it cannot see.** Whether the wall budget is
   spent and whether a resample is still available both live in `iterative_solver`. They are
   passed as `MaxComplexityContinuation`, a three-callable object whose default instance says
   "time left, no resample" — so a bare `ProblemIterator` (every existing test) continues only by
   adding the next unsolved problem, and needs no solver to be testable.
5. **No existing "unselect/restart" path was reusable.** `unselect_problems` *replaces* the
   training set inside `_add_next_problem` and is a different mechanism (it drops problems);
   `reset_complexity_on_state_space_change` and `resample_reset_complexity` restart the sweep but
   only as a side effect of a state-space change, and both are off by default. `_restart_sweep`
   factors out the three lines they share.

## Config

```yaml
continue_after_max_complexity: true
```

Interacts with `resample_on_stall` / `resample_max` (which continuation is available first) and is
bounded by `max_wall_time` (a continuation is not started once `wall_deadline` has passed or a
SIGINT/SIGTERM is pending — the same two conditions the round loop breaks on).

## Tests

`tests/test_no_maxc_stop.py`, 12 tests, no solver calls:

* iterator-level with stub round results: (a) a resample at the top of the sweep continues the
  run, clears the refutations, restarts at `min_complexity` and raises `sweep_target`; the
  resample is preferred over adding a problem; a resample that replaces nothing falls through;
  (b) with no resample available the next unsolved problem joins the training set and the sweep
  restarts at `min_complexity` rather than at `succ_complexity`, keeping the monotone bound;
  (c) everything in the training set and nothing to resample → `StopIteration` as before, and the
  wall deadline stops the run even with both continuations available; (d) the option off
  reproduces the old behaviour; (e) `max_cost` is unchanged across both kinds of continuation;
  the log line.
* end to end through `solve_iteratively` (mocked `solve_step`, real `ProblemIterator`, real plans
  for the `simple_blocks` fixture): the run continues past `max_complexity` by spending H32's
  resample (complexities `2, 3, 2, 3` instead of `2, 3`), with the fresh planner config
  `{seed: 1000003, restarts: 2}` and the shared `resamples`/`resampledPlans` accounting, and
  reports `maxComplexityContinuations`; with the option off the same run stops after two rounds
  with the resample unspent.

Full suite: 335 passed, 1 skipped (baseline 323 + 12).

A smoke run (`--type datalog-sig --max-complexity 2`, two gripper-local problems) shows the whole
path in a real run: resample attempted → nothing new → `continuing by adding problem gripper-1-2`
→ restarted sweep → second continuation exhausted → clean stop.
