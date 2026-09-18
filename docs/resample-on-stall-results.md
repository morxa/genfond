# Resampling the example plans on a stall (H32): implementation

Date: 2026-09-18. Branch: `hyp/resample-on-stall` (parent: `hyp/frontier-bound`, commit
`19e2452`).

**Status: implemented and unit-tested. No benchmark numbers yet** — this document describes the
mechanism and the decisions it rests on, not a measured result. A/B it per
`docs/experiments-log.md` before drawing any conclusion about coverage.

## The observation

blocks3ops, `--type datalog-sig`, six seeds, rounds now taking seconds thanks to H31:

* 4 of 6 seeds reach 95/95 within minutes.
* The other 2 converge within a handful of rounds on a **patchwork** policy — feature cost 13–16
  on a 7–8 problem training set — and the best coverage then does not move for the next 20–40
  rounds.

The two arms differ only in which example plans SIW happened to hand over. Everything the
escalation ladder can do to a bad sample is *add* to it: INC_PLANS draws one more plan, the
frontier expansion splices in a plan from a re-rooted state, H27/H30 add the policy's own
trajectories. None of that removes the plans that made the instance express a patchwork cheaply
in the first place, and `max_cost` then keeps every later round below the patchwork's cost, so
the run is stuck with the sample for good.

## The mechanism

`solve_iteratively` counts rounds since the best coverage last **strictly** improved. A round
that produces no policy at all never improves anything and counts as a stalled round like any
other. When the counter reaches `stall_rounds` (default 6) and fewer than `resample_max`
(default 3) resamples have happened, the round about to start first resamples:

For every problem in the training set,

1. split its example plans at `policy_plan_counts[name]`. The front part is the H27
   policy-conformant plans and the H30 policy-prefix plans; the back part is the planner's
   sample (the `min_number_of_plans` floor, INC_PLANS additions, frontier plans);
2. discard the back part and draw exactly as many plans again, skipping anything identical to a
   plan the problem keeps or to one just discarded;
3. rebuild that problem's `PlanStateCoverage` from the plans that remain.

A problem whose stream can offer nothing new keeps what it has.

### Where the new plans come from

First from the problem's **own** stream, simply continued. The planner yields lazily and dedupes
within a stream (`siw.diverse.unique_plans`), so its next plans are guaranteed-new ones that cost
no more search than a later INC_PLANS would have paid anyway.

Only when that stream is exhausted is a **fresh** one built, and the problem's stream is then
replaced by it so later INC_PLANS draws continue there. `iterative_solver._fresh_plan_iterator`
changes two things about the planner config for it, and both are needed:

* `seed` → `seed + resample_index * 1000003`;
* `restarts` → `max(restarts, 2)`.

The second is not cosmetic. SIW's restart 1 is the **identity** view of the task
(`siw.diverse.restart_view`: `if restart == 1: return task, None`), and the permuted action and
candidate-goal orders the seed drives only exist from restart 2 on. At the measured
`planners.siw.restarts: 1` a merely reseeded stream would therefore replay exactly the plans the
resample is trying to get away from.

The whole draw runs under a reseeded **global** RNG
(`random.seed((config["seed"] or 0) + resample_index * 1000003)`), whose state is saved and
restored around it. The reseed covers planners that draw from `random` rather than from their own
`Random`; the restore is what keeps policy execution — which does draw from the global RNG — and
so `--seed` reproducibility unaffected. It wraps the *draw*, not just the construction of the
stream, because plans are pulled lazily: by the time the generator body runs, a reseed placed
only around `compute_plans(...)` would already have been undone.

Pulls per problem are capped at `4 * wanted + 8`, so a stream that keeps handing back plans the
problem already has cannot turn the resample into an unbounded planner loop.

### What the iterator does afterwards

Exactly what a plan-set change requires, and nothing more:

| | |
|---|---|
| `_invalidate_refutations()` | The state space changed, so no complexity level is refuted any more — the same rule INC_PLANS, `record_policy_plans` and the frontier expansion follow. |
| `plans_added_since_success = True` | The plan set changed after the success that established `succ_complexity`, so `_add_next_problem` must not carry that bound over: enlarging/changing an instance's state space is not the monotone step that argument covers. |
| `max_cost` | **Unchanged.** A resample is not permission to accept a worse policy. The best policy so far still stands (`keep_best_policy`), and the next round is still asked to beat its cost. |
| training set | **Unchanged.** No problem is added in the same step. |
| `last_step` | **Untouched.** The next round runs the same training set at the current complexity over the new sample. |
| `_plan_cap_logged` | The problem discarded to log the `max_plans_per_problem` message again if the cap is hit once more. |

`resample_reset_complexity` (default **off**) additionally restarts the sweep at
`min_complexity`, mirroring `reset_complexity_on_state_space_change` and defaulting off for the
same reason: it does not change which policies are reachable, it only trades a full re-sweep for
smaller instances on the way back up.

Because the resample happens at the top of the round body rather than inside
`ProblemIterator.__next__`, the round that triggers it already runs over the new sample:
`iter_kwargs["example_plans"]` *is* the iterator's `active_plans` dict, so the new plans are
already in this round's configuration, and only `complexity` and `enforce_highest_complexity` —
the two values a resample can change — are refreshed from the iterator.

## Config

```yaml
resample_on_stall: true       # gated on use_example_plans, so inert for state/trans/d2l
stall_rounds: 6
resample_max: 3
resample_reset_complexity: false
```

Stats: `resamples`, `resampledPlans`, `stallRoundsMax`.

## Where the code and the hypothesis disagreed

Worth recording, because the hypothesis as stated would have produced a no-op:

1. **SIW does not draw from the global RNG.** `siw.diverse.restart_view` builds its own
   `random.Random(f"{seed}:{restart}")` from `planners.siw.seed` (config default `0`), which is
   *not* derived from genfond's `seed`. Reseeding the global RNG around the call therefore
   changes nothing for SIW; what changes the sample is `planner_config["seed"]`. The global
   reseed is kept anyway (cheap, and it covers a planner that does use `random`), but the
   planner-config seed is the one that does the work.
2. **A fresh seed alone is inert at `restarts: 1`.** Restart 1 is the identity view, so the run's
   whole measured configuration produces the same plans for every seed. Hence the
   `max(restarts, 2)` above. It also means the seed-to-seed variance the observation describes
   cannot come from the SIW sample being seeded differently — with `planners.siw.seed: 0` fixed
   and `restarts: 1`, every seed draws the *same* plans. What genfond's `seed` varies is policy
   **execution** (rule order, object bindings, successor draws), hence which problems count as
   solved, hence which problems get added to the training set and in what order, hence which
   policy-conformant and policy-prefix plans the instance ends up carrying. The stall is real
   either way and the fix addresses it at the level of the plan set, but the causal story in the
   hypothesis ("an unlucky early plan sample") is, on this configuration, one step removed from
   the seed.
3. **Plans are pulled lazily from a generator.** They can be re-drawn — a new generator is cheap
   to build — but a reseed that wraps only the construction call is undone before a single plan
   is generated. The reseed has to wrap the draw.
4. **`PlanStateCoverage` only grows.** It is built for a world where plans are only added; a
   resample removes them. Without `reset()` + re-add of the kept plans, the tracker would keep
   reporting the discarded plans' states as covered and reject replacements as redundant. This is
   the one existing structure the mechanism had to change.

## Tests

`tests/test_resample.py`, 14 tests, no solver calls:

* `ProblemIterator`-level: only planner plans are replaced (policy/prefix plans survive in front,
  `policy_plan_counts` intact); the coverage tracker resets; the refutation is invalidated while
  `max_cost`, the training set and the complexity are left alone; `resample_reset_complexity`
  restarts the sweep; the fresh-stream fallback fires only when the problem's own stream runs dry
  and becomes the problem's stream afterwards; an exhausted stream with no fallback is a no-op;
  a problem holding only policy plans has nothing to resample.
* End to end through `solve_iteratively` (mocked `solve_step`, real `ProblemIterator`, real plans
  for the `simple_blocks` fixture): the counter triggers on the round after `stall_rounds`
  non-improving rounds and not before; the fresh planner config is `{seed: 1000003, restarts: 2}`;
  `resample_max` is respected (`resamples` 1 and 2 for the two settings, with the expected seed
  sequence); the option off, `use_example_plans` off, and `stall_rounds`/`resample_max` of 0 each
  resample nothing; the global RNG state is bit-identical across a run that resamples twice.

Full suite: 323 passed, 1 skipped (baseline 309 + 14).

## H32b: a climbing sweep is not a stall

Date: 2026-09-18. Branch: `hyp/stall-refine` (parent: `hyp/no-maxc-stop`, commit `e9bbdac`).

**Status: implemented and unit-tested. No benchmark numbers yet** -- as with H32/H33 above, this
describes the mechanism, not a measured result.

### The observation

blocks3ops, full loop: after a near-general policy (88-93 of 95), the loop adds a 10-11-block
problem whose instance provably needs feature complexity 6. Thanks to H31, the frontier
lower-bound abort at complexity 4 and 5 each take only seconds. H32's stall counter, as written,
counted every round that did not *improve* the best coverage as evidence of a stall -- including
those abort rounds and any plain `Result.NO_SOLUTION` -- so it reached `stall_rounds` *during*
this climb and resampled, discarding the example plans the climb depended on. The run then had to
rebuild its state space from a different sample instead of simply finishing the climb it was
already seconds away from completing.

### The fix

Two changes, both against `StallTracker` (`genfond/iterative_solver.py`, new; `_record_round`,
`_do_resample` and the round loop's stall trigger are now thin call sites against it):

1. **Only a non-improving `Result.SUCCESS` counts.** `StallTracker.record` takes `coverage`
   (`None` for a round that produced no policy) and `counts_toward_stall`; every call site that
   passes `coverage=None` also passes `counts_toward_stall=False`. Concretely: `Result.FRONTIER`
   (the H31 lower-bound abort included -- it is reported as `Result.FRONTIER` too, see
   `docs/frontier-bound-results.md`), `Result.NO_SOLUTION`, `Result.TIMEOUT`/
   `Result.OUT_OF_RESOURCES`/`Result.UNKNOWN` never increment the counter; only a `SUCCESS` whose
   validated coverage does not exceed the best seen does. A resample itself
   (`StallTracker.note_resample`) resets the counter regardless of which trigger paid for it, so
   it is never "evidence of a stall" either -- it is the fix for one.
2. **A climbing sweep additionally blocks the trigger even if the counter is somehow at
   threshold.** `StallTracker.sweep_climbing` is recomputed on every `record` call from
   `refuting_or_abort` (was *this* round's `Result` a refutation or frontier lower-bound abort?)
   and `complexity < max_complexity` (is there a higher level left to escalate to?).
   `should_resample` -- and the round loop's own trigger -- refuse to fire while it is true, and
   log `Stall detected but the sweep is still climbing (complexity %d of %d); deferring the
   resample` (INFO) instead. `stats["stallDeferred"]` counts how often this happened.

`refuting_or_abort` is set from the *actual* `Result` `solve_step` returned for that round, not
from `ProblemIterator.last_result` -- which matters, because `solve_iteratively` forcibly
overwrites `problem_iterator.last_result` to `Result.NO_SOLUTION` whenever a round's policy solves
its training set but not every problem in the run (the `else` branch right after the round's own
`keep_best_policy` bookkeeping, present before this change). Reading that overwritten value
instead of the round's real `Result` would misclassify every "solved the training set, not the
whole suite" round as climbing and defer resampling indefinitely below `max_complexity`, which
would break fix 1's own test case: a run of nothing but non-improving `SUCCESS` rounds must still
resample once `stall_rounds` is reached, regardless of how far below `max_complexity` it is.

### A structural finding: fix 1 alone already closes the observed bug, and fix 2 cannot double-fire

Point 1 alone is sufficient for the *observed* blocks3ops scenario: the climb is a run of genuine
`Result.NO_SOLUTION`/`Result.FRONTIER` rounds with no `SUCCESS` in between, and none of those
increment the counter any more, so the counter never reaches `stall_rounds` during the climb at
all -- fix 1 already prevents the resample from firing mid-climb there.

Point 2's guard is real, reachable logic (tested directly against `StallTracker`, see below), but
it can be proven never to change the outcome of a round reached through the normal loop, given fix
1: `rounds_since_improvement` only ever increases on a `Result.SUCCESS` call to `record`, and
every such call passes `refuting_or_abort=False` (a `SUCCESS` is not a refutation), which sets
`sweep_climbing = False` in that same call. So whichever round is the *first* to push the counter
up to `stall_rounds` necessarily also clears `sweep_climbing` for the very next round's check --
the round that first becomes stall-ready is never preceded by a climbing round, because "stall
ready" and "just climbed" are set by two different kinds of round and the counter can only cross
the threshold via the kind that clears climbing. Reaching `stall_ready and sweep_climbing`
together would need the check to be skipped at the crossing round for some *other* reason (e.g.
`resamples_done >= resample_max`) and later re-evaluated with the counter still elevated and
budget restored -- and the budget only ever decreases. The guard is therefore a correct,
intentional piece of defense in depth (and the natural place to put "do not resample while the
sweep is expected to escalate on its own"), not dead code by design -- it is simply not
observable as a behavioural difference through the `Result` sequences this fix's own test harness
(a single, non-growing training set) can drive `solve_iteratively` through. `should_resample` and
`record` are exercised directly against the sequence that produces "deferred, then fires" once
climbing stops, since that is the actual mechanism the task asks for.

The same reasoning gives point 3 (no double resample with H33 in the same round) for free: H33's
continuation (`ProblemIterator._continue_past_max_complexity`, called from inside `__next__()`)
runs *before* the round loop's own H32b trigger check for that round, and its resample -- via the
same `_do_resample`/`StallTracker.note_resample` -- resets the counter to 0 first. Since
`resample_on_stall` forces `stall_rounds >= 1` (see the `resample_on_stall = ... and (stall_rounds
and resample_max)` guard near the top of `solve_iteratively`), a freshly-zeroed counter can never
satisfy `rounds_since_improvement >= stall_rounds` on the very next check, so H32b's own trigger
cannot also fire in the round H33 just resampled in.

### Config / stats

No new config keys. New stat: `stallDeferred` (only written once it is nonzero).

### Tests

`tests/test_stall_refine.py`, 16 tests, no solver calls:

* `StallTracker`-level (11 tests): the first `SUCCESS` establishes a baseline rather than counting
  as a stall; a non-improving `SUCCESS` increments the counter and an improving one resets it
  while `stall_rounds_max` keeps the historical peak; `NO_SOLUTION`/`FRONTIER`/`TIMEOUT`/
  `OUT_OF_RESOURCES` never increment it; `note_resample` resets the counter but not
  `best_coverage_seen`/`stall_rounds_max`; `sweep_climbing` is true only after a refuting/abort
  round strictly below `max_complexity`, false at the top of the sweep, and false after a
  `SUCCESS`/`TIMEOUT`/`OUT_OF_RESOURCES` round; `should_resample` requires the threshold, the
  budget, and not climbing together; and (c) the exact deferred-then-fires transition once the
  climb reaches the top of the sweep.
* End to end through `solve_iteratively` (mocked `solve_step`/`_test_policy_on_problems`, real
  `ProblemIterator`, real plans for `simple_blocks`, mirroring `tests/test_resample.py`'s and
  `tests/test_no_maxc_stop.py`'s harnesses): (a) six non-improving `SUCCESS` rounds (after the
  round-1 baseline) trigger a resample at the top of round 8, not before; (b) sixteen
  `NO_SOLUTION`/`FRONTIER`/`TIMEOUT`/`OUT_OF_RESOURCES` rounds with `stall_rounds: 1` never
  resample at all; (d) a pure `NO_SOLUTION` climb to `max_complexity` with `stall_rounds: 1`
  (stall-ready every round under the *old* counting) resamples exactly once, via H33's
  continuation, with `stallRoundsMax` staying 0 throughout.

Full suite: 351 passed, 1 skipped (baseline 335 + 16).

### Where the task description and the code disagree

The task's H32b brief frames the climbing guard as blocking a resample that the (fixed) counter
would otherwise have fired mid-climb. Given fix 1, that specific composition cannot arise through
`solve_iteratively`'s own round loop (see the structural finding above): the round that first
makes the counter stall-ready is always a `SUCCESS`, which always clears `sweep_climbing` for the
following check. The guard is implemented and tested exactly as specified regardless -- it is
correct, and it is the right place to encode "do not resample while the sweep is about to escalate
on its own" -- but it should be understood as defense in depth for a composition this
architecture's own accounting does not otherwise produce, not as the mechanism actually closing
the observed blocks3ops bug (fix 1 does that on its own).

The H31 frontier lower-bound abort is confirmed to be reported as plain `Result.FRONTIER`, exactly
as `docs/frontier-bound-results.md` describes -- there is no separate `Result` value for it, so
`refuting_or_abort=True` at the `Result.FRONTIER` call site is what makes it a "refuting/abort"
round for the climbing guard, indistinguishable at this layer from a `Result.FRONTIER` round that
is about to retry via frontier expansion instead of escalating complexity (`frontier_progress`
decides which happens next, but only inside `ProblemIterator.__next__`, not visible here). Treating
every `Result.FRONTIER` as a potential "about to climb" round rather than trying to predict
`frontier_progress` from `iterative_solver` can only ever cause an extra deferral, never a missed
one, so this is a deliberately conservative reading of "abort," not an approximation that could
let a resample through mid-climb.
