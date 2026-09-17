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
