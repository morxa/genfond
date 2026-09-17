# Policy-prefix example plans (H30): implementation

Date: 2026-09-17. Branch: `hyp/prefix-plans` (parent: `hyp/anchor-labels`, commit `e68be54`).

**Status: implemented and unit-tested. No benchmark numbers yet** — this document describes the
mechanism, not a measured result. A/B it per `docs/experiments-log.md` before drawing any
conclusion about coverage.

## The problem

Measured on blocks3ops with `--type datalog-sig`: round 9 finds a near-general policy P (feature
cost 8) from three training problems; executed, it solves 90 of 95. Round 10 adds a problem Q
that P fails on. Q's example plans come from SIW (`planners.siw.restarts: 1`,
`max_plans_per_problem: 8`) and have nothing to do with what P does, so on the enlarged
plan-restricted instance there is no model anywhere near cost 8 — the cheapest are frontier
models of cost 18–19 — and the loop drifts into patchworks for hours.

Neither of the two mechanisms already in the tree closes this:

- `policy_conformant_plans` (H27) records P's trajectory only on the problems P **solves**. Q is
  by definition not one of them, so Q's instance gets nothing.
- `anchor_policy_labels` (H29) constrains the model to agree with P where P already worked. It
  says nothing about Q, and it can only anchor steps that are in the state space at all.

What is missing is a plan for **Q** that resembles what P does.

## The mechanism

Whenever a problem joins the training set and a best policy P exists, before that problem's
first round:

1. **Execute P on Q** with the H27 `out_actions` mechanism, bounded by `validation_time_limit`.
   Whatever the executor raises — `CycleError` (`found cycle of length k`), `NoActionError` (`no
   action in reachable state`), `ExecutionTimeout`, or the plain `PolicyExecutionError` the loop
   raises after `policy_steps` when `abort_on_cycle` is off — the actions applied so far are a
   real trajectory.
2. **Cut it** at the point where P went wrong (`prefix_cut`): at the **first visit of the first
   repeated state** when the trajectory loops, otherwise at its end, and at
   `prefix_plan_max_length` in every case.
3. **Re-root Q** at the state that prefix reaches (`frontier.reroot`, which must be passed
   `domain_name=`, not `domain=`) and ask the planner for up to `prefix_plan_count` plans from
   there — the same call `frontier.expand_frontier` makes, pulled lazily so only the plans that
   are consumed are paid for.
4. **Splice** prefix and suffix with `frontier._root_anchored_plan`. `StateSpaceGraph` always
   replays a plan from `problem.init`, so a plan that does not start there is matched against the
   wrong states.
5. **Back off** if the full prefix yields no plan at all: attempt *k* drops the last 2^k actions
   (1, 2, 4, … for `prefix_plan_backoff` attempts), stopping as soon as the prefix would be
   empty. The steps a policy takes just before it starts looping are the ones most likely to have
   been the wrong ones.
6. **Seed** the resulting plans in front of Q's own example plans.

## Where the hook lives

`ProblemIterator.__init__` takes a new `on_problem_added` callback and
`_add_next_problem` calls it through `_seed_prefix_plans(next_problem)`. That is the single
funnel for both places a problem joins the training set — the default escalation ladder and the
`add_problem_after_success` branch — so there is exactly one call site and both behave
identically, which is what the comment on `_add_next_problem` already demands.

The iterator knows nothing about policies, executors or planners: the callback returns a
`PrefixPlans` record and the iterator only decides where to put the plans. `solve_iteratively`
installs a closure that reads `best_policy` **at call time**, so the hook always sees the best
policy as of the round that is about to start. The whole of the mechanism lives in the new
`genfond/prefix_plans.py`.

Ordering inside `_add_next_problem`: the hook runs **last**, after the `min_number_of_plans`
floor batch has been drawn. That makes the state-coverage dedupe meaningful (a prefix plan that
reaches no state the floor batch already covers cannot change the state space) and puts the
refutation reset after the branch that sets `refuted_complexity`.

`_final_cost_minimization_pass` never advances the iterator, so no problem is ever seeded there —
the same scope H29 chose for anchors.

## Decisions

**The cut rule is read off the state sequence, not off the exception type.** `prefix_cut` takes
the executed states and cuts at the first visit of the first state that repeats. That covers
`CycleError` (where the repeat is what the executor detected) and, for free, the
`abort_on_cycle: false` case, where the executor happily walks the loop until `policy_steps` and
then raises a plain "Goal not reached" with no trace at all. The exception type is used for one
thing only: to pick up `trace` when there is one.

**The state sequence is verified.** `executed_states` replays the recorded actions from
`problem.init`, checking each precondition. A deterministic action fixes its own successor; for a
nondeterministic one the executor's `trace` decides, and if that is missing or does not name a
successor of the current state, `None` is returned and the problem is skipped rather than
guessed at. Replaying `out_actions` blindly would be wrong under FOND, and walking `trace` alone
would be wrong whenever a state repeats (the executor overwrites its successor).

**`prefix_plan_max_length` applies to every branch,** not only to a timeout. Nothing else bounds
the length of a trajectory: a policy that neither loops nor gets stuck runs to `policy_steps`
(10000) or to `validation_time_limit`, and the resulting plan would be replayed into the state
space of every later round. This is a deliberate generalisation of the H30 brief, which named the
cap only for the timeout case.

**An empty prefix produces nothing.** If P loops straight back to the initial state, the prefix
is empty and planning from it is exactly the planner's own job on the unmodified problem, which
the `min_number_of_plans` batch already did.

**A policy that solves Q after all contributes its own trajectory** and no planner call.
Execution is randomized (rule order, object bindings, FOND outcomes), so a policy that failed
during validation can get through here; its trajectory is then the most valuable example plan
there is.

**Backoff triggers only on "no plan at all",** per the brief — not when the planner returns fewer
than `prefix_plan_count` plans.

**Caps, floor and dedupe follow H27 exactly.** Prefix plans go in *front* of the problem's other
plans and are counted in `policy_plan_counts`, which both `_plan_cap_reached` and the
`min_number_of_plans` floor subtract, so `max_plans_per_problem` never drops them and they never
suppress the planner's own diversity. They are deduped both by plan identity (`plan_key`) and by
state coverage (`PlanStateCoverage` via `_accept_plan`).

**Refutation bookkeeping.** Seeding plans drops `refuted_complexity` to `min_complexity - 1`,
exactly as `INC_PLANS` does, because the state space the next round is solved over is not the one
any refutation was established over. `_seed_prefix_plans` sets the field directly instead of
calling `_invalidate_refutations()`: that helper's optional `reset_complexity_on_state_space_change`
branch would also reset `complexity`, undoing the `complexity = succ_complexity` that
`_add_next_problem` has just set. (Strictly, the bound `_add_next_problem` carries over is about
the *previous* training set and survives the enlargement of the *new* instance's state space, so
this reset is conservative rather than necessary; it follows the H30 brief and can only widen the
search.)

**Once per problem.** `_prefix_plans_done` guards the hook, so a problem that leaves the training
set under `unselect_problems` and rejoins later is not seeded a second time.

## Config

```yaml
policy_prefix_plans: true   # gated on use_example_plans, hence inert for state/trans/d2l
prefix_plan_count: 2        # plans requested from the prefix's end state
prefix_plan_backoff: 3      # shorter prefixes to try when the full one yields nothing
prefix_plan_max_length: 200 # hard cap on the prefix
```

All four are documented in `genfond/config/default.yaml`. Like `policy_conformant_plans` this is
switched off from a `--config` YAML file; there is deliberately no CLI flag, because
`argparse`'s `store_true` defaults to `False` and a non-`None` CLI value overrides the merged
config (see the config-layering rule in `AGENTS.md`).

Cost when on: one policy execution (bounded by `validation_time_limit`) plus at most
`prefix_plan_backoff + 1` planner calls, and only on problems that are joining the training set
anyway.

## Stats and logging

| Key | Meaning |
|---|---|
| `prefixPlansAdded` | policy-prefix plans actually added over the run, after dedupe |
| `prefixPlanFailures` | problems where the hook ran and nothing was added — no plan from any prefix, an unreplayable execution, an empty prefix, *or* every plan already present |

Both are counted in `ProblemIterator` (where the hook runs) and read out once after the round
loop, so a run that stopped early still reports them. Neither key is written when the mechanism
never produced anything, so the stats CSV of a run with the mechanism off is unchanged.

One INFO line per seeded problem:
`Added N policy-prefix plan(s) for <problem> (prefix length L, backoff b)`.

## Files

| File | Change |
|---|---|
| `genfond/prefix_plans.py` | new: `PrefixPlans`, `executed_states`, `prefix_cut`, `_plans_from`, `policy_prefix_plans` |
| `genfond/problem_iterator.py` | `on_problem_added` hook, `_seed_prefix_plans`, `_prefix_plans_done`, `prefix_plans_added` / `prefix_plan_failures`, the call at the end of `_add_next_problem` |
| `genfond/iterative_solver.py` | the `policy_prefix_plans` gate, the `_prefix_plans_for` closure, `on_problem_added=` on the `ProblemIterator`, the stats read-out |
| `genfond/config/default.yaml` | `policy_prefix_plans`, `prefix_plan_count`, `prefix_plan_backoff`, `prefix_plan_max_length` |
| `tests/test_prefix_plans.py` | new, 18 tests |

## Tests

`tests/test_prefix_plans.py`, 18 tests, ~0.3 s. The prefix-building tests run the **real** datalog
executor over a tiny in-memory "corridor" domain (`go` follows `edge`, `jump` follows `link`;
l0 → l1 → l2 ↔ l3 and the goal l4 is only reachable by `jump`), so a policy with only a `go` rule
provably walks into a cycle. Only the planner is stubbed.

- (a) the prefix is cut at the first repeated state (l2, after two actions), and the planner is
  asked to plan from *that* state, not from the initial one; plus the `prefix_plan_max_length`
  cap, the `prefix_cut` cut rule itself, and `executed_states` rejecting a sequence whose
  preconditions do not hold.
- (b) the spliced plan is root-anchored: it replays from `problem.init` through
  `plan_visited_states` to a goal state, and inside a plan-restricted `StateSpaceGraph`.
- (c) backoff: nothing from the full prefix, a plan one action shorter → `backoff == 1`,
  `prefix_length == 1`, two planner calls with the right re-rooted states; and the
  attempted-but-empty result when the planner never answers.
- plus: a policy that loops back to the initial state yields nothing and calls no planner; a
  policy that solves the problem contributes its own trajectory with no planner call; a
  trajectory with no trace (`abort_on_cycle: false`, "Goal not reached") is still cut at the
  repeat.
- iterator level: front placement and exemption from `max_plans_per_problem` and the
  `min_number_of_plans` floor; the refuted level is dropped in the add-a-problem branch; a
  duplicate plan is not added; the failure counter distinguishes "ran and found nothing" from
  "never ran".
- (d) off: `on_problem_added is None`, nothing seeded, no stats keys — both with
  `policy_prefix_plans: false` and with `use_example_plans: false`.

Gates: `black`, `isort`, `mypy genfond tests` clean; `pytest --import-mode importlib -q` →
289 passed, 1 skipped (baseline 271 passed, 1 skipped, plus the 18 new tests).

## End-to-end sanity run

`--type datalog-sig` on `domains/deterministic/blocks4ops-clear/{p002-1,p003-1,p004-1}`,
seed 0, one thread, 0.4 s total. Round 1's policy solves 2 of 3; round 2 adds `blocks-004-1`, the
hook executes the policy on it, cuts a prefix of length 1, asks SIW from there and gets 2 plans of
cost 4 — and both spliced plans turn out to be identical to plans the SIW floor batch already
drew, so the dedupe drops them: `prefixPlansAdded=0`, `prefixPlanFailures=1`. That is the
intended behaviour on a problem this small, and the same thing H27's own sanity run observed.
A problem where the sampled plans and the policy's trajectory actually diverge is what the
mechanism is for, which is a benchmark question, not a unit-test one.

## What this does not do

- It does not make the new problem *solvable* — the planner may well fail from the prefix (SIW is
  incomplete, and the state may be a genuine dead end). Then the problem keeps only its SIW
  plans, exactly as before.
- It does not interact with `anchor_policy_labels` directly. H29 anchors the best policy's steps
  on problems it solved; H30 seeds plans for a problem it does not. Together they should keep P
  both feasible and preferred *and* give the new instance a trajectory in P's style, but that
  combination is untested beyond the unit level.
- It grows the state space of the newly added problem by up to `prefix_plan_count` plans, exempt
  from `max_plans_per_problem`. The per-round `Plans per problem: max=… mean=…` line is the thing
  to watch on a large suite.

## What to measure next

A/B `policy_prefix_plans: true` against the H29 baseline on the 95-problem blocks suite:
coverage, rounds to the best policy, whether the best policy's cost stops climbing after it
appears, and `prefixPlansAdded` / `prefixPlanFailures` (a run where failures dominate is a run
where the mechanism is paying planner calls for nothing).
