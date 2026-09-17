# The frontier lower bound (H31): implementation

Date: 2026-09-17. Branch: `hyp/frontier-bound` (parent: `hyp/prefix-plans`, commit `975deab`).

**Status: implemented and unit-tested. No benchmark numbers yet** — this document describes the
mechanism and the argument it rests on, not a measured result. A/B it per
`docs/experiments-log.md` before drawing any conclusion about coverage.

## The observation

From a cluster log (blocks3ops, `--type datalog-sig`, `lazy_pairs: true`, `anchor_policy_labels`
on, `solve_time_limit: 300`):

* The first lazy iteration grounds no `sig_pair`/`dist` facts, so its optimum is the cheapest
  selection the round can ever have. On the rounds in question that optimum already reported a
  non-zero frontier-transition count.
* The loop nevertheless ran nine or ten further 300 s solves per round — about 105 of 121 minutes
  across three rounds — refining a model that `solve_step` was always going to reject.
* The anchor fallback never fired, because a frontier-bearing SAT model is neither `UNSATISFIABLE`
  nor `UNKNOWN`.
* When a zero-frontier answer finally existed (complexity 6, `[0,17]` proven optimal in 295 s),
  later iterations timed out at 300 s with *regressing* incumbents (`[43,70]`, `[2,48]`, …),
  because each solve restarts from no bound at all.

## The argument

`#minimize { 1@3,I,S1,A,S2 : good_trans(I,S1,A,S2), pruned(I,S2) }` is the **highest**-priority
level of `solve_datalog_sig.lp` (`1@2` in `solve_datalog.lp` — the number differs, its being
highest does not), so clingo minimises the frontier-transition count lexicographically first.

The lazy loop's grounded program at any iteration carries a *subset* of the full round's
separation constraints — pairs are only ever added, never removed — so every model of the full
round is a model of it. That is the same relaxation argument the loop's existing `UNSATISFIABLE`
case rests on.

Put together: if an iteration's solve is **proven optimal** and its frontier component is k > 0,
then k is the minimum frontier count over all models of the relaxation, hence a **lower bound**
over all models of the full round. `solve_step` accepts only a zero-frontier model as a policy.
So the round has no policy, however many more pairs the loop would go on to add, and its only
remaining use is the frontier states it selected.

Three places where the argument would fail, and what guards each:

| Hazard | Guard |
|---|---|
| A model whose cost was never *proved* optimal bounds the minimum from above, not below | `solver.optimal` (false for a `SATISFIABLE` incumbent and for `UNKNOWN`) |
| Reading `cost[0]` when the frontier level never grounded (no reachable `pruned/2`) returns the feature complexity instead | `cost_utils.prune_cost` counts the levels the config accounts for and returns 0 unless the vector is one longer |
| The **anchored** program carries `:- anchor(I,S,A), not good_action(I,S,A)`, a constraint the full round does not | `iterative_solver.solve` re-solves without anchors before believing the bound |

Note that the argument does **not** depend on iteration 1 grounding nothing. With
`fix_forced_labels` on, `_seed_forced_pairs` grounds batch 0 before the first solve — but a seeded
pair is part of the full eager encoding too, so the program is still a relaxation and the bound
still holds. The abort is therefore checked after *every* iteration, not just the first.

## What was implemented

### 1. Frontier lower-bound abort (`frontier_lower_bound_abort: true`)

`lazy_pairs._frontier_lower_bound(solver)` is `solver.optimal and solver.solution and
solver.prune_count > 0`. When it holds, `solve_with_lazy_pairs` logs

```
Frontier lower bound k > 0 proven after lazy iteration i; no policy in this round
```

and returns `SolveStatus.OPTIMAL` with that model in hand. `iterative_solver.solve` extracts its
`frontier/2` states as usual and `solve_step` reports `Result.FRONTIER`, so the caller expands the
frontier and retries — the existing path, unchanged. `Result.FRONTIER` neither tightens `max_cost`
nor marks problems solved nor touches `refuted_complexity` (`ProblemIterator.set_last_result`), so
the abort cannot introduce a refutation the round did not earn.

The non-lazy path has no loop to cut short, but `solve` draws the same conclusion there: it logs
it, records it, and skips the `optimal_model_limit` enumeration whose models would all have been
discarded as frontier-bearing a few lines later.

New helper `cost_utils.prune_cost(cost, minimize_good_signatures, minimize_selected_count)`, the
mirror of `feature_cost`, plus `Solver.prune_count`.

### 2. Widened anchor fallback

`solve` previously re-solved without anchors on `UNSATISFIABLE` or `UNKNOWN`. It now also does so
when the anchored attempt proves the frontier lower bound, counting the same `anchorFallbacks`
stat and logging `Anchored solve needs a frontier transition in every model; retrying without
anchors`. Only the unanchored solve is ever the round's answer, so the anchored attempt's
`frontierLowerBound` / `frontierLowerBoundAbort` stats are cleared before the re-solve. With fix 1
this fallback costs seconds instead of the nine or ten further 300 s solves.

### 3. Round budget and warm-started bounds

`round_time_limit` (seconds, `null` = off) budgets one solver **attempt** — the whole lazy loop
rather than a single clingo call. It is implemented by narrowing the `Solver`'s own
`wall_deadline`, which `Solver.solve` already composes with `solve_time_limit` as whichever runs
out first and recomputes on every call, so each iteration gets
`min(solve_time_limit, remaining round budget)` with no further plumbing. Per *attempt*, not per
round: an anchored round and its unanchored fallback get one budget each, because only the
fallback's answer is ever used and leaving it with an exhausted budget would be worse than having
no budget at all.

`lazy_pairs_warm_start` (default true) replays the previous iteration's cost vector as clingo's
initial optimisation bound — `Solver.solve(bound=...)` sets `opt_mode = "opt,<b>,..."`, which is
inclusive (verified against clingo: a bound equal to the optimum admits it; one unit tighter is
`UNSATISFIABLE`). A timed-out iteration can then no longer come back with an incumbent worse than
the one the round already had.

**Deviation from the brief, and why.** The brief asked for the bound to be grounded through the
existing `limit_feature_cost` / `limit_prune_cost` `#program` parts. That would be **unsound**, and
the codebase says so itself — `Solver.solve` already carries the comment *"Constraints added since
the previous solve can only raise the optimum, so a bound carried over from it would be
unsound."* Adding pairs monotonically **raises** the lazy loop's optimum, so the previous
iteration's cost is a *lower* bound on the next one's, not an upper bound. Grounding it as a
constraint would make an iteration `UNSATISFIABLE` whenever the optimum genuinely rose, the loop
would report the round `UNSATISFIABLE`, and `ProblemIterator` would record a refutation the round
never proved. Worse, a grounded constraint cannot be retracted: `#program` parts are cumulative, so
there would be no way back.

The `opt_mode` bound has none of that: it is per-solve configuration, not part of the program, so
it can simply be dropped. The loop treats an `UNSATISFIABLE`/`UNKNOWN` under a warm bound as *"the
optimum rose above it"*, logs `re-solving without it`, re-solves unrestricted, and turns warm
starting off for the rest of the round — so the extra solve is paid at most once per round. That
makes the whole mechanism a pure search heuristic: it can change how long a round takes, never
what it concludes. The round's own `max_cost` is untouched and still in force as the grounded
`limit_feature_cost` constraint, so the effective bound is always the tighter of the two.

Because `limit_prune_cost` already caps the frontier level and the warm bound is the *whole* cost
vector, "at least as good on both levels" is exactly lexicographic `<=` here, not a per-level
conjunction.

### 4. Capped frontier plans

`ProblemIterator.record_frontier_expansion` already stopped adding plans at
`max_plans_per_problem`; it now says so at INFO (`Dropping N frontier plan(s) for P: it is already
at max_plans_per_problem=K`) and counts them in `frontier_plans_dropped`, surfaced as the
`frontierPlansDropped` stat. Each of those was a planner call that bought nothing.

`solve_iteratively` avoids making them in the first place: a new public
`ProblemIterator.plan_cap_reached(name)` filters the frontier states before `expand_frontier`, and
the skipped count goes into `frontierStatesSkipped`. This deliberately gives up the chance of
learning that such a state is a dead end — the round has already been told to stop growing that
problem's plan set, and `frontier_progress` stays false either way, so the iterator falls through
to the normal escalation ladder instead of spinning.

## Config

| Key | Default | Meaning |
|---|---|---|
| `frontier_lower_bound_abort` | `true` | stop the round on a proven non-zero frontier optimum |
| `lazy_pairs_warm_start` | `true` | replay the previous iteration's cost as clingo's initial bound |
| `round_time_limit` | `null` | wall-clock budget for one solver attempt of a round |

CLI: `--frontier-lower-bound-abort/--no-...`, `--lazy-pairs-warm-start/--no-...`,
`--round-time-limit`.

## Stats

`frontierLowerBound`, `frontierLowerBoundAbort` (per round, cleared each round),
`lazyWarmStartRelaxed`, `frontierPlansDropped`, `frontierStatesSkipped`.

## Files

| File | Change |
|---|---|
| `genfond/cost_utils.py` | `_above_levels`, `prune_cost` |
| `genfond/solver.py` | `prune_count` property; `solve(bound=...)` via `opt_mode` |
| `genfond/lazy_pairs.py` | `_frontier_lower_bound`; abort, warm start and round budget in the loop |
| `genfond/iterative_solver.py` | `solver_deadline`, `_proves_no_policy`, widened anchor fallback, non-lazy abort log/stat, frontier-state filter, `frontierPlansDropped` |
| `genfond/problem_iterator.py` | `plan_cap_reached`, `frontier_plans_dropped`, the drop log |
| `genfond/config/default.yaml` | three new keys |
| `genfond/__main__.py` | three new CLI flags |
| `tests/test_frontier_bound.py` | 20 tests |
| `tests/test_lazy_pairs.py`, `tests/test_anchor_labels.py` | test doubles updated for `solve(bound=...)` |

## Smoke figures (not a benchmark)

`tests/fixtures/pddl_files/gripper`, `--type datalog-sig --max-complexity 5 -n 1 --seed 0`, one
2-ball instance. All three configurations solve 1/1.

| configuration | clingo solves | wall |
|---|---|---|
| default (abort + warm start on) | 7 | 2.3 s |
| `--no-frontier-lower-bound-abort` | 12 | 2.5 s |
| `--no-lazy-pairs-warm-start` | 7 | 2.6 s |

One instance on one seed says nothing about coverage; it only shows the abort fires on a real run
and that the round loop still converges to the same policy with and without it. The real question
— whether the freed solves translate into coverage on blocks3ops — needs `scripts/ab_bench.sh`
on the cluster.

## What to watch for

* `frontierLowerBoundAbort` should be *common* on a frontier-expanding run and always accompanied
  by `Result.FRONTIER`; an abort that is never followed by an expansion means the frontier states
  are all in problems at `max_plans_per_problem` (check `frontierStatesSkipped`), and the round
  loop is spinning.
* `lazyWarmStartRelaxed` counts rounds where the optimum rose above the last iteration's cost —
  i.e. rounds that paid one extra solve for the warm start. If it equals the round count, the warm
  start is costing more than it saves on that domain and should be turned off.
* `anchorFallbacks` should go **up** with this change (the condition is strictly wider). It going
  up is expected, not a regression.
