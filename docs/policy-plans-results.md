# Policy-conformant example plans (H27): implementation

Date: 2026-09-17. Branch: `hyp/policy-plans` (parent: `hyp/combo`, commit `b531de1`).

**Status: implemented and unit-tested. No benchmark numbers yet** — this document describes the
mechanism, not a measured result. A/B it per `docs/experiments-log.md` before drawing any
conclusion about coverage.

## The problem

In the datalog loop (`--type datalog-sig`, `use_example_plans: true`), each training problem's
state space is restricted to the sampled SIW example plans
(`planners.siw.restarts`, `min_number_of_plans`, `max_plans_per_problem`). A round finds a
policy P — kept as the best-coverage policy by `keep_best_policy` — that, when *executed*, solves
e.g. 91 of 95 problems. The next round adds a failing problem Q to the training set.

Although some general policy P\* solves Q and everything else when executed, P\* is often
*infeasible* on the ASP instance: the sampled plans of Q, and of the earlier problems, do not
contain the trajectory P\* takes, so the plan-restricted `StateSpaceGraph` simply lacks those
transitions and no selection of good transitions corresponds to P\*. The minimum feature cost
then jumps (5 → 16 observed) and the run drifts into patchwork policies. Which seed happens to
sample a conformant plan decides the outcome.

## The mechanism

Whenever a candidate policy is executed during in-loop validation and **solves** a problem, the
action sequence it took is recorded as an example plan for that problem and is present in every
later round's state space. The trajectory is a free by-product of an execution that already
happens, so this costs no extra planner or executor work.

Flow, per round:

1. `_test_policy_on_problems` asks `execute_policy` for the ground actions it applied
   (`out_actions`), and for each problem the policy solved stores
   `problem name -> Plan` in its new `trajectories` out-parameter. The plan is built by
   `problem_iterator.plan_from_actions`, i.e. the same `pddl.core.Plan` representation
   `siw_planner.to_pddl_plan` and `frontier._root_anchored_plan` produce and
   `StateSpaceGraph.__init__` consumes via `Plan.instantiate(domain)`.
2. Plans are **root-anchored by construction**: policy execution always starts at
   `problem.init`, which is where `StateSpaceGraph` replays a plan from. No splicing is needed,
   unlike `frontier._root_anchored_plan`, which has to prepend the path from the root.
3. `solve_iteratively` hands the round's trajectories to
   `ProblemIterator.record_policy_plans`, which dedupes and stores them.
4. Nothing else happens: no extra round is triggered. The plans are simply part of the state
   space the next round is solved over.

## Decisions

**Which policies' trajectories.** Only the policy the round actually settled on — the winner of
`optimal_model_limit`'s candidate comparison, which is also the policy `keep_best_policy`
compares. Trajectories *are* collected for every validated candidate (they are a by-product of
the validation that already runs, and `candidate_results` caches the winner's validation so
re-running it is not an option), but the losing siblings' are discarded. Same rule as for
`outcomes`: a candidate the round rejected must not change the iterator's state.

**Trajectories are recorded for every problem the policy solved**, not only the ones in the
training set. That is the point of the mechanism: a problem is added to the training set exactly
when it is still unsolved, and what keeps the ASP instance feasible then are the conformant
plans of its already-solved neighbours. `_add_next_problem` therefore uses
`active_plans.setdefault(name, [])` instead of `= []`, so a plan recorded while the problem was
still outside the training set is not wiped when it joins.

**`max_plans_per_problem` does not apply to policy plans.** They go in *front* of a problem's
other example plans and are counted in `ProblemIterator.policy_plan_counts`, which both
`_plan_cap_reached` and the `min_number_of_plans` floor subtract. Rationale: the cap exists to
stop runaway growth from `INC_PLANS` and frontier expansion (four-figure plan counts on a single
problem, see `docs/plan-cap-results.md`), whereas a trajectory a working policy actually took is
the most valuable plan there is — capping it away is precisely what reintroduces the
infeasibility this mechanism removes. Excluding them from the `min_number_of_plans` floor too
means a policy plan never suppresses the planner's own diversity.

**Deduplication** reuses the existing machinery: `plan_key` (identical action sequence already
present) and `PlanStateCoverage` via `_accept_plan` (reaches no state the problem's existing
plans do not already cover). An empty trajectory — the goal already held at `problem.init` — is
dropped, as `siw_planner` drops empty plans.

**Refutation bookkeeping.** Adding a plan changes the state space, so `record_policy_plans` calls
`_invalidate_refutations()` exactly like `INC_PLANS` and `EXPAND_FRONTIER` do (see the "Refuted
complexity levels" section of `AGENTS.md`). One extra guard was needed: the add-a-problem branch
of `_add_next_problem` normally *keeps* its bound (`refuted_complexity = succ_complexity - 1`)
because adding an instance is monotone. Enlarging an *existing* instance's state space is not
covered by that argument, so a new flag `plans_added_since_success` (set by
`record_policy_plans`, cleared by `set_last_result` on `SUCCESS`) makes that branch fall back to
`min_complexity - 1`. Without it the bound established over the smaller state space would
survive and exclude a simpler policy that only became expressible after the plan was added.

## Config

`policy_conformant_plans: true` in `genfond/config/default.yaml`.

It is **gated on `use_example_plans`** and is therefore inert for the rule-based `state`, `trans`
and `d2l` types: without example plans `StateSpaceGraph` is not plan-restricted at all, so no
trajectory can be missing from it. Effectively on for the `datalog*` configs only.

**To switch it off**, set `policy_conformant_plans: false` in a `--config` YAML file (or in
`genfond/config/default.yaml`). There is deliberately **no CLI flag**: `argparse`'s
`store_true` defaults to `False`, and per the config-layering rule in `AGENTS.md` a non-`None`
CLI value overrides the merged config — a `--no-policy-conformant-plans` flag would silently
force the option off on every run that did not pass it.

## Stats

`policyPlansAdded` — cumulative number of policy-conformant plans actually added over the run
(after deduplication), accumulated in `stats` like `validationTime` / `optimalModelSwitches`.
One INFO line per round that recorded anything:
`Added N policy-conformant plan(s) for M problem(s)`.

## Files and functions touched

| File | Change |
|---|---|
| `genfond/execute_rule_policy.py` | `execute_rule_policy(..., out_actions=None)` — appends the ground actions applied |
| `genfond/execute_datalog_policy.py` | `execute_datalog_policy(..., out_actions=None)` — same |
| `genfond/execute_policy.py` | threads `out_actions` through |
| `genfond/problem_iterator.py` | new `plan_from_actions`; `ProblemIterator.record_policy_plans`; `policy_plan_counts` and `plans_added_since_success` state; `_plan_cap_reached` and the `min_number_of_plans` floor subtract policy plans; `_add_next_problem` uses `setdefault` and drops its carried-over bound when policy plans were added |
| `genfond/iterative_solver.py` | `_test_policy_on_problems(..., trajectories=None)`; `policy_conformant_plans` gate; `candidate_results` carries each candidate's trajectories; the round loop calls `record_policy_plans` and logs/accumulates `policyPlansAdded` |
| `genfond/config/default.yaml` | `policy_conformant_plans: true` |
| `tests/test_policy_plans.py` | new, 9 tests |

The new keyword is only ever passed when the mechanism is on (`trajectories is not None`), so
`execute_policy` is called with exactly its old signature otherwise — which is also what keeps
the existing `execute_policy` mocks in `tests/test_in_loop_validation.py`,
`tests/test_optimal_model_enumeration.py` etc. working unchanged.

## Tests

`tests/test_policy_plans.py`, 9 tests, ~0.2 s:

- (a) executing a real datalog policy on `blocks_clear` with `out_actions` yields a plan that
  replays from `problem.init` to a goal state, both through `plan_visited_states` and inside a
  plan-restricted `StateSpaceGraph`;
- cap exemption, front placement, and the `min_number_of_plans` floor;
- a policy plan recorded for a problem outside the training set survives that problem joining it;
- duplicate and empty trajectories are ignored;
- refutation invalidation, including the add-a-problem branch guard;
- (b) end to end through `solve_iteratively` (mocked `solve_step`/`execute_policy`, real
  `ProblemIterator` and `PlanStateCoverage` over the `simple_blocks` fixture): the iterator's
  plan set holds the trajectory, `policyPlansAdded == 1`, the refutation is invalidated;
- (c) with `policy_conformant_plans: false`, and with `use_example_plans: false`, nothing is
  added and the round's refutation stands.

Gates: `black`, `isort`, `mypy genfond tests`, `pytest --import-mode importlib`
(258 passed, 1 skipped; baseline was 249 + 1).

A tiny end-to-end sanity run (`--type datalog-sig` on
`tests/fixtures/pddl_files/blocks4ops-clear`) logs
`Policy-conformant example plans are enabled` and
`Added 0 policy-conformant plan(s) for 0 problem(s)` — the policy's trajectory was already one of
the SIW plans, so the dedupe dropped it, which is the intended behaviour on a problem this small.

## What this does not do

- It records nothing for the problem that is *failing* — that problem has no solving trajectory
  by definition. The win comes from keeping the already-solved problems' instances conformant.
- It cannot help a run whose very first policy already solves nothing.
- It grows the state space monotonically. `max_plans_per_problem` no longer bounds that growth
  for policy plans, so the per-round `Plans per problem: max=… mean=…` line is the thing to
  watch on a large suite.
