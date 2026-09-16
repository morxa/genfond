# One final cost-minimization pass after `add_problem_after_success`

Date: 2026-09-15. Branch: `hyp/min-count` → `hyp/final-climb`.

## The mechanism

`add_problem_after_success` (`docs/no-cost-climb-results.md`) skips the post-success complexity
climb so the loop grows the training set instead of re-solving it at increasing cost. That
trades cost minimization for training-set growth: `docs/experiments-log.md`'s c-base vs. c-combo
rows show the cheaper policy (cost 10, 9 training problems) generalizing *further* than the more
expensive one (cost 21, 19 training problems) despite the smaller training set, and the "Caution
on reading these two rows" paragraph names this exactly as the open question this hypothesis
answers.

`final_cost_minimization` (config key + `--final-cost-minimization` CLI flag, both default off,
`BooleanOptionalAction` with `default=None` like the other switches) is only meaningful together
with `add_problem_after_success`. Once `solve_iteratively`'s main loop ends (in practice: its
`stop_after_first_solution and solved` break, since a chain of successes that solves every
problem in a row is exactly the case where the complexity ladder never gets a turn),
`_final_cost_minimization_pass` (`genfond/iterative_solver.py`) runs the old climb exactly once
more, standalone, on the frozen final training set (`problem_iterator.active_problems`, with its
final `active_plans`/`dead_states`):

- reset to `succ_complexity`, tighten `max_cost` to `cost - 1`;
- increment complexity while `max_cost > complexity` and `complexity < max_complexity` — the
  same condition as `ProblemIterator`'s `INC_COMPLEXITY` branch, and, like that branch, `NO_SOLUTION`
  does not stop the climb, only `OUT_OF_RESOURCES`/`TIMEOUT` do (gracefully, keeping the best
  policy found so far);
- a candidate is tested with `_test_policy_on_problems` (execute every problem, no early break,
  unlike the main loop's smallest-first testing block — undercounting a later problem would bias
  the comparison between candidates) and kept only if it still solves every *training* problem
  (an ASP-level `Result.SUCCESS` only proves the model satisfies the constraints; execution can
  still fail, e.g. a cycle — the same distinction the main loop's own testing makes);
- among valid candidates, the pass keeps the one with (most problems solved overall, then lowest
  cost) — not cost alone. See the gripper finding below for why that guard matters.

`enforce_highest_complexity` and the frontier are both off during the pass: it does not track
refutations the way the main loop does, and the training set is already frozen, so there is
nothing left for the frontier to expand. Stats record `finalPassRounds`,
`finalPassCostBefore`/`finalPassCostAfter`. The pass is a no-op — and the run byte-identical —
whenever `final_cost_minimization` is false or `add_problem_after_success` is off.

## Local gates

`black`/`isort`/`mypy genfond tests`/`pytest` all green (154 passed, 1 skipped) on
`hyp/final-climb`, including 4 new unit tests in `tests/test_final_cost_minimization.py` that
mock out `solve_step`/`execute_policy` to check the pass's control flow directly: every problem
gets tested even after an earlier miss, `NO_SOLUTION` does not stop the climb, a candidate that
fails training-set execution is discarded, and `OUT_OF_RESOURCES` ends the pass gracefully.

## Workstation validation

All runs: `apptainer run ... genfond_env.sif python -m genfond --type datalog-sig -n 1
--max-memory 60000 --seed 0 --add-problem-after-success --solve-time-limit 300
--role-complexity-offset 2 --concept-complexity-offset 1 [--final-cost-minimization] ...`,
`timeout 40m` per arm (none came close: the slowest arm finished in 49s). desk-03,
`PYTHONHASHSEED=0`.

| suite | arm | solved | cost | rules | training problems | final-pass rounds | wall |
|---|---|---|---|---|---|---|---|
| blocks3ops-local (10) | off | 10/10 | 9 | 40 | 5 | – | 25.2s |
| blocks3ops-local (10) | on | 10/10 | **7** | **27** | 5 | 2 | 48.7s |
| gripper-local (5) | off | 5/5 | 7 | 4 | 1 | – | 2.3s |
| gripper-local (5) | on | 5/5 | 7 | 4 (identical policy) | 1 | 1 (rejected) | 3.7s |
| miconic-local (4) | off | 4/4 | 11 | 9 | 3 | – | 3.0s |
| miconic-local (4) | on | 4/4 | **8** | **8** | 3 | 2 | 5.7s |

Held-out generalization, blocks3ops (`scripts/eval_policy.py --seed 0 -i 3
domains/deterministic/blocks3ops-heldout`, 12 problems, 8–30 blocks, never seen during
training):

| arm | held-out solved |
|---|---|
| off (cost 9, 40 rules) | 4/12 |
| on (cost 7, 27 rules) | **8/12** |

The off policy solves the two smallest held-out sizes plus one lucky 15-block instance and fails
everywhere else with "no action found" (its rules are specific enough to miss states outside the
≤5-block training distribution). The on policy — same training set, two more rounds of climbing
— solves every size up to 20 blocks it's tested on plus both 30-block instances, doubling
coverage. This is the direct confirmation of the `docs/experiments-log.md` caution paragraph:
cost and generalization move together here, and the final pass buys back the generalization
`add_problem_after_success` was trading away, without touching the training-set growth that made
10/10 reachable in the first place.

miconic's cost-8 policy (8 rules) is structurally different from the cost-11 one (9 rules), not
just smaller — compare the two rule sets printed via `print_policy.py`; the after-pass policy
routes `up`/`down` off `origin`- and `destin`-relative concepts instead of testing `lift-at`
directly, and picks up an unconditional `depart` rule.

### The rejection case (gripper)

Gripper's single training problem (`gripper-1-1`) is the interesting negative case: the pass
*did* find a cheaper candidate (cost 6, complexity 5, log: `Found policy with cost [0, 6] for
gripper-1-1`) with an unconditional `pick(P, Q, R) :- b_empty(r_primitive(carry,0,1))` rule. It
still solves the one training problem (so it isn't discarded by the training-set check), but
`_test_policy_on_problems` against all 5 problems shows it fails to solve every sibling gripper
instance the way the cost-7 policy does — so the (most-solved, then cost) comparison rejects it
and keeps the original cost-7 policy, which `print_policy.py` on `results-ab/gripper-on.policy`
confirms is rule-for-rule identical to `gripper-off.policy` (just reordered). This is exactly why
the pass compares on solved count first and cost second rather than cost alone: at complexity 2
above `min_complexity`, gripper's tiny state space admits an overfit, cheaper-looking policy that
a pure cost climb (as in the pre-`add_problem_after_success` ladder, which also tests on all
problems before accepting) would have had to separately catch via its own testing block — here
the guard is built into the pass itself.

## Caveats

- One seed, one workstation run per arm, small suites — matches the validation protocol asked
  for, not a sweep. The blocks3ops-local training set (5 problems, ≤5 blocks) and the cost-9/
  cost-7 pair are a single data point; re-running with a different seed or a larger local suite
  before relying on the 4/12 → 8/12 number as a general claim would strengthen it.
- The pass roughly doubled wall time on all three suites here (25s→49s, 2.3s→3.7s, 3.0s→5.7s) —
  cheap at this scale, but it reruns solves on the full final training set at each climbed
  complexity, so the cost scales the same way the pre-`add_problem_after_success` climb did (see
  `docs/no-cost-climb-results.md`), just paid once at the end instead of after every success.
  Not yet measured on a training set large enough to make that matter (e.g. the 19-problem combo
  run in `docs/experiments-log.md`, where a single complexity round already exhausts memory).
- `all_features` during the pass is fixed to `config["use_unrestricted_features"]` for every
  round rather than the main ladder's toggle-then-climb sequence (try restricted, only escalate
  to unrestricted if that fails, then increment complexity) — a simplification, documented in
  the function's docstring, that trades a small amount of missed cost reduction for not needing
  a second piece of state-machine bookkeeping in a pass that only runs once.
- **Update (post-review):** the workstation runs above only exercised the
  `stop_after_first_solution`-break trigger (every problem solved immediately after training-set
  growth stops), which understated what the code actually does. `_final_cost_minimization_pass`
  is called unconditionally after `solve_iteratively`'s main loop, so it already runs whether
  that loop ends via the break *or* via the `for` loop exhausting `problem_iterator`
  (`StopIteration` — e.g. `OUT_OF_RESOURCES`/`TIMEOUT` closing off every escalation branch, or
  the ladder reaching `max_complexity` with problems still unsolved), as long as a policy exists;
  it is not reached after an unhandled exception. This is now covered by
  `test_final_pass_runs_when_the_iterator_exhausts_without_solving_everything` and
  `test_final_pass_does_not_run_after_an_unhandled_exception` in
  `tests/test_final_cost_minimization.py`, which drive `solve_iteratively` itself (with
  `solve_step`/`execute_policy` mocked) through a scripted SUCCESS → NO_SOLUTION →
  `OUT_OF_RESOURCES` sequence that never satisfies `stop_after_first_solution` and confirm the
  pass still runs on the frozen final training set. So a run that stalls with unsolved problems
  still remaining in the training set (the c-noclimb pattern in `docs/experiments-log.md`) *does*
  get a final pass, on whatever training set and cost it stopped at — no separate
  `final_pass_after_rounds` trigger was needed. No production code changed for this; only the
  docstring/comments were corrected to describe it accurately.

## Bounding the climb itself (`hyp/final-pass-bound`)

On the cluster the pass climbed complexity level by level, unbounded, until `max_cost <=
complexity`. On blocks3ops that reached a pool of 1,945 features / 5,738 concepts, where a single
`solve_step` call exceeds any budget and the job dies. `NO_SOLUTION` alone never stops the climb
(the same as the main loop's `INC_COMPLEXITY` branch), and `wall_deadline` only catches this if
the run has a wall budget configured at all. The pass is meant to find a *cheaper* policy near the
one the main loop already found, not to explore the whole ladder, so it needs its own bound
independent of `max_cost`/`max_complexity`/`wall_deadline`.

Two new config keys, both documented next to the other `final_pass_*` keys in
`genfond/config/default.yaml`:

- `final_pass_max_levels` (default `2`; also `--final-pass-max-levels`): caps the number of
  complexity levels the pass tries above `succ_complexity`. `null` reproduces the previous
  unbounded climb. Reported as `stats["finalPassLevelsTried"]` (identical to `finalPassRounds` by
  construction — one complexity level per loop iteration — under a name tied to what this key
  bounds).
- `final_pass_max_pool` (config only, no CLI flag; default `null` = no limit): skips an
  individual round, without grounding it at all, when the pool `FeaturePool` builds for that
  round (features + concepts + roles) exceeds this many elements. Implemented in `solve()`
  because the pool's size is known cheaply right after `FeaturePool` is built, before
  `to_clingo()`/`Solver.solve()` — the actual grounding/solving is what exhausts memory, not
  building the pool. A skipped round is reported like an ordinary `NO_SOLUTION` round, so this
  composes with, rather than replaces, `final_pass_max_levels`.

See `_final_cost_minimization_pass`'s docstring for the mechanics. Unit tests in
`tests/test_final_cost_minimization.py` (mocked `solve_step`) cover: the pass stopping at exactly
`final_pass_max_levels` rounds even when `max_cost` would otherwise allow many more, `null`
reproducing the old unbounded climb, and `final_pass_max_pool` being threaded through to
`solve_step`'s `max_pool_size` kwarg unchanged. Not re-validated on the cluster as part of this
change — this is the config/control-flow fix, not a new benchmark pass.
