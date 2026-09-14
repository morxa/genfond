# Skip the post-success complexity climb: implementation and benchmark results

Date: 2026-09-14. Branch: `learn-from-examples` → `hyp/no-cost-climb`.

## The mechanism

After a `Result.SUCCESS`, `ProblemIterator.set_last_result` sets `active_problems_solved =
True` and tightens `max_cost = cost[-1] - 1`. `ProblemIterator.__next__` then prefers, in
priority order: frontier retry → add an example plan → enable unrestricted feature generators →
`INC_COMPLEXITY` (guarded by `max_cost > complexity`) → only then the add-a-problem branch. So
after every success the loop climbs feature complexity on the *same* training set — each round
required to beat the previous cost — until `max_cost <= complexity` or `max_complexity` is
reached, before adding the next unsolved problem. On grounding-bound domains that climb re-runs
the largest training set so far at complexities that were never needed to find a policy, and
those are exactly the rounds that exhaust memory or hit clingo's 32-bit ground-fact id ceiling.

`add_problem_after_success` (config key + `--add-problem-after-success` CLI flag, both default
`false`) makes `__next__` take the add-a-problem branch immediately after any success, ahead of
the plan/feature/complexity escalation, whenever an unsolved problem is not yet in the training
set. `_add_next_problem()` factors out the branch body (reset `max_cost`, `active_problems_solved
= False`, `complexity = succ_complexity`, `refuted_complexity` carry-over, `unselect_problems`,
plan-iterator priming) so both call sites stay identical. With the flag off, `__next__` is
unchanged — the new branch's guard is simply false — and the existing `test_problem_iterator.py`
suite passes unmodified. Frontier retry keeps top priority in both cases.

## Sanity runs

`-n 1 --seed 0 --max-memory 4000 --type datalog-sig`, single-threaded and seeded so the two runs
are comparable (see `benchmark-runs-need-seed-and-single-thread`).

### `domains/suites/gripper-local` (5 problems)

| | off (default) | `--add-problem-after-success` |
|---|---|---|
| solved | 5/5 | 5/5 |
| final cost | 6 | 6 |
| wall time | 3.02s | 2.97s |
| Solving rounds | 8 | 7 |
| train problems | 2 | 2 |

Gripper is state-space-, not grounding-bound and needs almost no complexity climbing, so the
flag has negligible effect here — one fewer round, same policy, same cost.

### `domains/suites/blocks3ops-local` (10 problems, `timeout 30m`)

| | off (default) | `--add-problem-after-success` |
|---|---|---|
| solved | **9/10** (blocks-005-2 unsolved) | **10/10** |
| failure reason | `memory` | none |
| final cost | 4 | 5 |
| wall time | 147.96s | 43.34s |
| peak memory | 3564MB | 532MB |
| Solving rounds | 31 | 13 |
| train problems | 6 | 4 |

This is the domain the hypothesis targets, and the effect is large: off, the loop keeps
re-solving the 5-6-problem training set at climbing complexity (up to complexity 3 on 257
states) until it runs into `--max-memory`'s cap, is recorded as a `memory` failure, and never
gets to add `blocks-005-2` to the training set at all. On, each success adds the next unsolved
problem right away, stays at complexity 2 the whole run, and reaches all 10 problems — 3.4x
faster, ~6.7x less memory, less than half the training set size, at the cost of one extra unit
of feature complexity in the final policy (5 vs. 4).

## Round-sequence difference (blocks3ops-local)

- **Off**: `Solving` rounds climb complexity 2→3 on the growing training set between every
  problem addition (`... 3 states ... 40 states ... 257 states`), each round re-grounding the
  full training set at the next complexity level; the round at complexity 3 with all 6 training
  problems (257 states) is where memory runs out.
- **On**: every `Solving` round stays at complexity 2 (`enforced complexity 2` throughout); the
  only rounds that repeat the same training set are frontier-expansion retries (states grow
  3→10→26→...→135 within one problem's frontier loop), and each success moves straight to the
  next unsolved problem instead of climbing.

## Caveats

- Small suites, one seed, one domain pair — a single data point per domain, not a sweep. The
  effect should be re-measured on the full `blocks3ops` / `blocks3ops-heldout` suites and on a
  state-space-bound domain under load (miconic, gripper at scale) before treating the flag as a
  default.
- The final policy is one feature-complexity unit more expensive on blocks3ops-local (5 vs 4).
  Skipping the climb trades policy compactness for reaching more problems at all; whether that
  trade is worth it in general depends on whether the benchmark cares more about solved count or
  about minimal feature cost.
- This only matters for `use_example_plans` configurations where the add-a-problem branch and
  the complexity ladder actually compete; `state`/`trans`/`d2l` configs have empty plan
  iterators and are unaffected regardless of the flag.
- Not benchmarked against `reset_complexity_on_state_space_change` (currently off by default);
  the two features are orthogonal but their interaction on a grounding-bound domain is untested.
