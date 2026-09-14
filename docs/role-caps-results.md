# Concept/role complexity offsets

Date: 2026-09-14. Branch: `hyp/role-caps` (based on `learn-from-examples`).

## Approach

`FeaturePool` previously forced all five of dlplan's `generate_features` complexity limits
(concept, role, boolean, count-numerical, distance-numerical) to the same per-round
`max_complexity`. Roles cost `n*m^2` grounded values per state and concepts `n*m`, against `n`
for plain features, so at high complexity roles dominate the ground datalog program while
contributing only a fraction of the useful features. Added `concept_complexity_offset` and
`role_complexity_offset` (default `0`, in `genfond/config/default.yaml`, mirrored as
`--concept-complexity-offset`/`--role-complexity-offset` CLI flags), which cap the concept/role
limit at `max(1, complexity - offset)` each round while boolean/numerical limits stay at
`complexity`. `FeaturePool.__init__` logs the five effective limits at INFO once per round.
Confirmed the installed `dlplan/generator/__init__.pyi` stub really does declare the positional
order `(concept, role, boolean, count_numerical, distance_numerical)`, matching the existing
call.

## Sanity runs

All runs: `-n 1 --seed 0 --max-memory 4000`, `--type datalog`. "Baseline" is the unmodified
`learn-from-examples` code at `/home/thofmann/code/genfond` (commit `e39b6a7`, read-only,
untouched by this work); "default offsets" is this branch with `concept_complexity_offset` /
`role_complexity_offset` left at 0, which should (and does) reproduce the baseline exactly.

### `gripper-local` (5 problems, `problem-*.pddl`)

| Config | Solved | Final policy cost | Wall time | Final round pool (features / concepts / roles) |
|---|---|---|---|---|
| Baseline (unmodified) | 5/5 | [0, 6] | 11.25s | 8 / 32 / 30 |
| This branch, default offsets | 5/5 | [0, 6] | 11.29s | 8 / 32 / 30 |
| `--role-complexity-offset 2` | 5/5 | [0, 6] | 10.15s | 8 / 32 / 13 |
| `--role-complexity-offset 2 --concept-complexity-offset 1` | 5/5 | [0, 7] | 7.41s | 8 / 20 / 13 |

Capping roles alone cost nothing here (same policy, same cost) and shaved ~1s and more than
half the role pool at the final complexity (30 -> 13). Also capping concepts cost one extra
unit of feature complexity (a slightly larger policy) but ran fastest, since the solver never
grounds the complexity-5/6 rounds the uncapped/role-only runs still attempt while searching for
a cheaper policy.

### `blocks3ops` (`p002-1`, `p003-1`, `p004-1`)

| Config | Solved | Final policy cost | Wall time | Final round pool (features / concepts / roles) |
|---|---|---|---|---|
| Baseline (unmodified) | 3/3 | [0, 4] | 0.58s | 3 / 10 / 12 |
| This branch, default offsets | 3/3 | [0, 4] | 0.61s | 3 / 10 / 12 |
| `--role-complexity-offset 2` | 3/3 | [0, 4] | 0.56s | 3 / 10 / 2 |
| `--role-complexity-offset 2 --concept-complexity-offset 1` | 3/3 | [0, 4] | 0.56s | 3 / 6 / 2 |

`blocks3ops` solves at complexity 2 in this configuration, near the `max(1, ...)` floor, so both
offsets shrink the role/concept pools substantially (12 -> 2, 10 -> 6) with no cost or
solvability change and only noise-level timing differences at this scale.

## Caveats

- These are small, single-seed sanity runs (per instructions, no full benchmark suite was run,
  and another benchmark job was already using up to 24GB on this machine) — they demonstrate
  the mechanism is wired correctly and does not regress the tried cases, not that role/concept
  offsets are generally a good trade on the benchmark suites. `blocks3ops` here never leaves
  complexity 2, so it barely exercises the offsets; `gripper-local`'s complexity-4 round is the
  more informative data point.
- The `--concept-complexity-offset 1` case traded one unit of feature-cost for wall time on
  gripper; whether that trade is worth it is domain- and policy-dependent and was not swept.
- No full-benchmark-suite run was performed, per the task's memory note that research-code runs
  should be reported, not iterated on to chase a better number.
