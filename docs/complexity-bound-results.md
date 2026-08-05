# Complexity bounds after a state-space change: implementation and benchmark results

Date: 2026-08-05. Branch: `learn-from-examples` → `worktree-complexity-reset`.

## The bug

`min_feature_complexity(c)` (`solve_datalog.lp:109`) forbids models in which every selected
feature, concept and role has complexity below `c`. It is justified by exactly one fact: that
complexity `c-1` was **refuted** — proved to admit no solution — over the full feature pool.

That is a claim about one particular ASP instance. `iterative_solver.solve_iteratively` passed
`enforce_highest_complexity=not one_shot`, a constant, so the constraint was grounded in every
iterative round regardless of whether the instance had changed underneath it.

It does change, in two branches of `ProblemIterator.__next__`:

- `INC_PLANS` adds an example plan.
- `EXPAND_FRONTIER` adds plans and dead ends found by expanding frontier states.

Neither is monotone. A plan adds alive states that must now be solved, but it also expands
states that bring new transitions to states already in the space, so a *simpler* policy can
become possible. And the escalation ladder adds a plan *between* complexity levels, so under
any given plan set only one level is ever tried. From the first plan addition onward, every
enforced bound rested on evidence gathered from a different state space.

The impact is completeness, never correctness: the constraint can only exclude models, and the
final policy is re-validated by `execute_policy` on every problem. Scope is `use_example_plans`
configurations (`datalog`) — elsewhere the plan iterators are empty and neither branch can fire.

## The fix

`ProblemIterator` now tracks `refuted_complexity` and returns `enforce_highest_complexity` per
round instead of the caller hardcoding it.

- Only a **full feature pool** round refutes a level. A restricted-generator round proves
  nothing about the unrestricted superset.
- A `SUCCESS` also refutes its level: clingo minimises feature cost, so `cost[-1]` is optimal
  for that pool and nothing there beats the new `max_cost`. Keeping this is what preserves the
  `max_cost < complexity` short circuit in `solve()`, which is guarded on enforcement.
- `INC_PLANS` and `EXPAND_FRONTIER` invalidate. The add-a-problem branch keeps its bound,
  because adding an instance is monotone — a selection solving the larger set solves every
  subset — so it resumes at `succ_complexity` with `refuted_complexity = succ_complexity - 1`.
  `unselect_problems` *replaces* the training set instead of extending it, and then nothing
  carries over.
- Refutations are relative to the current `max_cost`; both branches that loosen `max_cost` back
  to `MAX_COST` also reset `refuted_complexity`, so the two never get out of step.

## The flag

`reset_complexity_on_state_space_change` (default off) additionally restarts the sweep at
`min_complexity` on a state-space change. The two settings avoid the stale bound by opposite
routes:

| | flag off | flag on |
|---|---|---|
| `complexity` | stays at `c` | → `min_complexity` |
| next round | complexity `c`, unenforced | complexity `min_complexity`, enforced |

**It does not change which policies are reachable.** The feature pool at complexity `c`
contains every feature of complexity below `c`, and the round right after a state-space change
runs without `min_feature_complexity`, so a policy expressible at complexity 2 is found by the
round at complexity 4 as well — and preferred, since the `#minimize` is over feature cost.
What the restart changes is the cost of getting there: the low-complexity instances are far
smaller, and they may still ground where the high-complexity one hits `Id out of range`. That
is the only way it can change an outcome rather than a runtime, and it is what the benchmark
has to measure.

`sweep_target` guards the interaction with plan addition: the next plan is only added once the
sweep has climbed *past* the level the last one was added at. With `>=` instead, the restart
and the addition alternate at one fixed level and the search never reaches the higher
complexities at all — caught by `test_restarted_sweep_gets_one_level_deeper_per_plan`.

## clingo's parallel mode makes A/B runs incomparable

clingo may return any optimal model. Under frontier expansion the model decides which states
are handed back to the planner, so the choice propagates into the training set and two runs of
the *identical* configuration diverge. Measured on 10 blocks3ops problems, two repetitions each:

| threads | repetition A | repetition B | round sequence |
|---|---|---|---|
| `-n 8` | 6 rounds | 4 rounds | **diverged** |
| `-n 1` | 4 rounds | 4 rounds | identical |

`run_benchmarks.bash` gained `THREADS` for this; set `THREADS=1` whenever two runs are to be
compared.

### What that did to the first benchmark batch

Four runs at `-n 32`, frontier on:

| run | solved | wall | rounds | unenforced | INC_PLANS | frontier | `Id out of range` |
|---|---|---|---|---|---|---|---|
| reset-off, 35 problems | 19/35 | 3384s | 18 | 0 | 0 | 7 | 3 |
| reset-on, 35 problems | **25/35** | 4927s | 20 | 0 | 0 | 7 | 4 |
| reset-off, 95 problems | **19/95** | 3269s | 18 | 0 | 0 | 5 | 3 |
| reset-on, 95 problems | 15/95 | 3191s | 18 | 0 | 0 | 5 | 3 |

`reset-on` wins by 6 on the subset and loses by 4 on the full set. `unenforced=0` and
`inc_plans=0` in all four: **neither mechanism under test was ever reachable**, so all four runs
executed the same algorithm and the whole spread is thread nondeterminism. The 25/35 also beats
the previous study's best blocks3ops result (22/35) and should not be quoted as progress.

### Why blocks3ops cannot exercise either mechanism

Frontier expansion only fires while `max_cost == MAX_COST` — `max_prune_cost` returns 0 once a
policy exists — which is only before the first success on the current training set. At that
moment `complexity` has just been reset to `succ_complexity`, the lowest level that set will be
tried at. So frontier expansions land at the *lowest complexity tried for the current training
set*, which was 2 here; the bound holds by definition at `min_complexity`, and resetting 2 → 2
is a no-op.

Verified on the ordered trace: rounds 9–13 are `EXPAND_FRONTIER` at complexity 2 and the
triggering round is also at complexity 2. Complexity did reach 3 and 4, but every such round
had `max_cost` tightened to 3, 8 or 9 — gate closed. The one exception, complexity 3 with
`max_cost=MAX_COST`, returned a policy instead of a frontier model.

`INC_PLANS` never fires because `Id out of range` (clingo's 32-bit ground-fact id ceiling)
raises `OUT_OF_RESOURCES`, which the branch excludes; the run ends after ~1h of its 12h budget.
blocks3ops is bounded by grounded program size, not by the complexity sweep, so neither change
here can move it. It is retained below only as a negative control.

## Deterministic results

All at `-n 1`, one partition, frontier on, 12 h / 120 GB, `--type datalog`.

| domain | arm | solved | wall | rounds | unenforced | frontier | max complexity | final cost |
|---|---|---|---|---|---|---|---|---|
| gripper (30) | off | 30/30 | 39s | 8 | 1 | 3 | 5 | `[0, 6]` |
| gripper (30) | on | 30/30 | 46s | 10 | 0 | 3 | 5 | `[0, 6]` |
| miconic (25) | off | 12/25 | 7923s | 31 | 1 | 1 | 11 | `[0, 17]` |
| miconic (25) | on | 12/25 | 8152s | 33 | 0 | 1 | 11 | `[0, 17]` |
| blocks3ops (95) | off | 21/95 | 3374s | 19 | 0 | 4 | 4 | `[0, 10]` |
| blocks3ops (95) | on | 20/95 | 3176s | 19 | 0 | 4 | 4 | `[0, 10]` |

**The restart never changed an outcome.** Every arm found the identical final policy cost, and
coverage is equal in every domain. The one apparent difference — blocks3ops 21 vs 20 — is not
the flag: both runs have byte-identical iteration traces and the identical policy sequence, and
differ only in `blocks-006-1` flaking under the unseeded execution RNG (see below).

**It costs two rounds every time it fires**, which is the re-climb from `min_complexity` back to
where the state space changed: +2 rounds and +18% wall on gripper, +2 rounds and +3% on miconic.

The mechanism did fire, in miconic, and the traces show exactly the intended behaviour:

| round | reset-off | reset-on |
|---|---|---|
| 20 | `START`, complexity 4 | `START`, complexity 4 |
| 21 | `EXPAND_FRONTIER`, complexity 4, **unenforced** | `EXPAND_FRONTIER`, **complexity 2**, enforced |
| 22–23 | complexity 5, 6 | complexity 3, 4 (re-climb) |

This is the frontier expansion at complexity 4 — the only state-space change above
`min_complexity` in the whole study. Note `succ_complexity` is 4 in miconic (every `START`
restarts there), which is why the expansion lands at 4 and not at 2 as in blocks3ops.

The result matches the theory: because the pool at complexity 4 contains every complexity-≤4
feature and the unenforced round considers all of them, retreating to complexity 2 cannot find
anything the complexity-4 round misses. It could only pay off where the larger instance fails
to ground, and in these three domains it never did — gripper and miconic ground comfortably,
and blocks3ops dies at complexity 4 in *both* arms.

**Recommendation: keep `reset_complexity_on_state_space_change: false`.** Revisit only for a
domain that hits `Id out of range` at a complexity the search has already climbed past, where
the smaller instance would still ground.

## The solved count is itself noisy

Policy execution is randomized — `execute_datalog_policy` tries rules in `random.sample` order,
shuffles object bindings, and draws successors with `random.choice` — and the global RNG was
never seeded. A problem counts as solved only if all `policy_iterations` attempts succeed, so a
borderline problem flakes.

The deterministic blocks3ops pair demonstrates it cleanly: identical traces, identical policy,
21/95 vs 20/95, the entire difference being `blocks-006-1`. `seed` (config, or `--seed`) now
fixes the sequence; it defaults to `null`, keeping the old behaviour. **Set it for any run that
will be compared against another.** Two seeded runs reproduce exactly, down to the mean plan
length.

## Reproducing

See the cluster section of `docs/frontier-expansion-results.md` for the host, partition and
paths; the same helpers apply. Configs are `claude-experiments/reset-off.yaml` and
`reset-on.yaml`, which set `frontier_expansion: true` and toggle
`reset_complexity_on_state_space_change`.

```bash
DOMAINS="<domain dirs>" THREADS=1 PARTITION=<one partition> TAG=n1-reset-off \
  CONFIG=claude-experiments/reset-off.yaml VERBOSE=1 ./run_benchmarks.bash
```

Pin `THREADS=1` and a single partition whenever runs must be comparable: the partitions have
different CPUs, and under a `--time` limit a slower node also reaches fewer training problems.
Always pass `--max-memory`; unbounded runs get reaped by an OOM watcher with no error in the log.
