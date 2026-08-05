# Frontier expansion: implementation and benchmark results

Date: 2026-08-05. Branch: `learn-from-examples`.

## What was built

`frontier_expansion` lets the datalog solver select a transition into an *unexpanded* state on
the optimistic assumption that it is solvable. Those states are handed back to the planner; a
returned plan is spliced onto the path from the root and added as a new example plan, and no
plan marks the state as a dead end. A model that uses any frontier transition is not accepted
as a policy — the round is retried once the states have been expanded.

The intended mechanism is *not* merely escaping an unsatisfiable round. A frontier state
imposes no separation constraint, so the solver may select a transition it cannot distinguish
from one it needs to select, instead of paying for features expressive enough to tell them
apart. The point is to **reduce the required feature complexity**.

The ASP side already existed on this branch from the dormant `use_selected_states` path:
`pruned/2`, the relaxed constraint at `solve_datalog.lp:22`, `safe_state(I,S) :- pruned(I,S)`,
the `1@2` minimize (a higher priority than the feature-complexity minimize at `@0`), and
`limit_prune_cost`. Only a projected `frontier/2` plus its `#show` are new.

## Bugs found and fixed along the way

Three were pre-existing and independent of this feature.

| Commit | Bug |
|---|---|
| `f1a791f` | `__main__.py` read `policy.cost[0]`. The `@2` level is absent from `model.cost` when no `pruned/2` fact grounds, so the vector is length 1 or 2 and `cost[0]` reported the frontier count as the feature complexity. |
| `4bc2faa` | `f_distinguished` ranged over `state/2`, not `alive/2`. States without `eval/4` facts vacuously differed in every feature, so every selected feature was attached as a condition to every learned rule. |
| `00e757d` | `--one-shot` raised `TypeError`: `OneShotProblemIterator` returned `selected_states` while `solve_step` takes `example_plans`. |
| `c9d49be` | Off-plan successors that already satisfy the goal were marked dead with `goal=False`, because the goal check only runs on states popped from the expansion queue. |
| `2b0a725` | **Plan-suffix propagation.** See below. |

### The plan-suffix propagation bug

States are expanded in a single LIFO pass, and `add_node` extended an already-expanded node's
`plan_suffixes` without re-queueing it. A plan reaching a node *after* it was expanded
therefore never had the actions it prescribes there matched, and its successors were treated as
off-plan. Instrumented on gripper `problem-2-1`:

```
step 3  move(roomb,rooma)       node=10  suffixes=4  prescribes_next=True
step 4  pick(ball1,rooma,right) node=11  suffixes=1  prescribes_next=False  -> succ PRUNED
```

Adding that plan left the graph unchanged at 40 nodes. `StateSpaceNode` now deduplicates its
suffixes and reports whether it gained any; a node that gains one is queued again. This
terminates because a node only re-enters the queue when its suffix set grows, and that set is
finite.

**This affected the supervised mode with or without frontier expansion** — example plans were
silently under-expanding the state space. Fixing it makes state spaces larger and therefore ASP
instances harder, which shows up as a *slower and slightly weaker baseline*. Results measured
before this fix were computed on incomplete state spaces.

Symptom without the fix, on gripper: the planner is deterministic, so a frontier state that a
plan failed to expand produced the same plan forever — 2 distinct plans added 52 times, plans
growing 18 → 63 while the state count stayed pinned at 50.

## Results

### Local (laptop, `--max-memory 6000 -n 8`), corrected code

| Domain | frontier off | frontier on | frontier rounds |
|---|---|---|---|
| gripper (3 problems) | 2/3, 123s | **3/3, 12s** | 3 |
| miconic (4 problems) | 4/4, 102s | 4/4, **10s** | 1 |
| blocks3ops (7 problems, `--max-complexity 5`) | 5/7, 64s | 5/7, **42s** | 1 |
| blocks4ops-clear (4 problems) | 4/4, 1s | 4/4, 1s | 0 |
| delivery (4 problems) | 4/4, 0s | 4/4, 0s | 0 |

gripper becomes solvable; miconic is 10× faster at equal coverage. The last two rows are the
important negative control: domains that do not need extra state space use **zero** frontier
transitions and pay no overhead, confirming that the `@2` priority keeps the mechanism out of
the way when it is not needed.

### Cluster (terbium, `rleap_cpu`, 32 cpus, 120 GB, 12 h), blocks3ops

| Run | Code | Frontier | Solved | Wall | Peak RSS | `Id out of range` |
|---|---|---|---|---|---|---|
| `frontier-off` | pre-fix | off | 20/95 | 9191s | 116 GB | 4 |
| `frontier-on` | pre-fix | on | 21/95 | 6473s | 102 GB | 3 |
| `fixed-off` | corrected | off | 19/95 | 6225s | 92 GB | 3 |
| `fixed-on` | corrected | on | 18/95 | 3103s | 75 GB | 3 |
| `small-off` (35 problems) | corrected | off | **22/35** | 4584s | 98 GB | 3 |
| `small-on` (35 problems) | corrected | on | 18/35 | 4723s | 90 GB | 4 |

Complexity reached and policy costs on the 35-problem subset:

| | max complexity reached | policy costs found |
|---|---|---|
| `small-off` | 7 | `[2]@2, [4]@2, [5]@3, [8]@3, [9]@3` |
| `small-on` | 5 | `[2]@2, [0,4]@2, [0,6]@3, [0,8]@3` |

Every accepted `small-on` policy has `cost[0] == 0`, i.e. zero frontier transitions, as
designed — the frontier was used during search but never in an accepted policy.

## Interpretation

**Frontier expansion did not help blocks3ops.** Every configuration has its smallest unsolved
instance at `blocks-005-*` and hits `Id out of range` 3–4 times. That error is simply clingo's
32-bit ground-fact id ceiling: the grounded program became too large. blocks3ops sits *at* the
size ceiling, so spending state space there reaches the wall sooner — `small-on` hits it a
fourth time and stalls a training problem earlier than `small-off`.

Where the bottleneck is missing state space rather than grounding size (gripper, miconic), the
mechanism works as intended.

This is the size/expressivity trade-off: grounded program size is driven by the state space
(number of training problems, their size, which subspace is generated) *and* by the feature
pool (max complexity, which generators are enabled, whether concepts and roles are needed).
Datalog needs concepts and roles, and with `n` states and `m` objects the grounded program
holds `n` values per feature, `n·m` per concept and `n·m·m` per role — so concepts and roles
dominate. For blocks3ops, reducing grounded concept and role values buys far more headroom than
reducing states or features.

## Reproducing

Configs and helpers live on terbium under `claude-experiments/`:

```bash
# on terbium, in /work/rleap1/till.hofmann/genfond
DOMAINS=domains/selected/blocks3ops PARTITION=rleap_cpu TAG=fixed-off \
  CONFIG=claude-experiments/frontier-off.yaml VERBOSE=1 ./run_benchmarks.bash

bash claude-experiments/status.sh 3372527:fixed-off 3372528:fixed-on
```

`run_benchmarks.bash` gained `PARTITION` (passes `--partition` to `sbatch`, overriding the
directive in `genfond.bash`) and `TAG` (suffixes the job name and results directory so several
experiments can be told apart).

Always pass `--max-memory`: unbounded runs get reaped by an OOM watcher with no error in the
log.
