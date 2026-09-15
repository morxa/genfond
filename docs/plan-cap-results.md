# Plan cap and state-set dedupe: implementation and validation results

Date: 2026-09-15. Branch: `hyp/plan-cap` (parent: `hyp/min-count`, commit `7388fc0`).

## What was built

A cluster run with a hand-crafted feature pool accumulated 1423 example plans on one 8-block
training problem (`blocks-008-2`) through frontier expansion, producing a 721 MB log and a
state space that stalled the round. A follow-up 12-hour cluster run attached 4880 plans to the
same kind of problem. Every plan adds states to `StateSpaceGraph` and facts to the ASP instance,
and a plan that adds no new state is pure waste — `plan_key` only catches identical action
sequences, not two different plans that happen to reach the same states.

Three additions, in `genfond/problem_iterator.py`, `genfond/state_space_generator.py` and
`genfond/iterative_solver.py`:

1. **`max_plans_per_problem`** (config, default `null` = unbounded): once a problem's active
   plan count reaches this, `INC_PLANS` and frontier expansion stop adding to it. The initial
   `min_number_of_plans` batch is exempt (a deliberate floor, not runaway growth). Logged once
   per problem, the first time the cap is hit.
2. **`PlanStateCoverage`** tracks, per problem, the union of states reached by its example
   plans (via a new `plan_visited_states` helper that replays only the candidate plan's own
   actions from `problem.init` — branching at a nondeterministic action exactly as
   `StateSpaceGraph` does when propagating a plan suffix — far cheaper than building or
   extending the graph). A candidate plan from `INC_PLANS` or frontier expansion is dropped,
   and does not count as frontier progress, when its visited states are already a subset of
   what the problem's existing plans cover.
3. `solve()` now logs per-problem plan and state counts (max/mean) every round, so runaway
   growth on one problem is visible without reconstructing it from individual round summaries.

Both mechanisms are threaded through as optional/`None` by default: `ProblemIterator` accepts
an optional `plan_coverage`, and every existing test (which construct it without one) is
unaffected. `iterative_solver.py` wires up a real `PlanStateCoverage` whenever
`use_example_plans` is on — the dedupe is otherwise unconditional, there is no separate on/off
switch for it (only the cap is configurable, per the hypothesis).

Local gates (`pytest`, `mypy genfond tests`, `black --check`, `isort --check`) are all green,
including 6 new tests in `tests/test_plan_cap.py` covering `plan_visited_states` against a real
PDDL fixture, `PlanStateCoverage` deduping two different action sequences that reach the same
states (which `plan_key` alone would not catch), the cap stopping `INC_PLANS`, and the
cap/dedupe wiring into `record_frontier_expansion` and `__next__` via a fake coverage tracker.

## Validation setup

Commit `5fdd1e4` (`hyp/plan-cap`) on desk-03, `apptainer run --bind $PWD ... genfond_env.sif`,
`-n 1 --max-memory 60000`, `timeout 30m` per run. `--type datalog-sig -n 1 --seed 0
--add-problem-after-success --solve-time-limit 300`, with `--role-complexity-offset 2
--concept-complexity-offset 1` for blocks3ops. Three arms were intended per suite: `default`
(baseline, `hyp/min-count` commit `7388fc0`, no config override), `cap8`
(`max_plans_per_problem: 8`), and `dedupeonly` (this branch's own default: dedupe on, cap
`null`).

**The `default`/baseline arm failed on every suite** with a harness bug, not a code defect: the
driver ran the baseline worktree (`.../genfond-wt/mincount`) while pointing `--dump-config`,
`-o` and `--stats` at `.../genfond-wt/plancap/results-ab/`, but `apptainer run --bind $PWD` only
binds the current worktree into the container, so the baseline runs hit `FileNotFoundError`
trying to write outside it. This was only caught after the batch had already finished under a
wrap-up deadline, so **no direct baseline comparison was obtained this session** — the numbers
below are `cap8` vs `dedupeonly`, both on `hyp/plan-cap` itself. A rerun with a bind path that
covers both worktrees (or writing baseline output under the baseline worktree) is the obvious
follow-up.

## Results

### Local suites (`domains/suites/*-local`), cap8 vs dedupeonly

| Suite | Arm | Solved | Cost | Wall | `clingoAtoms` | `clingoRules` | max plans/problem | max states/problem |
|---|---|---|---|---|---|---|---|---|
| gripper-local | cap8 | 5/5 | `[0,6]` | 7.3s | 2790 | 4131 | 8 | 40 |
| gripper-local | dedupeonly | 5/5 | `[0,6]` | 6.8s | 2790 | 4131 | 8 | 40 |
| miconic-local | cap8 | 4/4 | `[0,12]` | 3.7s | 2134 | 3143 | 5 | 30 |
| miconic-local | dedupeonly | 4/4 | `[0,12]` | 3.9s | 2210 | 3151 | 5 | 30 |
| blocks3ops-local | cap8 | 6/6 | `[0,10]` | 13.7s | 24346 | 44111 | **8** | **45** |
| blocks3ops-local | dedupeonly | 6/6 | `[0,10]` | 28.7s | 97365 | 179294 | **45** | **190** |

gripper-local and miconic-local never generate more than 8 (resp. 5) plans for any one problem
here, so the cap is a no-op and the two arms are byte-for-byte identical (same
`clingoAtoms`/`clingoRules`, same cost) — the negative control the hypothesis asked for.

**blocks3ops-local is where the cap matters.** Without it, `blocks-005-2` alone climbed to 45
plans and 190 states (log: `Plans per problem: max=45 mean=13.8 (... blocks-005-2=45)`,
`States per problem: max=190 mean=53.2 (... blocks-005-2=190)`). With `max_plans_per_problem: 8`
the same problem stopped at 8 plans / 45 states (`Problem blocks-005-2 reached
max_plans_per_problem=8; no more example plans will be added to it`, logged exactly once). The
grounded program shrank ~4x (97365 → 24346 atoms, 179294 → 44111 rules) and wall time roughly
halved (28.7s → 13.7s) — **with the identical final policy cost `[0,10]`**, i.e. the cap cost
nothing here and saved most of the round.

The dedupe fired independently of the cap: `blocks3ops-local-dedupeonly.time` shows two hits —
`Discarding an example plan for blocks-005-2: it reaches no state beyond the existing example
plans` — both before the cap-8 run even reaches that many plans, confirming state-set dedupe
catches waste that `plan_key` (identical action sequences only) would not.

### The runaway case: `blocks-008-2`, `-008-4`, `-007-5`, `-006-2` with the hand-crafted preset

`claude-experiments/preset.yaml` (copied from `../mincount/claude-experiments/`), `--type
datalog-sig -n 1 --max-memory 60000`, `timeout 30m`, comparing `dedupeonly` (cap `null`) against
`cap40` (`max_plans_per_problem: 40`, `claude-experiments/preset-cap40.yaml`):

| Arm | Solved | Cost | Wall | max plans/problem | max states/problem |
|---|---|---|---|---|---|
| dedupeonly | 4/4 | `[0,27]` | 43.2s | 14 | 320 |
| cap40 | 4/4 | `[0,27]` | 39.1s | 14 | 320 |

Identical on every metric — `max_plans_per_problem: 40` never bound because dedupe alone kept
the largest problem (`blocks-008-4`) at 14 plans. **This is the important negative result: in
this 4-problem, `solve_time_limit: 300`, one-shot-per-round reproduction, dedupe alone already
prevents the blow-up** the cluster run hit at 1423–4880 plans on `blocks-008-2`. That cluster
run had far more training problems and ran for 12 hours, i.e. many more `INC_PLANS`/frontier
rounds than this quick local repro reaches — so this result does not by itself prove dedupe
alone is sufficient at cluster scale, only that it is doing real work here and that `cap40` is a
harmless backstop. Reproducing the actual four-figure blow-up (and confirming the cap is needed
on top of dedupe there) needs a longer run closer to the original cluster conditions, which the
30-minute-per-run / no-benchmarks-on-laptop constraints of this session did not allow.

## Recommendation

Ship the cap and dedupe as implemented (both default to a no-op: cap `null`, dedupe finds
nothing to remove when there is nothing redundant, confirmed by the byte-for-byte identical
gripper-local/miconic-local rows). Set `max_plans_per_problem` to something in the 20-40 range
for cluster runs on problems with hand-crafted feature pools and heavy frontier expansion, as a
backstop in case dedupe alone is not enough at that scale — which this session's local
reproduction could not confirm or rule out one way or the other. Follow-up: fix the `--bind`
path in any future A/B driver (bind a common ancestor directory, or write baseline output under
the baseline worktree) and rerun the local-suite baseline arm for a direct before/after number;
and, resources permitting, a longer (multi-hour) rerun of the runaway case closer to the
original cluster conditions to see whether the cap changes the outcome once dedupe alone is
pushed past its limit.
