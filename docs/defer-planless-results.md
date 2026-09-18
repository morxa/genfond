# Defer problems the planner cannot plan for (H25): implementation

Date: 2026-09-18. Branch: `hyp/defer-planless` (parent: `hyp/no-maxc-stop`, commit `e9bbdac`).

**Status: implemented and unit-tested. No benchmark numbers yet** — this document describes the
mechanism and the decisions it rests on, not a measured result. A/B it per
`docs/experiments-log.md` before drawing any conclusion about coverage.

## The observation

sokoban, `--type datalog-sig` (`use_example_plans: true`):

* SIW finds no plan for the **smallest** problem in the suite
  (`SIW found no plan, generating no example plans`);
* the first round's training set is therefore that problem **alone**, with an empty plan list;
* `StateSpaceGraph` is only plan-restricted while there are plans to restrict it to, so it falls
  back to the **unrestricted** expansion of an 8-object sokoban instance;
* the feature-pool stage never returns — 12 h, 4 h, repeatedly. Neither the solve budget
  (`solve_time_limit`, enforced inside `Solver.solve`) nor the graceful stop
  (`shutdown.stop_requested()`, polled in the round loop and in `Solver.solve`) is reached from
  inside that expansion, so a SIGTERM is only seen once the expansion finishes on its own.

The run produces nothing at all. The failure is structural, not a matter of tuning: the problems
the planner cannot solve are exactly the hard ones, so the fallback picks the *worst* possible
instance to expand exhaustively.

### Where the fallback actually is

`StateSpaceGraph.__init__`, in one line:

```python
if plans:
    plans = [plan.instantiate(domain) for plan in plans]
else:
    plans = []
```

With `plans == []` every plan-restriction branch in the expansion loop is dead
(`elif plans and not matches_plan: ...`), so every successor is queued and the whole reachable
state space is enumerated. Two things follow that are easy to get wrong:

* **`frontier_expansion` does not cause this.** The frontier flag only chooses between
  `Alive.PRUNED` and `Alive.DEAD` *inside* the `plans and not matches_plan` branch. With no plans
  that branch never runs, so frontier expansion neither triggers nor mitigates the fallback. It
  can only make an already plan-restricted expansion bigger.
* **It is not the planner being slow.** The planner had already returned (with nothing). The
  hours are spent in `feature_generator.FeaturePool.__init__` → `generate_state_space`.

## The mechanism

### 1. `defer_planless_problems` (default **true**, inert without `use_example_plans`)

The gate lives in `ProblemIterator._defer_planless`: on top of the config flag it requires
`plan_iterators`, which only exist with `use_example_plans`. Without example plans *every*
problem is planless and the unrestricted expansion is the intended behaviour, so the option can
default to true without touching the rule-based `state`/`trans`/`d2l` types.

A problem is **planless** when, after drawing up to `min_number_of_plans` from its stream, it
holds no example plan of any kind. Policy-conformant plans (H27) and policy-prefix plans (H30)
count: they restrict the state space just as well as a planner plan, so a problem holding one is
never planless even with an empty stream.

Where it hooks in: `_next_addable_problem()`, which every caller invokes immediately before
`_add_next_problem()` — the initial selection, `add_problem_after_success`, the ladder's
add-a-problem branch and the max-complexity continuation all go through it. Resolving a candidate
now *draws its plans*, because whether a candidate is addable at all is decided by whether the
planner gives it anything; the result is cached in `_pending_candidate` so the pair of calls
agrees and no planner work is repeated or wasted.

A planless problem is:

* **not added** to the training set, and logged as `No example plan for <problem>; deferring it`;
* **kept in the evaluation set** — it still counts as unsolved, every candidate policy is still
  executed on it, and the run is not finished without it;
* **retried** the next time a problem is looked for, against a **freshly seeded** planner stream.

### 2. The retry, and why it needs a new stream

A retry against the exhausted stream would be a formality — the generator is done. The fresh
stream is H32's `iterative_solver._fresh_plan_iterator`: the seed is shifted *and* at least two
SIW restarts are requested, because restart 1 is SIW's identity view of the task and the
permutations the seed drives only exist from restart 2 on. A merely reseeded stream would replay
the very search that already found nothing.

`planless_retries` (default **2**) counts *retries*, so a problem gets three attempts in total:
the initial one plus two reseeded retries. After that it goes into `skipped_planless` and
disappears from the addable scan for good — which is also what keeps `_add_next_problem`'s "there
is a problem to add" assertion true.

**One attempt per candidate per resolution.** `_next_addable_problem` excludes the candidates it
has already tried in this pass; without that, all three attempts are spent back to back on the
round that first found the problem planless, and each attempt is a full planner run.

### 3. Every problem planless → a named stop

`stopped_without_training_problems()` reports the one failure mode this can produce on its own:
no problem in the suite has an example plan, so the training set was never non-empty. The run
ends with `failureReason=no_example_plans` and an ERROR line, instead of the unrestricted
expansion that never returned.

### 4. `max_states_per_problem` (default **null**, off) — the safety net

The deferral cannot catch a problem that *has* example plans but whose plan-restricted expansion
is still far too large. `StateSpaceGraph` therefore takes a `max_states` bound: once the node
count reaches it, the expansion is abandoned with a WARNING and `ExpansionLimitExceeded`, which
carries the problem name. `iterative_solver`'s round loop catches it, calls
`ProblemIterator.defer_active_problem`, records `Result.DEFERRED` and retries the **same
configuration** over the training set minus that problem (`_retry_after_defer`, the new
top-priority branch of `__next__`; if the deferred problem was the only one, the next candidate
is added instead).

The bound is checked per *expanded* state, so the final count may overshoot by one state's worth
of successors. That is deliberate: this is a safety net against an expansion that never finishes,
not an exact budget, and one branching factor of slack costs nothing.

Two decisions inside `defer_active_problem`:

* **An expansion cutoff is permanent**, unlike a planless deferral. Every way the run can change
  a problem's state space afterwards — an INC_PLANS draw, a frontier expansion, a
  policy-conformant or policy-prefix plan — only ever *adds* plans, and every added plan can only
  make the plan-restricted expansion larger. A problem cut off at N states would be cut off
  again, at the price of another abandoned expansion, which is precisely the cost being avoided.
* **The refutations go, `max_cost` stays.** The training set *shrinks*, which is not a monotone
  change: a complexity level refuted over the larger set says nothing about the smaller, easier
  one, so `refuted_complexity` drops to the floor. `max_cost` is left alone exactly as on a
  resample — the best policy so far still stands and the next round is still asked to beat it.

`Result.DEFERRED` is a new enum member for exactly this: no solve ran, so it refutes nothing,
tightens nothing and marks nothing solved.

### 5. The stop check inside the expansion (always on)

`StateSpaceGraph`'s expansion loop polls `shutdown.stop_requested()` every `STOP_CHECK_INTERVAL`
(128) popped nodes and raises `ExpansionInterrupted`; the round loop turns that into the same
`stoppedBy=signal` stop it does everywhere else, provisional stats row included. This is
independent of `max_states_per_problem` and of `defer_planless_problems`, because it addresses
the other half of the observation: a run killed inside a multi-hour expansion produced nothing,
not even a provisional row. The flag is a `threading.Event`, i.e. a bare bool read, so the
interval is conservative rather than necessary.

## Stats

| stat | meaning |
|---|---|
| `deferredProblems` | distinct problems deferred at least once (planless or cut off) |
| `planlessRetries` | reseeded planner streams actually spent on retries |
| `expansionCutoffs` | expansions abandoned at `max_states_per_problem` |

All three are exported at the top of every round (so a provisional row written by an early break
carries them) and once more after the loop.

## What this does *not* do

* It does not make an unsolvable problem solvable. A deferred problem stays unsolved and counts
  against coverage; the gain is that the run gets to spend its budget on the problems it *can*
  learn from instead of on one expansion that never ends.
* It does not change any run where SIW finds a plan for every problem — which is every domain in
  the current suites except sokoban. Expect an A/B on those to be a no-op, and treat any
  difference as noise until it reproduces with `THREADS=1` and a fixed seed.
* It does not bound the **planner**. SIW itself can run for a long time on a problem it will not
  solve; that is a separate budget (`planners.siw.max_nodes`) and a separate hypothesis.

## Tests

`tests/test_defer_planless.py`, 17 tests, all fast (stub problems and plan streams at the
iterator level, `simple_blocks` for the real expansion, mocked `solve_step` + planner for the
three end-to-end ones):

* (a) a planless problem is deferred and the next one is added; it stays in the evaluation set; a
  problem holding only policy plans is *not* planless;
* (b) retries draw from a fresh stream, up to `planless_retries`, then the problem is skipped
  permanently and never offered again; a retry that finds a plan adds the problem; end to end,
  the retry stream carries the shifted seed and `restarts: 2`;
* (c) every problem planless → `StopIteration` with no round built at all, and
  `failureReason=no_example_plans` end to end (`solve_step` is asserted never to be called, which
  is what "no unrestricted expansion" means at this level);
* (d) `max_states_per_problem` raises on the real expansion and `null` expands everything;
  `defer_active_problem` shrinks the training set, retries the identical configuration and drops
  the refutations; end to end, a cutoff on the first problem retries the round over the second;
* (e) the stop flag interrupts the expansion loop (`STOP_CHECK_INTERVAL` monkeypatched to 1) and
  an unset flag does not;
* (f) `defer_planless_problems: false` adds the planless problem exactly as before.
