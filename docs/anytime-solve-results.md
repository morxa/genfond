# Anytime solving and core-guided optimisation

## The observation this attacks

On the cluster (128 GB, one thread), `--type datalog-sig` with lazy pair constraints on
blocks3ops shows a very lopsided round: the *first*, relaxation-only solve finishes in 0.2–100 s,
and the *second* — the one that first carries a batch of 5000 violated pairs, ≈400 k `dist/2`
facts — runs for six hours or more without returning. Grounding is not the problem (3–26 MB
instance, memory flat), and neither is satisfiability: what clingo is doing is branch-and-bound
optimisation over the `#minimize` on selected feature/concept/role complexity subject to the
hitting-set constraints, and it gets stuck improving or proving its bound.

`Solver` built its `clingo.Control()` with default options and set only `parallel_mode`, so there
was no way to change either the strategy or to stop and take what clingo already had.

## Mechanism

Two independent knobs, both defaulting to the previous behaviour.

### 1. `clingo_opt_strategy` / `clingo_options`

`clingo_opt_strategy` (CLI `--clingo-opt-strategy`) is passed to the `Control` constructor as
`--opt-strategy=<value>`. `bb` is clingo's own default, so at the default value **no option is
added at all** and the command line is exactly what it was. Anything else is passed verbatim, so
`usc`, `usc,oll`, `bb,dec`, … all work.

* `bb` (branch and bound) finds a model, asserts that the next must be cheaper, and repeats. It
  produces a *stream of improving models*, which is what gives an anytime solve something to keep.
* `usc` (core-guided / unsatisfiable core) raises a lower bound by extracting cores instead of
  lowering an upper bound, and reports a model only once the bounds meet. It often closes
  instances `bb` cannot — but under a budget it is all-or-nothing: cancelled early it usually has
  no model at all. The two knobs therefore do not compose well, which the measurements confirm.

`clingo_options` (CLI `--clingo-option`, repeatable) passes arbitrary further options to the same
`Control`, for experimentation (`--opt-usc-shrink=…`, `--heuristic=Domain`, …). Nothing in the
code inspects them.

### 2. `solve_time_limit`

`solve_time_limit` (CLI `--solve-time-limit`, seconds, `null` = unbounded) makes `Solver.solve`
anytime. Instead of `control.solve(on_model=…)` it runs

```python
with self.control.solve(on_model=self.on_model, async_=True) as handle:
    finished = handle.wait(self.time_limit)
    if not finished:
        handle.cancel()
    res = handle.get()
```

and keeps whatever `on_model` last stored. `Solver` now classifies the outcome into
`SolveStatus.{OPTIMAL, SATISFIABLE, UNSATISFIABLE, UNKNOWN}` and exposes `self.optimal` (the
answer was *proved*: an optimal model, or unsatisfiability), plus `self.timed_out` and
`self.elapsed`. Every solve logs strategy, budget, status, elapsed and best cost:

```
clingo solve [opt-strategy=bb, limit=60s]: SATISFIABLE in 60.00s (timed out), cost [0, 14]
```

Without a budget only `OPTIMAL` and `UNSATISFIABLE` can occur, which are exactly the two answers
the old code's `bool` return distinguished.

## Soundness of the non-optimal path

The claim is: **a model kept from a cancelled solve is a valid policy; what is lost is only the
optimality argument, and every place that consumed optimality has been told.**

*The model is a model.* `on_model` is called only on models, i.e. assignments satisfying every
rule and integrity constraint of the grounded program. Cancelling the search does not retroactively
weaken that: the "best so far" is a genuine model, just not necessarily a cheapest one. So policy
extraction (`generate_policy`) runs on it unchanged, and the outer loop re-validates the resulting
policy with `execute_policy` on every problem regardless.

*The lazy pair loop still returns something feasible for the full problem.* `lazy_pairs` stops
when the current model violates **no** separation pair. That stopping rule is a property of the
model, not of how the model was found, so it holds for a cut-off model exactly as for an optimal
one: whatever the loop returns satisfies every pair the full (eagerly grounded) problem would
impose. `tests/test_lazy_pairs.py::test_a_cut_off_solve_still_yields_a_feasible_selection_but_is_not_optimal`
pins this by forcing the selection the loop returns onto the eagerly grounded program and checking
it stays satisfiable.

*What breaks is the optimality step.* The lazy loop's optimality argument is "the relaxed optimum
is feasible for the full problem, and the relaxation's optimum lower-bounds the full problem's, so
it is a full optimum". A merely feasible relaxed model bounds nothing. Hence the loop reports
`OPTIMAL` only when the **last** solve proved its own cost optimal, and `SATISFIABLE` otherwise.

*Downstream consumers of optimality.* There are exactly two.

1. `ProblemIterator.set_last_result(SUCCESS, cost)` sets `max_cost = cost[-1] - 1`. This is sound
   either way: the next round is asked to beat a cost that was actually achieved, which is a real
   upper bound whether or not it is the optimum. It is a preference, not a claim.
2. The same call sets `refuted_complexity = complexity`, i.e. "no policy at this complexity costs
   less than `max_cost`". *That* is only justified by optimality; it is what licenses
   `min_feature_complexity` (`enforce_highest_complexity`) in later rounds, and a wrong refutation
   silently excludes policies the solver could otherwise have found. `set_last_result` therefore
   takes an `optimal` flag (default `True`, so nothing changes without a budget) and guards only
   the refutation with it.

*A cancelled solve with no model at all is not a refutation.* Mapping it to `NO_SOLUTION` would
have set `refuted_complexity` — the same unsound refutation, arrived at from the other side. It
gets a new `Result.TIMEOUT` instead: `set_last_result` records nothing from it, and
`ProblemIterator.__next__` escalates exactly as it does after `NO_SOLUTION` (next example plan →
unrestricted generators → higher complexity → next problem). `OUT_OF_RESOURCES` was the other
candidate; it was rejected because every escalation branch is gated on
`last_result != OUT_OF_RESOURCES`, so a timed-out round would have skipped straight to adding
another problem instead of trying a cheaper configuration of the same one.

*The `UNSATISFIABLE` path keeps its proof.* clingo reporting UNSAT within the budget is still a
proof, so `solver.optimal` is `True` there and the relaxation argument ("an unsatisfiable
relaxation proves the full problem unsatisfiable") is untouched.

### Statistics

Per round: `solveStatus` (the `SolveStatus` name), `solveOptimal`, and `nonOptimalRounds` (a
running count of rounds that did not prove their answer — absent from the row when every round
did). On success: `bestSolveOptimal`. From the lazy loop: `lazyPairsOptimal` alongside the
existing `lazyPairIterations` / `lazyPairsGrounded` / `lazyPairsTotal` / `lazyDistFacts`. No
existing key changed meaning.

## Measurements

All runs: `--type datalog-sig -n 1 --seed 0 --max-memory 6000 --add-problem-after-success`, one
core, on a single local machine.

### End to end, the `*-local` suites

`base` = defaults (`bb`, unbounded), `usc` = `--clingo-opt-strategy usc`, `tl60` =
`--solve-time-limit 60`.

| suite | config | solved | cost | wall | train problems | max complexity | every round proven optimal |
|---|---|---|---|---|---|---|---|
| gripper-local (5) | base | 5/5 | 6 | 2.35 s | 2 | 4 | yes |
| gripper-local (5) | usc | 5/5 | 6 | 2.12 s | 2 | 4 | yes |
| gripper-local (5) | tl60 | 5/5 | 6 | 2.40 s | 2 | 4 | yes |
| blocks4ops-clear-local (4) | base | 4/4 | 2 | 0.46 s | 2 | 2 | yes |
| blocks4ops-clear-local (4) | usc | 4/4 | 2 | 0.30 s | 1 | 2 | yes |
| blocks4ops-clear-local (4) | tl60 | 4/4 | 2 | 0.43 s | 2 | 2 | yes |
| miconic-local (4) | base | 4/4 | 12 | 0.88 s | 3 | 4 | yes |
| miconic-local (4) | usc | 4/4 | 12 | 0.81 s | 3 | 4 | yes |
| miconic-local (4) | tl60 | 4/4 | 12 | 0.93 s | 3 | 4 | yes |
| delivery-local (4) | base | 4/4 | 9 | 0.99 s | 4 | 4 | yes |
| delivery-local (4) | usc | 4/4 | 9 | 1.03 s | 4 | 4 | yes |
| delivery-local (4) | tl60 | 4/4 | 9 | 1.10 s | 4 | 4 | yes |
| blocks3ops-local (10) | base | 10/10 | 9 | 15.99 s | 4 | 3 | yes |
| blocks3ops-local (10) | usc | 10/10 | **12** | 11.73 s | 5 | 3 | yes |
| blocks3ops-local (10) | tl60 | 10/10 | 9 | 15.35 s | 4 | 3 | yes |

**The longest single clingo solve across all fifteen runs is 0.20 s.** These suites are nowhere
near the regime the change targets, so the honest reading of this table is a null result plus two
sanity checks:

* the 60 s budget never binds, and `tl60` reproduces `base`'s coverage and cost everywhere;
* `usc` costs nothing in coverage and is a shade faster, but on `blocks3ops-local` it ends at cost
  12 from five training problems instead of cost 9 from four. Both policies are re-validated by
  `execute_policy` on all ten problems, so both are correct. As the lazy-pairs write-up already
  noted for its own change, an equally optimal but *different* model in an early round sends the
  escalation ladder down a different path; one suite is one sample.

One thing the table hides: `tl60` is not a byte-identical rerun of `base` even though its budget
never binds. `control.solve(async_=True)` reports a different (equally optimal) model on some
rounds — on `miconic-local` round 1 it returns a different cost-`[0,2]` model, which violates 2
pairs instead of 0 and costs the round one extra lazy iteration. Costs are provably equal at every
such point, so this is the same model non-determinism the dist-sets and lazy-pairs write-ups
report, not a regression; but a run with `--solve-time-limit` set is not comparable
instruction-for-instruction with one without it.

### The hard case: blocks3ops p006-1, one shot at complexity 4

`--one-shot --max-complexity 4` on `domains/deterministic/blocks3ops/{domain,p006-1}.pddl`, which
is the smallest local reproduction of the cluster's stalling round: 1.15 M separation pairs, of
which the lazy loop grounds ~2%.

| config | lazy iterations | clingo (sum of solves) | wall | peak mem | pairs grounded | `dist` facts | cost | every solve proven optimal |
|---|---|---|---|---|---|---|---|---|
| `bb` (default) | 10 | **142.6 s** | 210.7 s | 590 MB | 25,735 | 490,151 | 4 | yes |
| `usc` | 9 | **11.2 s** | 79.3 s | 590 MB | 21,990 | 419,280 | 4 | yes |
| `bb` + 60 s limit | 7 | **115.3 s** | 181.0 s | 588 MB | 17,056 | 249,659 | 4 | yes |

All three end at the same cost 4 policy, which solves the instance.

**`usc` is the result.** It cuts the solving part of the round by **12.7x** (142.6 s → 11.2 s) on
exactly the workload that stalls on the cluster, and no single solve takes longer than 2.0 s where
`bb` has individual solves of 32 s and 30 s. The remaining 68 s of wall clock is state-space
expansion and feature evaluation, which is identical in all three runs and is the other known
bottleneck on this domain — so 210.7 s → 79.3 s is close to the best this knob can do here.

The 60 s budget **never binds**: `bb`'s longest solve in that run is 40.6 s. The `tl60` column is
therefore not a measurement of the anytime path but another instance of the async model
non-determinism noted above — it happens to land on the 7-iteration trajectory the lazy-pairs
write-up recorded (identical 17,056 pairs / 249,659 `dist` facts), where the plain `bb` run here
takes a 10-iteration one. That the two `bb` columns differ by 30 s with the same strategy and the
same instance is a useful measure of how noisy a single sample is on this domain.

### The stalling round reproduced: blocks3ops p005-1 + p005-2 + p006-1, one shot at complexity 4

Three problems at once (5,053 states, 3.76 M separation pairs) is the closest local analogue of
the cluster rounds that hang, and it behaves like them. `timeout 25m` per configuration.

| config | outcome | lazy iterations finished | violated pairs left when the budget ran out | pairs grounded | cost of last model |
|---|---|---|---|---|---|
| `bb` (default) | **0/3, killed at 25 m** | 4 | 80,078 of 3,761,162 | 20,000 | 4 (proven optimal) |
| `usc` | **0/3, killed at 25 m** | 2 | 1,412,518 of 3,761,162 | 10,000 | 0 (proven optimal) |
| `bb` + 60 s limit | **0/3, killed at 25 m** | **24** (22 of them cut off) | **277** of 3,761,162 | 34,934 | 37 (not proven optimal) |

No configuration produces a policy, so on coverage this is 0/3 three times over. What differs is
everything else:

* `bb` finishes four solves — 2.2 s, 47.7 s, 95.1 s, 43.7 s — and then its fifth runs for **21
  minutes without returning**, having reported its first model 0.7 s in. That is the cluster
  symptom, reproduced in one instance on a laptop.
* `usc` is *worse* here, and instructively so. Its first two solves take 2.1 s and 1.0 s — the
  same 13x speedup as on p006-1 — and then its third burns the remaining 24 minutes and yields
  **nothing**, because core-guided search reports no model until the bounds meet. A stalled `usc`
  solve wastes its whole budget; a stalled `bb` solve at least has a model in hand.
* The 60 s budget converts the stall into steady progress: 24 iterations instead of 4, and the
  violated-pair count falls from 1.53 M to **277**, three orders of magnitude closer to closing
  the loop than `bb` gets. It still did not close within 25 minutes.

The cost column shows the mechanism and its price. Each cut-off model is expensive (16, 29, 34,
36, 33, 38, 45, 25, …, 48, 37 against `bb`'s proven optimum of 4 at the same point), because
what `bb` has after 60 s is a model that selects far more than it needs. An expensive selection
separates *more* pairs, which is why the loop closes faster — the budget is effectively trading
policy cost for loop progress. Had this round closed, the policy would have been valid but far
from minimal, `max_cost` would have been set to `cost - 1` as a preference, and the complexity
level would **not** have been refuted, so the search would have gone on looking for the cheap
policy. That is precisely the semantics the soundness argument above is there to justify.

## Verdict

* **`--clingo-opt-strategy usc` is the change worth taking to the cluster.** On the single-problem
  version of the stalling round it cuts clingo time 12.7x at identical cost, and it costs nothing
  on any of the five regression suites. It is not a cure: on the three-problem round it stalls
  too, and stalls *harder* than `bb` because it has no model to fall back on.
* **`--solve-time-limit` is the safety net, not a speedup.** Every suite here finishes its solves
  in 0.20 s, so the budget is invisible; on the round that hangs it is the only configuration that
  keeps making progress. Its cost is policy quality: the models it keeps are 8–12x more expensive
  than the optimum, and the honest bookkeeping for that (no refutation, `max_cost` as a preference
  only) is what the implementation adds.
* **Do not combine them.** `usc` under a budget is all-or-nothing; cancelled early it has nothing
  to keep, and the round degrades to `Result.TIMEOUT`.
* Untested here: whether a larger budget (say 600 s, which is still 1/4 of what a single stalled
  `bb` solve consumed) closes the three-problem round, and whether `usc` plus a *generous* budget
  is better than either alone. Both need cluster time, not a laptop.
