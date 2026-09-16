# min_train_objects validation

## What was built

`min_train_objects` (config key, `--min-train-objects` CLI flag; default `null`) stops
`ProblemIterator._next_addable_problem` from ever selecting a problem with fewer objects than
the threshold as a *training* problem. Everything downstream is unaffected: `solved` still
tracks every problem, `iterative_solver.solve_iteratively`'s testing loop (`for problem in
problems`) and its final verification pass still require the policy to solve every problem
regardless of training membership, and `add_problem_after_success` / `unselect_problems` both
route through `_next_addable_problem` so they inherit the filter automatically. If no problem in
the run meets the threshold, the run falls back to the old (unfiltered) behaviour with a
warning, rather than `_next_addable_problem` returning `None` forever and the iterator raising
`StopIteration` before a single problem is ever added. `null` reproduces the previous behaviour
byte-for-byte. Four new tests in `tests/test_problem_iterator.py` cover: the initial skip, the
`add_problem_after_success` path, the no-problem-meets-it fallback, and that `null` is a no-op.

Gates on the laptop (AC power confirmed on first): `pytest` (193 passed, 1 skipped), `mypy
genfond tests`, `black --check`, `isort --check` all clean.

## Validation setup

Workstation (`desk-03.ml.rwth-aachen.de`), one apptainer run per arm, `-n 1 --max-memory 60000`,
`timeout 40m`, `--seed 0`, all three arms launched together (96 cores, so no contention):

```
--type datalog-sig -n 1 --seed 0 --add-problem-after-success --solve-time-limit 300 \
--role-complexity-offset 2 --concept-complexity-offset 1 \
--config results-ab/extra.yaml   # planners.siw.restarts: 1, max_plans_per_problem: 8
domains/deterministic/blocks3ops/{domain,p002-*,p003-*,p004-*,p005-*,p006-*,p007-*}.pddl
```

Arms: `--min-train-objects` unset / `5` / `7`. Held-out: `scripts/eval_policy.py --seed 0 -i 3`
against `domains/deterministic/blocks3ops-heldout` (12 problems, 8-30 blocks).

## Results

| arm | final training set | trainProblems | solved (in-suite, /30) | cost | rules | wall | held-out (/12) |
|---|---|---|---|---|---|---|---|
| unset (baseline) | blocks-002-1, 003-1, 004-1, 003-4, 003-3, 004-2, 004-3 (2-4 objects) | 7 | 30/30 | 8 | 7 | 14.0s | **11/12** |
| `--min-train-objects 5` | did not finish | - | - | - | - | timed out at 40min | N/A (no policy produced) |
| `--min-train-objects 7` | blocks-007-1, blocks-007-2 (7 objects) | 2 | 8/30 | 22 | 26 | 2404.6s (40min, killed) | **0/12** |

### unset baseline
Escalates through 2-4-object problems as designed, converges normally (`maxcomplexity`,
`solveOptimal=True`) in 14 seconds, solves all 30 in-suite problems, and generalizes almost
perfectly to the held-out suite: 11/12, the sole failure being `blocks-heldout-020-2`.

### `--min-train-objects 5`
Confirms the config is honored (`min_train_objects=5: starting the training set at
blocks-005-1`), climbs to an 11-problem training set (all of blocks-005-*, blocks-006-*, plus
blocks-007-1) at complexity 6, but never converges: grounding/solving each round over that set
got progressively more expensive (successive OPTIMAL solves at complexity 6 took 97s, 61s, 48s,
33s, 24s — a lazy-pairs loop still grinding) and `timeout 40m` killed the run mid-round. No
`.policy` file was written (the process never returned from `solve_iteratively`), so no
in-suite or held-out numbers exist for this arm. This is a genuine cost finding, not a bug: the
5-7 object mid-size instances are dramatically more expensive to ground under this config than
either extreme.

### `--min-train-objects 7`
Also honored (`min_train_objects=7: starting the training set at blocks-007-1`), but only ever
added a second problem (blocks-007-2) before the 40-minute wall clock hit. Unlike the `5` arm it
returned a policy, because `SIGTERM` is handled gracefully (`iterative_solver`'s
`stop_requested()` cancels the in-flight clingo solve and falls through to the normal
verification/pickling tail) rather than the double-`SIGTERM` immediate exit the `5` arm hit —
this looks like circumstance (which round `timeout` happened to interrupt), not something the
`min_train_objects` change controls. The stats mark this candidly as non-convergent:
`solveOptimal=False`, `stoppedBy=signal`, `failureReason=maxcomplexity`. That policy (26 rules,
cost 22) solves only 8/30 in-suite problems and 0/12 held-out — every held-out failure is "no
matching rule for state," i.e. the 2-problem, complexity-6 policy is both incomplete and
non-generalizing.

## Conclusion

This single-seed run does **not** support the motivating hypothesis. The arm that actually
converged (`unset`, training on the smallest instances) generalized best (11/12); the arm that
excluded them (`min-train-objects 7`) generalized worst (0/12) — but that arm is confounded by
non-convergence: it was cut off by the 40-minute budget with a training set of only 2 problems
and an admittedly sub-optimal policy, not a policy the search had finished refining. The
`min-train-objects 5` arm didn't even produce a policy: under this config (`datalog-sig` with
lazy pairs and offset complexities), the mid-size 5-7 object instances are far more expensive to
solve than either the tiny instances `unset` stays on as long as possible, or the single large
instance `min-train-objects 7` starts from. Whether excluding small instances helps
generalization is untested here — the mechanism works as specified (confirmed by the log lines
and training-set membership above), but this config/budget never let a `min_train_objects`-driven
run reach a converged, comparable policy to test that against the strong `unset` baseline. A
follow-up would need either a much larger wall-clock budget, a cheaper config (no
`role_complexity_offset`/`concept_complexity_offset`, or a non-`datalog-sig` type) to let the
mid-size training rounds actually finish, or a threshold that starts training higher but still
lets multiple problems accumulate within the budget.
