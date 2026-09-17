# H26: checkpoint the best policy and stats row against a hard kill

Date: 2026-09-17. Branch: `hyp/combo` → `hyp/checkpoint-best`.

## The problem

`solve_iteratively` (`genfond/iterative_solver.py`) tracks the best-coverage policy the whole run
sees (`keep_best_policy`) and `__main__.py` pickles it to `--output` and appends the stats row to
`--stats` only once — at the very end, after `solve_iteratively` returns and the final
verification loop has run. A graceful stop (SIGTERM → `genfond.shutdown.request_stop`) is polled
between rounds and inside a single clingo solve (`Solver.solve`'s async wait loop), but
*grounding* (`Solver.__init__`'s `self.control.ground(parts)`) is one blocking C call with no poll
point at all. When grounding alone outlasts the SLURM time-limit reserve, the SIGKILL that
follows the SIGTERM warning arrives before `solve_iteratively` ever returns, and a run that had
already found a policy solving most of the suite loses both the policy and the stats row.

## The mechanism

Two independent, immediate write-backs, both gated by one new config flag,
`checkpoint_best_policy` (default `true`, no CLI flag — set it via a `--config` file, matching
`final_pass_max_pool`'s precedent for a config-only knob):

* **Policy checkpoint** (`genfond/checkpoint.py`'s `checkpoint_best_policy`, called from
  `iterative_solver.py` at both places `best_policy` is updated — the round loop and the final
  combination after an optional `_final_cost_minimization_pass`): every time the best-coverage
  policy's *solved count* strictly increases (a cost-only tie-break is not re-checkpointed — it
  does not change what a kill would lose), the policy is pickled to the `output` path
  (`config["output"]`, which mirrors `--output`/`-o`) atomically — write `<path>.tmp`, then
  `os.replace` — and one `INFO` line is logged: `"Checkpointed best policy (N/M) to <path>"`. A
  no-op when `checkpoint_best_policy` is off or no `output` path is configured. `__main__.py`'s
  own two `pickle.dump` calls were switched to the same `atomic_pickle_dump` helper, so every
  write to the output path — checkpoint or final — uses the identical atomic pattern and format
  `scripts/eval_policy.py` already loads.

* **Provisional stats row** (`genfond/checkpoint.py`'s `append_stats_row` /
  `remove_provisional_row`, called from `iterative_solver._write_provisional_stats_row` at the
  three places the round loop notices a graceful stop — an exhausted `--max-wall-time` budget, a
  pending SIGINT/SIGTERM, or in-loop validation hitting either mid-test): a row is appended to the
  `stats` CSV (`config["stats"]`, mirroring `--stats`) immediately, using whatever the `stats`
  dict holds at that instant (now including live `bestSolved`/`bestCost`/`bestRound`, updated at
  the same two call sites as the policy checkpoint, not only once at the very end as before) plus
  `stoppedBy` and a `provisional: 1` column. Every run gets a `runId` (a `uuid4` hex, generated
  once at the top of `solve_iteratively`) so `__main__.py`'s normal end-of-run write can find and
  delete this run's own provisional row (`remove_provisional_row`, matched by `runId` — safe
  under concurrent writers, e.g. a parallel SLURM array job sharing one `--stats` path, since it
  only ever deletes rows tagged with its own id) right before appending the real final row
  (`provisional: 0`), so a run that does finish normally never leaves a duplicate behind. Both
  functions use the same `<path>.lock` `FileLock` the original inline CSV-writing code in
  `__main__.py` used, now factored out so both the provisional and the final write share it.

  One header gotcha this had to work around: a provisional row is written mid-run, before
  `totalWallTime`/`cost`/`memUsage`/... are known, so if it were allowed to establish a brand-new
  file's header, that header would then be too narrow for every later row — including this run's
  own final one — for the rest of the file's life (the existing "dropped" mechanism, which was
  already tolerated for genuinely run-to-run key differences like `failureReason`, would silently
  start eating real columns). `append_stats_row`'s `create_if_missing` parameter is `False` for
  the provisional write, so it is a no-op on a brand-new stats file; the final write always passes
  the default `True`. In practice a stats file is normally shared across many runs (see
  `docs/ab-benchmark-protocol.md`), so this only loses the provisional row on the very first run
  ever appended to a given file, and even a lost provisional row loses nothing that the final
  write would not have written anyway had the run succeeded. (A shared `--stats` path across many
  runs is the normal case for a benchmark suite — see the `scripts/ab_bench.sh` /
  `docs/experiments-log.md` workflow.)

## What this does and does not protect against

Both mechanisms run from inside `solve_iteratively`'s own Python code, at points execution
already reaches when a stop is noticed or a round's result improves — not from inside the SIGTERM
handler itself. Python cannot run a signal handler while a single blocking C call (grounding) has
control, so neither checkpoint can do anything about a kill that lands *during* that one
`ground()` call. What they do provide: the *previous* round's best policy and a provisional stats
row (from the last time a stop was noticed) are already on disk before that risky call even
starts, so a run that has already found a policy solving most of the suite does not lose that
progress to a single oversized grounding call later in the same run.

## Files touched

- `genfond/checkpoint.py` (new): `atomic_pickle_dump`, `checkpoint_best_policy`,
  `append_stats_row`, `remove_provisional_row`.
- `genfond/iterative_solver.py`: `solve_iteratively` gained an `extra_stats` parameter and now
  generates `stats["runId"]`; `_write_provisional_stats_row` (new); checkpoint calls at the two
  `best_policy` update sites (round loop, final combination); provisional-row calls at the three
  `stoppedBy` sites in the round loop.
- `genfond/__main__.py`: passes the initial `stats` dict into `solve_iteratively` as
  `extra_stats`; both `pickle.dump` calls now go through `atomic_pickle_dump`; the inline
  CSV-writing block is replaced by `remove_provisional_row` + `append_stats_row`.
- `genfond/config/default.yaml`: `output: null`, `stats: null` (so the `--output`/`--stats` CLI
  values reach `config` at all — the usual gotcha, see AGENTS.md's config-layering section) and
  `checkpoint_best_policy: true`.
- `tests/test_checkpoint.py` (new): unit tests for `atomic_pickle_dump`/`checkpoint_best_policy`
  (writes when enabled+configured, no-ops when disabled or unconfigured, default-on), for
  `append_stats_row`/`remove_provisional_row` (header handling, the `create_if_missing` guard,
  provisional-then-final leaves exactly one row), and two `solve_iteratively` integration tests
  with `solve_step`/`execute_policy` mocked out (matching `tests/test_final_cost_minimization.py`
  and `tests/test_iterative_solver_wall_budget.py`'s existing pattern): the checkpoint file
  already holds the round-1 policy by the time round 2 starts and ends up equal to the returned
  policy; a pending `request_stop()` produces a provisional row tagged with the run's own `runId`.

## Local gates

`black`/`isort`/`mypy genfond tests`/`pytest --import-mode importlib` all green: 249 passed, 1
skipped (231 passed, 1 skipped before this branch, plus 18 new tests in `test_checkpoint.py`).
