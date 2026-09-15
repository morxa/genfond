# A graceful wall-clock budget (`--max-wall-time`)

Date: 2026-09-15. Branch: `hyp/wall-budget` (from `hyp/min-count`).

## The problem

A cluster job killed by the SLURM time limit writes neither a policy nor a stats row, and loses
whatever round or clingo solve was in progress -- the only way to see what it had found is to
scrape the last "Policy solves" line out of the log with `awk`.

## Mechanism

Config `max_wall_time` (seconds, `--max-wall-time`) and `wall_time_reserve` (seconds, default
300, config-only). When `max_wall_time` is set, `solve_iteratively` computes one absolute
deadline, `wall_deadline = loop_start + max_wall_time - wall_time_reserve`, and:

1. **Between rounds**: the round loop checks `wall_deadline` (and `genfond.shutdown.
   stop_requested()`, see below) before calling `solve_step` for the next round, and breaks
   instead of starting one. `stats["stoppedBy"]` is set to `"wall_time"` or `"signal"` and
   `stats["wallBudgetUsed"]` records the elapsed time; both are absent when `max_wall_time` is
   unset, so a normal stats row is unaffected.
2. **Inside a round**: the same `wall_deadline` is threaded into `Solver` as `wall_deadline` and
   composes with `solve_time_limit` as whichever is sooner (`effective = min(time_limit,
   wall_deadline - now)`, recomputed on every `Solver.solve()` call since the lazy-pairs loop
   solves the same `Control` repeatedly and the remaining budget shrinks between those calls).
   This reuses the existing anytime/`SolveStatus` machinery `hyp/anytime-solve` built for
   `solve_time_limit`: a solve cut off with a model becomes `SATISFIABLE`/not-optimal (kept, not
   discarded), one cut off before any model is `UNKNOWN` (refutes nothing), and
   `solve_with_lazy_pairs` already reports `SATISFIABLE` vs. `UNKNOWN` correctly for a cut-off
   iteration -- no changes needed there at all, only in `Solver.solve`.
3. **After stopping**: `solve_iteratively` returns exactly as on a normal end (the last
   `SUCCESS`-round's policy, or `None`), so `__main__` falls through its usual tail unmodified:
   pickle the policy, verify it on every problem, write the stats row.

`Solver.solve()` was restructured to *always* solve asynchronously, polling in `POLL_INTERVAL`
(1s) steps instead of one blocking `control.solve()` call, even with no `time_limit` and no
`wall_deadline` at all. That is the only way `genfond.shutdown.stop_requested()` -- set by
SIGINT or SIGTERM -- can reach a solve already in flight: a signal handler only runs once control
returns to the Python interpreter, and a plain blocking call would not return it until clingo
itself finished. With nothing ever cutting it off the polling loop only exits once the solve
completes on its own, reproducing the previous blocking call's result exactly.

`genfond/shutdown.py` is a tiny process-wide stop flag (`request_stop`/`stop_requested`/
`reset_stop`). `__main__.py`'s signal handler calls `request_stop()` for **both** SIGINT and
SIGTERM -- previously SIGINT just logged and re-raised itself as SIGTERM to fall through to the
default (hard-kill) action; now both signals request the same graceful stop. A second signal
force-exits immediately, in case the process is stuck somewhere the flag isn't polled (see
Limitations). `genfond.bash` gained `#SBATCH --signal=B:TERM@600` (SLURM sends SIGTERM 10 minutes
before the `--time` limit) and now forwards it from the batch shell to the `apptainer` child
explicitly, since `B:` targets the batch script only, not srun steps.

## Validation (`desk-03`, `hyp/wall-budget` @ `b315f9f`)

All three cases below ran under `apptainer run ... -n 1 --max-memory 60000`.

**1. `--max-wall-time 60` (default `wall_time_reserve: 300`), blocks3ops-local, `datalog-sig`.**
`60 - 300 < 0`, so the deadline is already in the past when the loop starts -- a deliberately
degenerate case to check the zero-round path without waiting minutes for a real one. Result:
finished in 0.85s, `stoppedBy=wall_time`, `wallBudgetUsed=0.15s`, `totalSolveCpuTime=0`, a
(empty, since no round ever ran) policy pickled, stats row written, and the "Policy solves 0 out
of 10" line present. This run is what caught the bug below.

**2. `timeout -s TERM 90` wrapping the same command family, no `--max-wall-time`.** The
blocks3ops-local suite solves in ~15-25s regardless of the flags used here, so it never actually
hit the 90s mark on the first two attempts -- the default speedup flags
(`--add-problem-after-success --role-complexity-offset 2 --concept-complexity-offset 1`) matter
more to this suite's difficulty than any wall budget. Switched to `domains/deterministic/
blocks3ops/{p005-1,p005-2,p006-1}` with plain `--type datalog-sig` (no speedup flags): SIGTERM
landed mid-round (log: `Received SIGTERM, requesting a graceful stop ...`), the in-flight solve
was cut from a `solve_time_limit=300` budget down to `lastSolveWallTime=69.99s`
(`solveStatus=SATISFIABLE`, `solveOptimal=False` -- proof `Solver.solve`'s poll loop caught the
signal and cancelled), the round loop then saw the pending stop request before starting the next
round (`Stop requested; stopping without starting another round`), and the process exited on its
own ~8.5s after the signal (`real 1m38s` vs. `timeout 90s` -- not SIGKILLed) with
`stoppedBy=signal`, a 9.4KB policy, a stats row, and "Policy solves 2 out of 3".

**3. `--one-shot --max-complexity 4` on `{p005-1,p005-2,p006-1}`, `--max-wall-time 120`.** This
is where it got interesting -- see Limitations: this exact case is dominated by clingo
*grounding*, which `--max-wall-time` cannot cut short, so it did not reproduce the "stalls for
20+ minutes" claim being fixed within 120s+reserve. It was still running after 13 minutes
(11GB RSS, 100% CPU, no log line since "Solving ... 5053 states") and did not react to SIGTERM
for over two more minutes, confirming the stall is inside the uninterruptible `Control.ground()`
call rather than the search; it had to be SIGKILLed. Substituting a case that stalls in the
*search* instead -- the same `{p005-1,p005-2,p006-1}` problem set but iterative (not `--one-shot`)
with `--type datalog-sig --solve-time-limit 300 --max-wall-time 90` and `wall_time_reserve: 10`
-- demonstrates the mechanism directly, with no external signal at all: a round's solve is
visibly cut mid-search (`clingo solve [...]: SATISFIABLE in 21.13s (timed out), cost [0, 8]`),
the very next lazy-pairs iteration is correctly refused any time at all
(`clingo solve [...]: UNKNOWN in 0.00s (timed out), cost -`), the round loop stops
(`Wall-clock budget exhausted (80.1s used of 90.0s, 10.0s reserve)`), and the run finishes
gracefully at 80.6s total with a 412-byte policy, a stats row (`stoppedBy=wall_time`), and
"Policy solves 1 out of 3".

## A bug this validation caught

Case 1 first crashed with `KeyError: 'totalSolveCpuTime'` right after "Total wall time" -- before
writing the stats row at all. `__main__.py` reads that key unconditionally (unlike `bestSolve*`,
guarded by `if policy:`), and it used to always exist because every prior code path called
`solve_step` (which seeds it) at least once: `problems` is never empty, so the round loop's first
`__next__()` always adds a problem and the loop body always ran. The new wall-budget/stop-request
check is the first way to `break` before that ever happens. Fixed with a `stats.setdefault
("totalSolveCpuTime", 0)` right after the loop, and pinned with a test asserting the key exists
on that path (`tests/test_iterative_solver_wall_budget.py`).

## Limitations

**`--max-wall-time` (and SIGTERM) cannot interrupt clingo grounding or feature generation.**
`Solver.wall_deadline` only bounds the `Control.solve()` call; `Control.ground()` (inside
`Solver.__init__`) and `FeaturePool`'s dlplan feature synthesis (before any `Solver` exists) are
both synchronous C/C++ calls with no async or cancel hook in clingo's Python API -- unlike
`solve()`, which exposes exactly that (the `async_=True` + `handle.wait()`/`.cancel()` pattern
this and `solve_time_limit` before it are built on). A run stalled in either of those phases
(validation case 3's literal command, on this domain, at this complexity, per AGENTS.md's own
note that "blocks3ops is limited by ASP grounding") will not honor the budget: the process stays
alive, unresponsive to SIGTERM, until that call returns on its own or the SLURM hard-kill (or an
operator's SIGKILL) arrives. Once it *does* return, a pending stop request or an already-passed
deadline is honored immediately (confirmed: `solve_step`'s next `Solver.solve()` sees
`stop_requested()` and cuts the solve to 0s). This is the one gap between what was asked for
("finish gracefully instead of losing the round in progress") and what clingo's API makes
possible; closing it would need either a grounding-side budget (nothing in clingo exposes one)
or running the solver in a subprocess that can be killed outright and restarted from the last
completed round -- a bigger change than this hypothesis.

**The signal path force-exits on a second SIGINT/SIGTERM.** If a run is genuinely stuck in
grounding/feature-generation as above, a second signal (`os._exit(1)`, no cleanup) is the only
way out short of SIGKILL -- deliberate, but it means a second Ctrl+C also forgoes the graceful
tail.

## Config / CLI

- `max_wall_time: null` / `--max-wall-time SECONDS` -- unset (default) reproduces the previous
  behaviour exactly: no deadline anywhere, `Solver.wall_deadline=None`, same clingo outcome.
- `wall_time_reserve: 300` -- config-only, no CLI flag; held back from the round-stopping
  deadline for the (currently unbounded) final verification pass in `__main__`.
- `scripts/ab_bench.sh` / `run_benchmarks.bash`: `WALL_TIME` env var, passed through as
  `--max-wall-time`; unset by default (byte-identical invocation).
- `genfond.bash`: `#SBATCH --signal=B:TERM@600`, forwarded to the apptainer child.

## Gates

`pytest` (163 passed, 1 skipped), `mypy genfond tests`, `black --check`, `isort --check`: all
green on `hyp/wall-budget`. New tests: `tests/test_shutdown.py`,
`tests/test_iterative_solver_wall_budget.py` (the round-loop gate, with `solve_step` faked out --
fast, no real solve), and additions to `tests/test_solver.py` for `wall_deadline` composing with
`time_limit`, and for `stop_requested()` cutting an otherwise-unbounded solve. One pre-existing
test, `test_a_time_limit_too_short_for_any_model_is_unknown_not_unsatisfiable`
(`solve_time_limit=0.0`), is racy on this laptop under load -- confirmed present and at a
comparable rate on unmodified `hyp/min-count` too (in-process measurement, isolated from process-
startup noise: ~50% either way), so this is pre-existing flakiness in that specific
razor's-edge test, not a regression; left unchanged.
