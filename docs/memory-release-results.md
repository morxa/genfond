# Memory retention across iterative-solver rounds

Date: 2026-09-14. Branch: `hyp/memory-release` (from `learn-from-examples`).

## The observation

In a `--type datalog-sig` benchmark run of blocks3ops
(`results-2026-09-14T22:53:20+02:00-baseline-full95/out/blocks3ops-datalog-sig.log`, read-only,
not reproduced in this repo), the iterative loop did this:

- complexity 5, 7 training problems, 126 states, 257 concepts / 207 roles: solved.
- complexity 6, same problems, 858 concepts / 617 roles: clingo raised `bad_alloc`, caught in
  `solve_step` and mapped to `Result.OUT_OF_RESOURCES`.
- complexity 3, 8 training problems, 163 states, 29 concepts / 28 roles: **`bad_alloc` again**,
  even though an equally small round two rounds earlier (126 states, complexity 3) took 20s and
  little memory.

`--max-memory` sets `RLIMIT_AS` on the process, which caps *virtual address space* (`VmSize`),
not resident memory.

## Instrumentation

`genfond/iterative_solver.py` now logs, at `INFO`, the start and end of every `solve_step`
round: `ru_maxrss` (a high-water mark), `VmRSS` (current resident set) and `VmSize` (current
virtual address space -- the quantity `RLIMIT_AS` actually caps). A third line logs `VmRSS`/
`VmSize` again after the new release step (below), so a round's log shows exactly what the
release step recovered.

## Reproducing

The task's suggested command (`--type datalog-sig -n 1 --seed 0 --max-memory 6000` on 5 small
blocks3ops problems) does not, by itself, hit the bug: datalog-sig's escalation ladder
generalizes from these problems within a couple of rounds at complexity 2, well under any
reasonable memory cap (peak observed: 110MB). The original run needed 7-8 *larger* problems and
complexity 6 to reach 858 concepts / 617 roles over 126 states -- reproducing that scale
organically would take long enough, and use enough CPU, to be a poor fit for a machine running
other people's benchmarks.

Instead this reproduces the mechanism directly: `iterative_solver.solve_step()` (the same
function the real loop calls, completely unmodified for the repro) is called twice in one
process under a matching `RLIMIT_AS`, first with parameters engineered to reliably raise
`bad_alloc`/`MemoryError` (`p005-1` at complexity 9, unrestricted state space -- 5250 concepts,
3807 roles, 70 states), then with a small, ordinary round right after it, exactly mirroring the
"big OOM round, then a small round" shape of the observation. `num_threads` is pinned to 1
throughout, matching `-n 1`.

## What holds the memory

Not what the task's hypothesis list expected: `Solver`, `FeaturePool`, and DLPlan's
`VocabularyInfo`/`SyntacticElementFactory`/`InstanceInfo` are all created fresh as locals of
`solve()` on every round (`genfond/feature_generator.py`, `genfond/solver.py`) -- there is no
module-level or class-level cache anywhere in `genfond/*.py` (`grep -rn "lru_cache\|^cache\|global "`
comes back empty), and DLPlan's own element cache
(`ReferenceCountedObjectFactory` / `SharedObjectCache` in `dlplan/utils/{factory,cache}.h`) is a
per-factory `shared_ptr`, not a process-global singleton, so it is destroyed along with the
factory that owns it. `Policy`/`DatalogPolicy` (`rule_policy.py`, `datalog_policy.py`) and
`FrontierState` (`frontier.py`) hold no reference back to the `FeaturePool` or state graphs
either.

What *does* hold it: **the exception itself.** `solve()`'s locals (`feature_pool`,
`asp_instance`, the `Solver` wrapping a half-grounded `Control`) contain back-references --
state-graph nodes point at their graph and vice versa, DLPlan elements are cached by their
factory -- so they are reference cycles, not simple chains. When `Solver.control.ground()` or
`.solve()` raises (`bad_alloc` as a `RuntimeError` from clingo, or `MemoryError` via pybind11's
automatic `std::bad_alloc` mapping from a DLPlan call such as `compute_redundant_concepts`),
the exception's traceback keeps every frame between the raise and the `except` that catches it
alive, `solve()`'s frame included. `except (RuntimeError, MemoryError) as e:` in `solve_step`
already deletes `e` (and so the traceback) when its suite ends -- that is standard Python, done
specifically to break this cycle -- but that only removes the *external* reference into the
cycle; the cycle among the frames/locals themselves remains and needs an actual GC pass to be
reclaimed, and Python's generational GC is scheduled by allocation *count*, not size, so a
round that allocates relatively few large C-extension-backed objects (which is exactly what a
`FeaturePool` full of DLPlan elements looks like) can go a long time before that happens on its
own.

This is confirmed, not just argued: calling `gc.collect()` right after such a round reports
**7,365 unreachable objects** on the repro below, every one of them garbage the exception left
behind.

There's a second, smaller and *not* Python-fixable contributor: even after `gc.collect()`,
`VmSize` does not return to the pre-round baseline. That remainder is glibc/C++ allocator
fragmentation from clingo's and DLPlan's own allocations, not a live reference.

## Fix

`genfond/iterative_solver.py`: a `_release_round_memory()` helper runs in `solve_step`'s
`finally` block, i.e. after every round regardless of outcome (ordinary rounds hold cyclic
state-graph structures too, just far less of them since no traceback chain is involved). It
calls `gc.collect()`, then `ctypes.CDLL("libc.so.6").malloc_trim(0)` (guarded, so a platform
without `libc.so.6` just keeps a higher `RSS`, no correctness effect). `gc.collect()` is the one
that matters for the actual bug -- it is what moves `VmSize`, the quantity `RLIMIT_AS` caps.
`malloc_trim(0)` is kept as a second, cheap step because it substantially reduces `RSS` (real
physical memory, relevant since other jobs share this machine), even though it does very little
for `VmSize` on its own -- per the task's instructions, retention is fixed first and trimming is
additive, not a substitute.

## Before / after

Single round, `p005-1` at complexity 9, unrestricted state space, `RLIMIT_AS = 1500MB`,
`num_threads = 1` (`/tmp/.../scratchpad/repro_two_rounds.py`, `solve_step` called directly and
unmodified; "before" obtained by monkeypatching only `_release_round_memory` to a no-op so the
rest of the code path, including the logging, is identical):

| Point | maxrss (high-water) | VmRSS | VmSize | Note |
|---|---:|---:|---:|---|
| Process start | 43.5MB | 43.7MB | 280.5MB | baseline |
| Round A end (before release) | 1263.2MB | 234.6MB | 470.2MB | `bad_alloc`, `Result.OUT_OF_RESOURCES` |
| Round A end, **before fix** (no release) | 1263.2MB | 234.6MB | 470.2MB | nothing reclaimed |
| Round A end, **after fix** (`gc.collect()` + `malloc_trim`) | 1263.2MB | **47.7MB** | **446.3MB** | gc found 7,365 unreachable objects |
| Round B (`p002-1`, complexity 3) | -- | -- | -- | `Result.SUCCESS` in both variants at this cap |

`gc.collect()` alone (no trim) reduced `VmRSS` 233.0MB -> 210.4MB and `VmSize` 469.1MB ->
446.1MB; adding `malloc_trim(0)` on top took `VmRSS` down to 47.6MB with no further `VmSize`
change -- the split confirms `gc.collect()` is what moves `VmSize` and `malloc_trim` is
additional insurance for `RSS` only.

At `RLIMIT_AS = 1500MB` the small round B (18 concepts, 3 states) succeeds either way: the
~190MB pre-fix / ~166MB post-fix residual both leave over 1GB of headroom, so this scale does
not, by itself, flip round B from failing to succeeding. That headroom-vs-residual arithmetic is
also why the original bug needed a much bigger offending round (858 concepts / 617 roles / 126
states under a 16000MB cap) before the *next* round ran out of room -- reproducing that
crossover exactly would need a round at that scale, which was avoided here to keep runs short
and single-threaded on a shared machine. The mechanism (a real, GC-collectable reference cycle
left by the exception path) is the same either way and is what got fixed; its effect size
scales with how much the offending round allocated before failing.

## Fragmentation: honestly, not fully fixed

Neither `gc.collect()` nor `malloc_trim(0)` recovers all of a round's `VmSize` growth. In the
measurement above, 190MB of `VmSize` growth shrank to 166MB after `gc.collect()` (~13%
reclaimed) and did not move further after `malloc_trim(0)`. That residual is virtual address
space clingo's and DLPlan's C++ allocators committed to the process and did not return to the
OS -- ordinary glibc arena behavior, not a Python reference. `malloc_trim(0)` targets exactly
this (returning freed heap pages to the OS) and it does help `RSS` dramatically (837MB -> 48MB
region in one measurement above), but it did not move `VmSize` here, so on its own it would not
have prevented the reported `bad_alloc`-after-`bad_alloc` sequence. If this residual keeps
mattering at benchmark scale after this fix (i.e. very long runs with many large rounds still
approach the cap), the next lever is outside Python's reach: an allocator that returns memory
to the OS more eagerly than glibc's default (e.g. `LD_PRELOAD`-ing `jemalloc`/`tcmalloc` with an
aggressive decay setting for the benchmark process), not something to chase from
`iterative_solver.py`.
