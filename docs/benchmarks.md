# Benchmark domains

`domains/` holds the benchmark instances used for manual runs and the SLURM/workstation benchmarks (the test
suite uses its own fixtures under `tests/fixtures/`). Layout, after the consolidation of 2026-09-17:

```
domains/
  deterministic/<domain>/domain.pddl + p*.pddl     training / benchmark instances (datalog, datalog-sig, d2l)
  deterministic/<domain>/test/*.pddl               larger held-out instances, same domain file (optional)
  non-deterministic/<domain>/...                   FOND domains (state / trans types), untouched
  suites/<name>/                                   small local A/B suites (symlinks into deterministic/)
```

Rules: one directory per domain *encoding*; `test/` never contains a domain file and is not picked up by
`run_benchmarks.bash` (it lists `*.pddl` non-recursively) — pass it explicitly for generalisation checks
(`scripts/eval_policy.py`). Every `(problem NAME)` is unique within a directory, because genfond's
bookkeeping is keyed by problem name.

## Deterministic domains

| domain | train | test | origin / notes |
|---|---|---|---|
| barman | 30 (16–27 objects) | – | IPC 2011 encoding (`:action-costs` declared, unused) |
| blocks3ops | 95 (2–20 blocks, 5 per size, `generate.py`) | 12 (8–30 blocks, seed 7) | 3-operator blocksworld (`newtower`, `stack`, `move`); the main study domain |
| blocks-atomic | 107 (5–30 blocks) | – | D2L "blocksworld-atomic" (`move`, `move-to-table`); a different encoding, kept as its own domain |
| blocks4ops / -clear / -on | 95 / 95 / 190 (`generate.py`) | – | 4-operator blocksworld, one domain file, three goal families |
| childsnack | 25 | – | IPC 2014 (see `notes.md`: two instances are unsolvable by design) |
| delivery | 225 (3–29) | 30 (12–85, from D2L) | |
| depot | 22 | – | IPC 2002 |
| floortile | 21 | – | IPC 2011 |
| grid | 14 (7–48) | 17 (11–98; 12 former `test/` + 5 from D2L) | |
| gripper | 30 (5–10) | 20 (16–54) | |
| hanoi | 30 (6–11) | 30 (4–33, from D2L) | |
| logistics | 25 (6–22) | 22 (14–30) | `logistics_dp` was a byte-identical copy and is gone |
| miconic | 25 (3–15, typed) | – | the untyped IPC/D2L copies used an incompatible encoding and were dropped |
| reward | 20 (9–100) | 15 (225–625) | |
| satellite | 36 | – | IPC 2002 |
| sokoban | 20 (57–499) | – | SIW finds no plans here; unsolved by every configuration so far |
| spanner | 140 (6–32) | 22 (23–59) | |
| storage | 30 (7–97) | – | |
| visitall | 500 (1–100, `generate.py`) | – | use a suite for a small subset |

## What was removed and why

- `domains/d2l/` and `domains/deterministic-new/` as directories: their usable content now lives under
  `deterministic/` (moves above); the rest were duplicates or unusable.
- Unparseable / unsupported: `d2l/blocks` (uppercase `:INIT`), `d2l/blocks-tower` (functional STRIPS),
  `d2l/graph-traversal` (empty requirements), `d2l/gridworld` (numeric, `:bounds`), `d2l/gripper-m`,
  `d2l/logistics98` (both use `=` without `:equality`), `deterministic-new/miconic` (typed without `:typing`).
- Duplicate encodings of the 4-operator blocksworld: `d2l/blocks`, `d2l/blocks-clear`, `d2l/blocks-on`,
  `deterministic-new/blocks(+test)`, `deterministic-new/blocks-multiple(+test)` (also had duplicate problem
  names) — `blocks4ops*` covers the same operators with generated instances.
- Same domain, redundant instance sets: `d2l/barman-opt11-strips` (one effect literal differs from
  `deterministic/barman`), `d2l/gripper` and `deterministic-new/gripper` (sizes covered by train/test),
  `d2l/miconic` + `deterministic-new/miconic/test` (untyped encoding), `d2l/reward`, `d2l/spanner-ipc11-learning`,
  `d2l/visitall-opt11-strips`, `deterministic-new/visitall(+test)` (duplicate problem names, sizes covered by
  the 500-instance set), `deterministic-new/logistics_dp` (byte-identical to `logistics`),
  `deterministic/blocks4ops/old`.
- Companion artefacts: `*.pddl.plan` reference plans and empty `derived_predicates.json` files.
- `domains/suites/blocks3ops` (unreferenced symlink); `domains/suites/blocks3ops-heldout` now points to
  `deterministic/blocks3ops/test` (it used to point at the *other* blocks encoding).
- Problem names made unique in `blocks-atomic`, `depot`, `floortile`, `satellite` (35 satellite files shared
  one name) by suffixing the file stem.

`run_benchmarks.bash` now defaults to every directory under `domains/deterministic/` (it used to default to
the untracked, cluster-only `domains/selected`).
