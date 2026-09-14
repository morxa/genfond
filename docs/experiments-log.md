# Experiments log: scaling `--type datalog` to blocks3ops

Running record of every hypothesis tried under `auto-experiments.md`. One entry per hypothesis:
the idea, the branch, the numbers against the baseline, and the verdict. Details live in each
branch's own `docs/<name>-results.md`.

## Protocol

Everything is measured with `scripts/ab_bench.sh`, which fixes the things that otherwise make
two runs incomparable (see `docs/complexity-bound-results.md`):

- `--seed 0` — policy execution draws rule order, object bindings and successors from the
  global RNG; unseeded, the same policy reports different solved counts.
- `-n 1` — clingo's parallel mode returns an arbitrary optimal model, and under frontier
  expansion the model chooses which states the planner sees next, so runs diverge.
- `--max-memory` (24000 MB locally, 120000 MB on the cluster) and a wall-clock limit per
  suite × type (`timeout`, default 2 h locally; 12 h SLURM limit on the cluster).
- Fixed problem suites, checked in as symlink directories under `domains/suites/`:

| suite | problems | purpose |
|---|---|---|
| `blocks3ops-local` | p002-{1,2,3}, p003-{1,2,3}, p004-{1,2}, p005-{1,2} (2–5 blocks) | local stage; the 5-block instances are where the baseline dies |
| `blocks3ops` | all 95 (2–20 blocks) | cluster stage; the primary metric is solved/95 |
| `blocks3ops-heldout` | `domains/d2l/blocks3ops`, 105 problems with 10–30 blocks | generalisation check of a learned policy via `scripts/eval_policy.py`; never trained on |
| `gripper-local` | problem-1-{1,2,3}, problem-2-{1,2} | regression |
| `miconic-local` | problem-1-{1,2,3}, problem-2-1 | regression |
| `blocks4ops-clear-local` | p002-1, p003-1, p004-1, p005-1 | regression |
| `delivery-local` | 1x1-1-1-0, 1x2-1-1-0, 2x2-1-1-0, 2x2-1-2-0 | regression |

Metric: problems solved out of the 95 blocks3ops instances (full coverage is the goal, a clear
gain in coverage counts as progress). Regression suites must not lose coverage. Secondary
observations: wall time, peak memory, number of `Id out of range` events, policy cost, and
held-out coverage.

Stages: local (this machine, 24 GB) → an idle desk machine or SLURM `rleap_cpu` with the
apptainer image (`RUNNER="apptainer run --bind $PWD genfond_env.sif"`, or `run_benchmarks.bash`
with `DOMAINS=domains/suites/... THREADS=1 PARTITION=rleap_cpu`).

`python scripts/ab_summary.py results-<a> results-<b>` prints the comparison table.

## Baseline

Branch `learn-from-examples` after merging `worktree-complexity-reset` (seed, `THREADS`,
per-round `refuted_complexity` enforcement) and `reduce-asp-separation-layer` (`trans_diff`
no longer materialised, `--type datalog-sig`). `reset_complexity_on_state_space_change` stays
off (benchmarked as a no-op in `docs/complexity-bound-results.md`).

_Local stage, 2026-09-14, commit e39b6a7, `-n 1 --seed 0 --max-memory 24000`:_

| suite | type | solved | wall | cost | peak MB | clingo atoms | note |
|---|---|---|---|---|---|---|---|
| blocks3ops-local (10) | datalog | 9/10 | 606 s | 4 | 21 290 | 64.1 M | `bad_alloc` at complexity 2 with 6 training problems after 20 frontier rounds; blocks-005-2 unsolved |
| blocks3ops-local (10) | datalog-sig | **10/10** | 175 s | 9 | 5 917 | 8.6 M | solved at complexity 3, 39 rules |
| gripper-local (5) | datalog / datalog-sig | 5/5 / 5/5 | 13 s / 4 s | 6 / 6 | 912 / 98 | | |
| miconic-local (4) | datalog / datalog-sig | 4/4 / 4/4 | 4 s / 3 s | 12 / 12 | 283 / 143 | | |
| blocks4ops-clear-local (4) | datalog / datalog-sig | 4/4 / 4/4 | 1 s / 0 s | 2 / 2 | 85 / 86 | | |
| delivery-local (4) | datalog / datalog-sig | 4/4 / 4/4 | 14 s / 5 s | 9 / 9 | 790 / 315 | | |

**Decision:** `datalog-sig` is at least as good everywhere and strictly better on blocks3ops, so it is the
baseline encoding for all hypotheses from here on. Note that even datalog-sig spends most of its rounds on
frontier expansion at complexity 2 (blocks-005-2 went from 5 to 39 example plans before complexity 3 solved
the set); the frontier loop, not the solver, dominates wall time on the local suite.

Full-95 and held-out results: see below as they arrive.

