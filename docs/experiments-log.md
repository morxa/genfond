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

_Results pending._
