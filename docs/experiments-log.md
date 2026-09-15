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

**Held-out** (`scripts/eval_policy.py --seed 0 -i 3`, 12 instances with 8–30 blocks from the same generator,
seed 7): the local-suite datalog-sig policy solves **1/12** (blocks-heldout-015-1); every other instance reaches a
state where no rule applies. A policy learned from ≤5-block instances does not generalise; the training set
must contain larger instances, which is exactly what the grounding ceiling prevents.

**Full 95, local** (`datalog-sig`, `--max-memory 16000`, 3 h limit): **19/95** in 410 s, policy cost 8 from 8
training problems, ended by `Result.OUT_OF_RESOURCES`. Round history that matters:

1. 7 training problems (2–4 blocks, 126 states): success at complexity 3 with cost 8; the policy solves every
   ≤4-block instance and blocks-005-1 but not blocks-005-2.
2. The loop then climbs complexity on the *same* seven problems to beat cost 8: complexity 4 (88 concepts,
   73 roles, 51 s), 5 (257 / 207, 166 s), 6 (858 / 617) → clingo `bad_alloc` at the 16 GB cap.
3. blocks-005-2 is added; the next round at complexity 3 with 163 states and 29 concepts / 28 roles — an
   instance that took 20 s two rounds earlier — hits `bad_alloc` again. An out-of-resources result has no
   escalation branch, so the run ends.

Two conclusions: the post-success cost climb, not the training set, is what reaches the grounding ceiling
(hypothesis `no-cost-climb`); and memory from a failed round appears to be retained into the next one
(hypothesis `memory-release`).


## H1: separate complexity caps for concepts and roles (`hyp/role-caps`)

Idea: roles cost n·m² and concepts n·m grounded values per state against n per feature, so cap them below
the feature complexity. Two new config keys / flags, `concept_complexity_offset` and
`role_complexity_offset` (limit = max(1, complexity − offset)); `include_roles: false` is the extreme.

Local blocks3ops suite (10 problems, datalog-sig, `-n 1 --seed 0`, 8 GB cap):

| arm | solved | wall | cost | rules | peak MB | clingo atoms | training set at success |
|---|---|---|---|---|---|---|---|
| baseline (offsets 0/0) | 10/10 | 175 s | 9 | 39 | 5 917 | 8.6 M | 6 problems, complexity 3 |
| role offset 2 | 10/10 | 333 s | 11 | 51 | 7 586 | 5.3 M | 7 problems, complexity 3 |
| role offset 2 + concept offset 1 | 10/10 | **31 s** | **6** | 31 | **358** | **0.19 M** | 4 problems, complexity 2 |
| no roles | 10/10 | 151 s | 8 | 20 | 3 717 | 0.52 M | 7 problems, complexity 3 |

Regression suites: unchanged coverage in the sonnet sanity runs (gripper 5/5 at cost 6 with role offset 2;
cost 7 with the concept offset). Held-out and full-95 results: below.

Held-out (12 instances, 8–30 blocks): role offset 2 → 1/12, both offsets → 0/12, no roles → 1/12. No local-suite
policy generalises; the training set has to include larger instances first.

Full 95, local, `--max-memory 14000`, 90 min limit (weak signal where a run ends in `bad_alloc`; only coverage and
`Id out of range` count, see the protocol note):

| arm | solved | wall | cost | ended by |
|---|---|---|---|---|
| baseline datalog-sig (16 GB) | 19/95 | 410 s | 8 | `bad_alloc` ×2 (complexity 6 climb, then a 163-state complexity-3 round) |
| role offset 2 + concept offset 1 | **31/95** | ~900 s | – | `bad_alloc` ×2 at complexity 5 on 6–7 training problems (≤5 blocks) |
| no roles | 25/95 | 498 s | – | `bad_alloc` at complexity 4 on 9 problems, then instantly again at complexity 3 on 10 problems (29 concepts) |

## Cluster batch 1 (submitted 2026-09-14 23:20, `rleap_cpu`, 1 thread, seed 0, 128 GB, 12 h)

| job | tag | branch | config |
|---|---|---|---|
| 4133176 | c-base | learn-from-examples @ 238ae61 | datalog-sig |
| 4133177 | c-r2c1 | hyp/role-caps | role offset 2, concept offset 1 |
| 4133178 | c-noroles | hyp/role-caps | `include_roles: false` |
| 4133179 | c-noclimb | hyp/no-cost-climb | `add_problem_after_success: true` |
| 4133180 | c-distsets | hyp/dist-sets | precomputed distinguishing sets |

All on the full 95-problem blocks3ops suite. Results: pending.

## H2: add the next problem right after a success (`hyp/no-cost-climb`)

Mechanism: after a success whose policy fails some remaining problem, the loop used to climb complexity on the
same training set to beat the previous cost before adding the failing problem. `add_problem_after_success: true`
adds it immediately. Local blocks3ops suite (sonnet sanity run, 4 GB cap): baseline 9/10 in 148 s (climb hits
the cap), switch on **10/10 in 43 s** at complexity 2 throughout; gripper unchanged 5/5.

## H3: precomputed distinguishing sets (`hyp/dist-sets`)

Mechanism: the datalog-sig separation layer grounded |K|² pairs × every feature/concept/role. The set of elements
distinguishing a pair is instance data, so it is computed in Python (packed bit vectors) and emitted as
`sig_pair/3` + deduplicated `dist/2` facts with a hitting-set constraint; the quadratic `#show` rules are gone and
rule conditions are reconstructed in Python. Same semantics (verified on the same model).

One-shot blocks3ops at complexity 4, 4 GB cap: p004-1 atoms 1.82 M → 0.12 M, clingo CPU 15.0 s → 0.85 s;
p005-1 old `bad_alloc`, new solved in 14.6 s / 377 MB. Remaining term: Σ|D| (55 M dist facts at p006-1) →
follow-up H5 (lazy pair constraints).

## H4: combination dist-sets + no-cost-climb (`hyp/combo`)

Local suites, 8 GB cap, `--add-problem-after-success`:

| suite | baseline | combo |
|---|---|---|
| blocks3ops-local | 10/10, 175 s, cost 9, 5.9 GB | **10/10, 8 s, cost 5, 73 MB** |
| gripper-local | 5/5, 4 s, cost 6 | 5/5, 3 s, cost 6 |

Cluster: job 4133189 (tag c-combo), same protocol as batch 1. Results: pending.

Combo, full 95 **locally** (16 GB cap, 2 h limit): **29/95** in 1641 s, cost 21, training set of 21 problems
(2–7 blocks, 1781 states) still at complexity 3; ended by a Python `MemoryError` building the complexity-4
instance (88 concepts / 73 roles × 21 problems) and a failed complexity-5 round. Baseline died with 8 training
problems. Weak signal on the memory end, strong on coverage: every ≤7-block instance except blocks-007-4 is
solved. Note the frontier loop: blocks-007-1 accumulated 19 example plans.

**Cluster batch 1 after ~35 min** (peek at the last round of each log): the four arms without the
add-problem switch are all inside the post-success climb on 5–8 training problems of ≤5 blocks: base at
complexity 6 (858 concepts / 617 roles), dist-sets at complexity 8 (10 131 concepts / 2 090 roles),
no-roles at 8 (9 439 concepts), role-caps at 8 (2 856 / 615). The two arms with the switch are past that:
no-cost-climb has 14 training problems (≤6 blocks) at complexity 3, combo has 19 (≤7 blocks) at complexity 4.

Held-out (8–30 blocks) for the combo local full-95 policy (cost 21, trained on ≤7 blocks): **0/12**, all
"no action found". Training up to 7 blocks does not transfer either; the learned rule sets only cover the
signatures they were trained on. Generalisation is a separate question from coverage and needs its own
hypotheses (policy language bias, or training on far larger instances) once coverage stops being the wall.

## H5: lazy (counterexample-guided) pair constraints (`hyp/lazy-pairs`, on top of H3)

Mechanism: no `sig_pair`/`dist` facts up front. Solve the relaxation to optimality, read selection and
good/bad labelling, find violated pairs in Python (masked equality of packed bit vectors, bucketed per action
name, so pairs are never enumerated), ground one batch (default 5000, smallest |D| first) into the same
clingo `Control` as `#program pairs(k)`, re-solve. Zero violations ⇒ optimal for the full problem.

One-shot blocks3ops, complexity 4, 5 GB cap:

| instance | eager (H3) | lazy |
|---|---|---|
| p004-1 | 2,128 pairs, 2.2 s, 108 MB, cost 2 | 789 pairs, 6 iterations, 2.3 s, 87 MB, cost 2 |
| p005-1 | 17,083 pairs, 11.3 s, 380 MB, cost 2 | 5,000 pairs, 2 iterations, 11.4 s, 201 MB, cost 2 |
| p006-1 | `MemoryError` building 55 M dist facts | **17,056 of 1,150,646 pairs, 7 iterations, 243 s, 605 MB, cost 4** |
| p007-1 | – | `bad_alloc` in state-space expansion / first grounding, before any lazy iteration |

The complexity-4 ceiling moves from 5 to 6 blocks; at 7 blocks the wall is now the state space and the fact
layer (`c_eval`/`r_eval`, n·|C|·m and n·|R|·m²), not the separation layer. Regression suites identical.

Merged into `hyp/combo` (now H2 + H3 + memory-release + H5): local blocks3ops suite 10/10 in 15 s, gripper 5/5.
Cluster: job 4133419 (tag c3-combo).

Cluster batch 2 (jobs 4133410–4133412, `hyp/combo` at 162c640 with the memory fix): c2-combo (reference),
c2-unselect (`unselect_problems: true`), c2-frontier2 (`max_frontier_states_per_round: 2`).

## Cluster results

| job | arm | solved | wall | peak RSS | train set | ended by |
|---|---|---|---|---|---|---|
| 4133189 | c-combo (H2+H3, no memory fix, eager pairs) | **28/95** | 1 h 26 min | 106 GB | 19 problems (≤7 blocks), complexity 3, cost 21 | 2× memory error at complexity 4; no `Id out of range` |
| 4133176 | c-base (datalog-sig) | **33/95** | 1 h 31 min | 115 GB | 9 problems (≤5 blocks), complexity 3, cost 10 | 3× `bad_alloc` (complexity 4, then a 219-state complexity-3 round right after); no `Id out of range` |
| 4133410 | c2-combo (H2+H3+memory fix, eager pairs) | 28/95 | 1 h 17 min | 107 GB | 19 problems (≤7 blocks), cost 21 | complexities 3–5 refuted for the 19-problem set; `bad_alloc` at complexity 6 (858 concepts / 617 roles) |
| 4133411 | c2-unselect (c2-combo + `unselect_problems`) | 27/95 | 2 h 04 min | 102 GB | 11 problems (≤7 blocks), cost 19 | complexity 6 refuted too; memory error at complexity 7 |
| 4133177 | c-r2c1 (H1 alone: role offset 2, concept offset 1) | 24/95 | 3 h 06 min | 116 GB | 7 problems (≤6 blocks), cost 12 | 3× `bad_alloc` in the climb (complexity 6–7) |
| 4133178 | c-noroles (H1 alone: no roles) | 22/95 | 7 h 38 min | 115 GB | 12 problems (≤6 blocks), cost 14 | 6× `bad_alloc`, incl. a 29-concept complexity-3 round right after a failed one (memory retention, no fix on this branch) |
| 4133179 | c-noclimb (H2 alone, eager sig encoding) | 25/95 (last policy) | 12 h TIMEOUT | – | 18 problems (≤7 blocks) | stuck 4 h in a complexity-3 round on 18 problems (eager separation layer) |
| 4133180 | c-distsets (H3 alone) | – (no policy test in the last 200 MB) | 12 h TIMEOUT | – | 15 problems (≤6 blocks) | entire budget in the post-success climb at complexity 6 (858 / 617); 30 GB log |
| 4133412 | c2-frontier2 (c2-combo + `max_frontier_states_per_round: 2`) | 29/95 (last policy) | 12 h TIMEOUT | – | 20 problems (≤7 blocks) | 11 h in one complexity-4 round: first model after 19 min, optimality never proven (eager pairs, bb) |
| 4133419 | c3-combo (H2+H3+H5 lazy pairs+memory fix) | 29/95 (last policy) | 12 h TIMEOUT | – | 22 problems (≤7 blocks) | lazy iterations 3 and 4 took 2291 s and 6225 s (cost 26 → 28), iteration 5 never returned |
| 4133421 | c3-r2c1 (c3-combo + role offset 2, concept offset 1) | **33/95** (last policy) | 12 h TIMEOUT | – | 27 problems incl. 8-block (08-1, 08-3, 08-4, 08-5) | 11 h in lazy iteration 2 of a round |

The eager distinguishing-set instance for 19 problems at complexity 4 exhausts 128 GB. This is the case the
lazy pairs arm (c3-combo, job 4133419) targets.

Caution on reading these two rows: the base arm's cheaper policy (cost 10) generalised further than the combo's
cost-21 policy learned from twice the training set, so coverage is not monotone in training-set size. Policy
cost matters for generalisation, and the add-problem switch trades the climb's cost minimisation for training
growth. The remaining arms decide whether a bounded climb or a final minimisation pass is needed.

## H7: restricted feature grammar (config only, on `hyp/combo`)

The complexity-6 pool (858 concepts / 617 roles) is what every combo arm dies on, and a blocks policy needs a
handful of constructors. DLPlan's defaults enable and/all/some/not/equal/one_of/bot/top concepts and
primitive/inverse/transitive-closure/restrict/identity/and roles (plus `til_c` from the datalog config).
Arms keep primitive, not, some, all, and concepts and primitive, inverse, transitive-closure roles:

| job | tag | config |
|---|---|---|
| 4133890 | c3-grammar | grammar restriction + add-problem switch |
| 4133891 | c3-grammar-r2c1 | same + role offset 2, concept offset 1 |

Also pending: 4133419 c3-combo (lazy pairs), 4133421 c3-r2c1 (lazy pairs + role caps), 4133412 c2-frontier2,
4133177–4133180 (batch 1 single hypotheses).

**Cluster, 07:00 on 2026-09-15 (batch 1 at 7.5 h, batch 3 at 6.5 h):** the lazy-pairs arms are the first to put
8-block instances into the training set (c3-r2c1: 27 training problems, c3-grammar-r2c1: 26), and the grammar
arm has 20. But every combo arm has been inside one round for 5–6 h. The log tails show where: the first relaxed
lazy solve finishes in 0.2–100 s, then the second solve, after grounding the first batch of 5000 pair constraints
(400 k `dist` facts on c3-combo), never returns. The optimisation (minimise feature cost subject to hitting-set
constraints) is now the wall, not grounding and not memory. Exploration copies with 32 clingo threads were
submitted as x32-combo (4135666) and x32-r2c1 (4135667); they are not comparable to the seeded 1-thread arms.

## H6: emit only the facts the sig program reads (`hyp/lean-facts`, on `hyp/combo`)

`solve_datalog_sig.lp` never references `c_eval/4`, `r_eval/5`, `aname/2`, `aparam/3` (the Python side owns
the memberships since H3), yet the instance emitted all of them: n·|C|·m and n·|R|·m² ground atoms for nothing.
New key `emit_object_facts` (true by default, false in `default_datalog-sig.yaml`) gates the text emission only.
One-shot p005-1 / p006-1: identical cost and outcome; 1.3–1.8× fewer atoms, 9–13× smaller instance text,
7–32 % less peak memory, growing with instance size. p007-1 at complexity 4 (6 GB cap, 40 min) now passes
expansion (18 min), feature generation (9 min), grounding and one lazy iteration before the time limit; it died
in `bad_alloc` before the first grounding without this. Regression suites identical. Merged into `hyp/combo`.

## H8: core-guided / anytime optimisation (`hyp/anytime-solve`, on `hyp/combo`)

Knobs: `clingo_opt_strategy` (`bb` default, `usc`), `clingo_options`, `solve_time_limit` (seconds). With a
limit, the best model found is kept; a not-proven-optimal success keeps `max_cost = cost − 1` but refutes no
complexity level, and a cut-off solve without a model is a new `Result.TIMEOUT` (escalates like NO_SOLUTION).
The lazy loop still terminates only on zero violated pairs, so every returned policy is feasible.

| case | bb | usc | bb + 60 s limit |
|---|---|---|---|
| five local suites | all identical, every solve ≤ 0.2 s | same (one equal-cost model swap on blocks3ops-local) | same |
| p006-1 one-shot c=4 | 10 iterations, 142.6 s clingo, 210.7 s wall, cost 4 | 9 iterations, **11.2 s clingo, 79.3 s wall**, cost 4 | 7 iterations, 115.3 s, cost 4 |
| p005-1 + p005-2 + p006-1, c=4, 25 min | 4 solves, then hangs 21 min (80 k violated pairs left) | 2 solves, then 24 min with **no model** (1.41 M left) | **24 iterations**, 22 cut off, 277 violated pairs left, models cost 16–48 |

usc is 12.7× faster on the solving part when it works and useless when the core extraction does not converge;
the time budget keeps the loop progressing at the price of policy cost. Merged into `hyp/combo` (6f2d2cf).

Cluster batch 4 on that commit (`c4-*`, jobs 4136480–4136483): usc, time limit 300 s, and both with the role caps.

## H9: hand-crafted features (diagnostic, `preset_features` on `hyp/combo`)

Question: can the learner produce a general blocks3ops policy at all if the features are right? Preset
(no synthesis, `min_complexity = max_complexity = 1`): booleans `b_empty(¬well)`; concepts clear, ontable,
ontable_G, clear_G, `match = c_equal(on, on_G)` (a block sits on its goal support), `well = c_all(tr-refl-closure(on), match)`,
`¬well`, `c_some(on_G, well ∧ clear)` (goal support ready), `c_some(tr-closure(on), ¬well)` (above a misplaced block);
roles on, on_G. Config: `claude-experiments/preset-blocks.yaml` on the cluster worktree combo5.

| run | result |
|---|---|
| blocks3ops-local (10 problems, desk-03, 4 threads) | **10/10 in 4 s**, 13 rules, cost 6, 4 lazy iterations; the learner used `match`, ontable, ontable_G and the two roles, never `well` |
| held-out 8–30 blocks, that policy | **5/12** (008-1, 010-1, 010-2, 020-1, 030-1); failures are "no action found" |
| full 95 | job 4138079 (c5-preset), pending |

Every synthesised-feature policy so far scored 0/12 on the held-out set. So the pipeline can learn general
policies when the pool is small and relevant; the problem is the synthesised pool combined with cost-minimal
selection, which yields rule sets that fit the training signatures rather than the strategy. The features the
policy actually used have complexity ≤ 3 and are all generated by DLPlan's default grammar, so it is a selection
problem, not an expressivity problem.

**Why the synthesised pool cannot express the preset policy.** One-shot dumps on the workstation
(`domains/deterministic/blocks3ops/p003-1 + p004-1`): the complexity-3 pool has 29 concepts and the
complexity-4 pool 88, and neither contains a single `c_equal(...)` concept, although `generate_equal_concept`
is on and `c_equal(on, on_G)` has complexity 3. DLPlan deduplicates generated elements by their denotation on
the sample states, and on small plan-restricted state sets the general concept coincides with a shallower one
(the pool keeps e.g. `c_some(on_G, ontable)`, `c_all(on, ontable_G)`), so the general representative is
discarded before the solver ever sees it. genfond's own `prune_redundant_*` does the same over the instance
states. This is the mechanism behind 0/12 held-out: the learner can only combine accidental coincidences of
the training states. Hypothesis H10: synthesise (and deduplicate) features over a richer state sample, e.g.
random-walk states from all problems including the large unsolved ones, while the ASP instance keeps only the
training states.
