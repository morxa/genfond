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
| 4133890 | c3-grammar (H7 grammar restriction) | 28/95 (last policy) | 12 h TIMEOUT | – | 21 problems (≤7 blocks) | still progressing at the end (lazy iteration 2 in 241 s); no gain over c3-combo |
| 4133891 | c3-grammar-r2c1 (H7 + role caps) | 33/95 (last policy) | 12 h TIMEOUT | – | 26 problems incl. 8-block | 11 h in one lazy iteration; same coverage as c3-r2c1 |
| 4135666 | x32-combo (c3-combo, 32 threads, not comparable) | 31/95 (last policy) | 12 h TIMEOUT | – | 23 incl. 8-block | lazy iteration 3 took 62 min, iteration 4 never returned |
| 4135667 | x32-r2c1 (c3-r2c1, 32 threads) | 33/95 (last policy) | 12 h TIMEOUT | – | 26 incl. 8-block | iteration 3 took 3 h; threads do not remove the stall |
| 4136480 | c4-usc (combo + usc) | 27/95 (last policy) | 12 h TIMEOUT | – | 17 (≤7 blocks) | stuck from 09:02 in lazy iteration 2 |
| 4136481 | c4-tl300 (combo + 300 s solve budget) | 34/95 (last policy) | 12 h TIMEOUT | – | 27 incl. 8-block | still progressing at the end (last round 17:40) |
| 4136482 | c4-usc-r2c1 (usc + role caps) | 33/95 (last policy) | 12 h TIMEOUT | – | 24 incl. 8-block | stuck from 08:54 |
| 4136483 | c4-tl300-r2c1 (300 s budget + role caps) | **40/95** (last policy) | 12 h TIMEOUT | – | 29+ incl. 8-block | still progressing at the end (last round 20:00) |

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

**Correction to the paragraph above (H10 result, `hyp/rich-sample`):** `c_equal(on, on_G)` is not lost to
deduplication. Its denotation matches no generated concept on 86 training states or on 9 489 training+sample
states, and DLPlan's generator emits no `c_equal` at all in this build, at any complexity, even on a synthetic
two-role vocabulary with `generate_equal_concept=True`. The richer sample changes nothing on blocks3ops
(pool sizes identical at complexity 3 and 4; held-out 0–1/12 either way) and stays unmerged. The missing
constructor is a DLPlan issue (or needs a genfond-side goal-comparison augmentation).

**Protocol correction:** `State` is a `frozenset` of pddl atoms, so iteration order and the frontier loop's
choices follow `PYTHONHASHSEED`; `--seed 0 -n 1` alone does not make two runs identical. Both benchmark scripts
now pin `PYTHONHASHSEED`. Earlier single-run A/Bs carry that noise.

## H11: DLPlan never generates goal-comparison concepts for genfond (bug)

`dlplan/src/generator/rules/concepts/equal.cpp` (pinned rev cfd4561) generates `c_equal(R, R_goal)` only when
the goal role's predicate name is the other's plus the **lowercase** suffix `_g`. genfond names goal predicates
`on_G`, `clear_G`, … (uppercase), so the rule never matches and no `c_equal` concept, hence no "block sits on
its goal support", ever enters a synthesised pool, at any complexity. The hand-written preset (H9) bypassed
the generator, which is why it generalised. Decision (Till): genfond adopts DLPlan's convention and names goal
predicates `_g` (`hyp/goal-suffix`); DLPlan and the apptainer image stay as they are. Old `.policy` pickles
become incompatible. Validation on the local suite and the held-out set is running.

H11 result (`hyp/goal-suffix`, 17ed4c2): with `_g` the pools gain `c_equal` (29→30 / 88→90 concepts at
complexity 3 / 4). blocks3ops-local 10/10, cost 7, **36 rules** using `c_equal` heavily, held-out **0/12**;
regression suites identical. So availability of the right concept is necessary but not sufficient: among thirty
concepts the cost-minimal selection still yields a training-fitted rule set, where the tiny preset pool forced
a 13-rule policy (cost 6, 5/12). Next: an Occam bias on the number of good signatures (H12, `hyp/min-rules`).
Gotcha found: `scripts/eval_policy.py` run as a script imports the editable-installed genfond from the main
checkout unless `PYTHONPATH=.` is set.

**Is the add-problem switch now detrimental?** (Till's hypothesis: the post-success climb was the cost
minimisation across complexity levels, and cheaper policies generalised better.) A/B on `hyp/goal-suffix`,
workstation, blocks3ops-local, 4 threads: climb on → 10/10, 26 rules, 18 rounds, 19 s, held-out 0/12;
switch on → 10/10, 26 rules, 14 rounds, 14 s, held-out 0/12. No difference at this scale. Full-95 arms
gs-climb (4139572) and gs-noclimb (4139573) test it at scale.

## H12: minimise the number of good signatures (`hyp/min-rules`, on `hyp/goal-suffix`)

`minimize_good_signatures: none|below|above` adds `#minimize { 1@P, K : good_sig(K) }` under or over the
complexity level (frontier stays highest); `cost_utils.feature_cost` now finds the complexity level for every
layout (this also fixed `max_cost` being read from the wrong level once extra levels exist). Also repaired the
gripper encoding-equivalence test to compare cost, since `c_equal` created a tie.

blocks3ops-local (10 problems), local, seed 0: none → 36 rules / 72 good signatures; below → 26 / 30;
above → 20 / 23. Held-out **0/12 in all three**. Regression suites: `below` identical to baseline everywhere,
`above` +1 cost on blocks4ops-clear. Fewer rules alone does not give generalisation. The preset policy had
5 selected elements; the synthesised ones select more, cheaper elements → H13 (`hyp/min-count`): minimise
the number of selected elements.

**Cluster snapshot 13:06 (batch 4 at 4 h 20 min, x32 at 6 h, preset at 1 h 40 min):**

| arm | training set | last policy solved | state |
|---|---|---|---|
| c4-usc | 17 (≤7 blocks) | 27 | stuck 4 h in lazy iteration 2 (usc) |
| c4-tl300 | 26 incl. 8-block | 32 | progressing, iterations at the 300 s budget, not proven optimal |
| c4-usc-r2c1 | 24 incl. 8-block | 33 | iteration 2 took 47 min, stuck since 08:54 |
| c4-tl300-r2c1 | 29 incl. 8-block | **38** | progressing, 18 budgeted iterations per round |
| x32-combo | 23 incl. 8-block | 31 | stuck 3.5 h in iteration 2 with 32 threads |
| x32-r2c1 | 26 incl. 8-block | 33 | iteration 3 took 3 h |
| c5-preset | 18 | – | frontier expansion runaway on 08-2 (1 423 plans, 721 MB log); preset pool unsatisfiable for the set |

The per-solve time budget is the only setting that keeps rounds moving at this scale; core-guided search and
32 threads do not. The gs-climb / gs-noclimb arms were submitted without the budget and will stall the same way,
so they are resubmitted with `solve_time_limit: 300` as gs-climb-tl / gs-noclimb-tl.

## H13: minimise the number of selected elements (`hyp/min-count`, on `hyp/min-rules`)

`minimize_selected_count: none|below|above` (count of selected concepts+features+roles at its own priority
level; frontier moved to `@3`; `cost_utils.feature_cost` derives the complexity level from both knobs).
blocks3ops-local: `above` → 27 rules, 5 selected elements (drops `c_equal`), held-out **0/12** again;
regression suites unchanged. Composing `count=above` with `sigs=above` (two stacked levels above the
complexity level) did not finish in 30 min on an idle 96-core node. Neither Occam bias reproduces the preset's
generalisation locally; both are cheap to test for coverage at scale.

Cluster batch 6 on `hyp/min-count` (goal suffix `_g`, all H2–H8 machinery), all with
`add_problem_after_success`, `solve_time_limit: 300`, role offset 2 / concept offset 1: gs-tl-r2c1 (no
bias), mc-sigs (good-signature bias above), mc-count (selected-count bias above).

## Faster cadence (from 2026-09-15 afternoon)

Till: too few ideas at a time. Changes: exploration arms get a 4 h limit (`SBATCH_TIMELIMIT=4:00:00`) on the
otherwise idle `rleap_cpu_modern` node (wall times there are not comparable to `rleap_cpu`, coverage is), and
four implementation agents run in parallel.

Batch 7 (`m-*`, jobs 4143934–4143941, `hyp/min-count`, all with add-problem, role offset 2 / concept offset 1,
`solve_time_limit: 300` unless stated): tl60, tl900 (budget sweep), siw-r1, siw-r10, siw-nobranch (planner
diversity), unrestricted (all DLPlan generators), frontier2 (≤2 frontier states per round), preset (hand-crafted
pool with `_g`, + frontier2).

In implementation (own branches from `hyp/min-count`): fixed-labels (precompute forced good/bad transitions so
pair constraints between forced classes carry no labelling choice; optional plan heuristic), plan-cap (cap and
state-set dedupe of example plans; the preset arm hit 1 423 plans on one problem), final-climb (one cost
minimisation pass at the end instead of after every success), wall-budget (graceful `max_wall_time`/SIGTERM
handling so killed runs still write stats and policy).

## H14: one final cost-minimisation pass (`hyp/final-climb`, on `hyp/min-count`)

`final_cost_minimization: true`: after the fast add-problem loop ends, run the old complexity climb once on the
frozen final training set and keep the candidate that solves the most problems, ties by lowest cost.
Workstation, `--type datalog-sig -n 1 --seed 0`, add-problem, `solve_time_limit: 300`, role offset 2 /
concept offset 1, goal suffix `_g`:

| suite | pass off | pass on |
|---|---|---|
| blocks3ops-local | 10/10, cost 9, 40 rules, 25 s, **held-out 4/12** | 10/10, cost 7, 27 rules, 49 s, **held-out 8/12** |
| gripper-local | 5/5, cost 7 | unchanged (a cheaper cost-6 candidate solved fewer problems and was rejected) |
| miconic-local | 4/4, cost 11 | 4/4, cost 8 |

Two things to note: this configuration (goal suffix + role caps + budget) already generalises to 4/12 without
the pass, the first synthesised-pool policy to transfer at all; and the cheaper policy transfers twice as far.
The pass triggered only when all problems were solved; the end-of-run trigger is being added before the
full-95 cluster arms.

The pass also runs when the loop ends without solving everything (exhausted iterator), verified by tests
(a8df8dc). Cluster arms with the full configuration (add-problem, budget 300 s, role caps, final pass):
fc-cpu (4144318, `rleap_cpu`, 12 h) and fc-mod (4144319, `rleap_cpu_modern`, 4 h).

## H15: data-driven grammar (Till's suggestion)

Constructor usage over the selected elements of the final rules of 115 blocks3ops run logs (26 202 rules,
occurrence counts): c_primitive 370 k, r_primitive 311 k, c_not 184 k, c_some 137 k, c_all 120 k, b_empty 53 k,
c_and 26 k, c_equal 21 k, r_inverse 13 k; rare: r_not 5 k, r_transitive_closure 4 k, r_til_c 1.6 k,
r_identity 121; never: c_or, c_diff, c_one_of, c_projection, c_subset, r_and, r_restrict, r_compose,
nullary/inclusion booleans. Caveat: circular (usage reflects what was offered and what cost minimisation picks).
Mined grammar = default minus one_of/bot/top concepts, identity/restrict/and/til_c roles, nullary/inclusion
booleans. Arms on `rleap_cpu_modern`, 4 h: m-mined (`hyp/min-count` config) and m-mined-fc (+ final pass).

## H16: forced transition labels (`hyp/fixed-labels`, on `hyp/min-count`)

Before search, a fixpoint derives the labels every model agrees on (an action whose outcome is neither alive
nor pruned is bad; a bad class is bad at every occurrence; a state whose remaining candidates share one class
forces it good; a good class is good everywhere) and emits `forced_good/3`, `forced_bad/3`; forced-bad actions
leave the choice, forced-good become constraints, forced×forced pairs seed the lazy loop. Sound by
construction (tests refute the opposite of each forced label on the unmodified program). Stall case
(p005-1+p005-2+p006-1, c=4, 30 min, no budget): 2 lazy iterations / 1.68 M violated pairs left → 6 / 3 179.
With the 300 s budget both arms are level (8 vs 9 iterations). Largest finishing one-shot: 5:31 → 2:50 at the
same optimum. Coverage and cost identical everywhere. Forced-*bad* never fires: frontier expansion makes every
off-plan successor `pruned`, which counts as safe, so only the forced-good rule acts. The optional plan
heuristic cancels the gain and is left off. Merged into `hyp/combo` (6ac9f3f, 183 tests).

Cluster: fc-wall arms (4144770 modern 4 h, 4144771 cpu 12 h: final pass + graceful wall budget, from the
combo commit before forced labels) and full2 arms (modern 4 h, cpu 12 h: the same plus `fix_forced_labels`).

**Batch 7 results (4 h on `rleap_cpu_modern`, `hyp/min-count`, add-problem + role caps, budget 300 s unless
stated; coverage = last policy tested, from the log tail):**

| arm | training set at 4 h | last policy solved |
|---|---|---|
| tl60 (budget 60 s) | 28, ≤8 blocks | – (no policy test found in the 441 MB log tail) |
| tl900 (budget 900 s) | 28, ≤8 blocks | 33 |
| **siw-r1 (SIW restarts 1)** | 11 incl. **10-block** | **43** |
| siw-r10 (restarts 10) | 28, ≤8 blocks | 33 |
| siw-nobranch | 23 incl. 10-block | 29 |
| unrestricted generators | 26, ≤8 blocks | – (as tl60) |
| frontier2 | 29, ≤8 blocks | 33 |
| preset (hand-crafted pool) | 10 incl. 10-block | – (pool unsatisfiable at the end; 1.4 GB log) |

Fewer example plans per problem (`restarts: 1`) is the largest single gain so far: 43/95 in 4 h against
33–34 for the 300 s-budget arms at 12 h, and the training set reaches 10-block instances with only 11
problems. Smaller plan sets mean smaller state spaces and fewer pairs per round. Submitted the full
configuration with `restarts: 1` as full2-r1-mod (4148282, 4 h) and full2-r1-cpu (4148283, 12 h).

Follow-up arms on the plan-count finding (full configuration, `rleap_cpu_modern`, 4 h, graceful wall budget):
m2-plans1-r1 (4148300: `min_number_of_plans: 1`, restarts 1), m2-plans2-r1 (4148301), m2-r1-nobranch
(4148302: restarts 1 and `branch: false`, i.e. a single deterministic plan per problem).

fc-mod (4144319, 4 h, `hyp/final-climb` config without graceful budget): 30/95 (last policy), 24 training
problems ≤8 blocks; killed by the limit before the final pass could run. The `full*` arms carry the graceful
wall budget so their final pass executes.

full-mod (4144770, 4 h, combo before forced labels, graceful budget 13 200 s): first arm to end on its own —
`stoppedBy=wall_time`, stats row and policy written, final pass ran (1 round, cost 37 → 37; the 300 s reserve
leaves no room for it). 30/95, 24 training problems incl. 10-block. Future arms set `wall_time_reserve: 1800`
so the pass has half an hour.

H15 result (4 h, modern node): m-mined **36/95** (31 training problems, ≤9 blocks; pool 89 features /
221 concepts / 8 roles at the last round) against 33 for tl900/frontier2 with the default grammar in the same
4 h; m-mined-fc 33/95 (27 problems, killed before the pass; no graceful budget on that branch). A modest but
consistent gain; combined with `restarts: 1` next (m4-mined-r1).

H9 at scale (c5-preset, 4138079, 12 h): the hand-crafted pool became unsatisfiable once the training set
reached 18–20 problems (≤8 blocks) and stayed so for the last 59 rounds; frontier expansion meanwhile
attached **4 880** example plans to blocks-008-2 (7.8 GB log). No policy test in the tail. Conclusion: the
preset shows generalisation is attainable on small sets but is not a complete policy language for blocks3ops,
and unbounded frontier plan growth is a real defect (addressed by `hyp/plan-cap`).

## H17: cap and dedupe example plans (`hyp/plan-cap`, on `hyp/min-count`)

`max_plans_per_problem` (null = unbounded) stops INC_PLANS/frontier expansion adding plans beyond the cap;
plans whose replayed state set adds nothing to a problem's covered states are dropped (catches what the
action-sequence key misses). blocks3ops-local with cap 8: largest problem 45 → 8 plans, 190 → 45 states,
grounded atoms 97 365 → 24 346, wall 28.7 s → 13.7 s, identical policy cost; gripper/miconic byte-identical.
The unbounded-baseline driver on the workstation failed on a bind-path error, so no direct A/B there; the
cluster arms provide it. Merged into `hyp/combo` (189 tests). Arms: m5-r1-cap8 (modern, 4 h) and c-r1-cap8
(cpu, 12 h) = full configuration + restarts 1 + reserve 1800 + cap 8.

Also on the modern node (4 h, full configuration + restarts 1 + cap 8 + reserve 1800): m6-s1 and m6-s2
(seeds 1 and 2 incl. `PYTHONHASHSEED`, to measure run-to-run variance of the best configuration) and
m6-minc3 (`min_complexity: 3`, skipping the complexity-2 rounds). In implementation: `hyp/min-train-size`
(`min_train_objects`: never train on instances below a size threshold; small instances may induce the
overfitted rule sets).

full2-mod (4144791, modern, 4 h, full configuration incl. forced labels): queued until ~20:55, then 68 rounds,
last policy **33/95**; killed by SLURM rather than stopping gracefully — the final pass started within the
budget but its climb rounds (118 s, 392 s solves) ran past the 300 s reserve, so no stats row or policy was
written. Fix in progress (`hyp/final-pass-deadline`: the pass gets its own budget and respects the deadline).

m2 results (modern, 4 h, full configuration + restarts 1, graceful stop worked, stats written):
m2-plans1-r1 (`min_number_of_plans: 1`) **35/95**, 27 training problems ≤9 blocks, final pass 41 → 41;
m2-r1-nobranch (`branch: false`) **36/95**, 28 problems ≤9 blocks, pass 39 → 39. Both below the plain
siw-r1 arm (43) on the earlier branch; the m6 seed arms will show how much of that is variance.

gs-climb / gs-noclimb (4139572/3, `hyp/goal-suffix`, no solve budget, 12 h): climb 16/95 (15 training
problems ≤6 blocks, stuck from 16:09), switch 28/95 (22 problems ≤7 blocks, stuck from 19:55). Without a
budget the climb loses badly at scale; the budgeted pair (gs-climb-tl / gs-noclimb-tl) is the fair test.

full2-r1-mod (4148282, modern, 4 h): last policy 33/95, 63 rounds, ≤8 blocks; killed in the final pass (same
deadline bug as full2-mod, no stats row).
m2-plans2-r1 (4148301): last policy 33/95, 68 rounds, ≤8 blocks; also killed in the final pass.

gs-climb-tl / gs-noclimb-tl (4139742/3, goal suffix + 300 s budget, 12 h): climb **24/95** (17 problems
≤6 blocks), add-problem switch **31/95** (25 problems ≤8 blocks). Verdict on Till's hypothesis: at scale the
switch is right; the cost minimisation the climb provided is better spent once at the end (H14 final pass),
and the base arm's 33 with a cost-10 policy looks like a lucky model rather than a systematic effect.

Final-pass deadline fix (`hyp/final-pass-deadline`, 6f43768, merged into `hyp/combo`): the pass gets its own
slice (`final_pass_budget: 1800`, `final_pass_min_time: 600`), respects the deadline between rounds, and the
main loop stops earlier by that budget. Resubmitted the best configuration as m7-r1-cap8 (4152032, modern,
4 h). One pre-existing flaky solver test (zero-second limit race) is being made deterministic on
`hyp/fix-flaky-limit-test`.
m3-r1-res (4149378, modern, 4 h, reserve 1800 but pre-fix pass): last policy 33/95, 65 rounds, ≤8 blocks; killed.
Note the cluster of full-configuration arms at 33–36 in 4 h against 43 for the plain siw-r1 arm on
`hyp/min-count`; the seed arms (m6-s1/s2) decide whether that 43 was a lucky trajectory.
m4-mined-r1 (4149500, modern, 4 h): last policy 33/95, 71 rounds, ≤8 blocks; killed in the pass with a
1 945-feature / 5 738-concept pool (the pass climbs into the huge pools the loop otherwise avoids).

Final-pass bound (`hyp/final-pass-bound`, b23c6b8, merged into `hyp/combo`, 200 tests): the pass climbs at
most `final_pass_max_levels: 2` levels above the success complexity and can skip rounds whose pool exceeds
`final_pass_max_pool`. Submitted m8-r1-cap8 (4153234, modern, 4 h) with the full configuration on that commit.

Batch 6 (12 h, `hyp/min-count`, goal suffix + budget + role caps): b6-gs-tl-r2c1 33/95 (29 problems),
b6-mc-sigs 33/95 (27), b6-mc-count 33/95 (28). Neither Occam bias changes coverage at scale. The recurring
33 = all instances up to 7 blocks (30) plus three 8-block ones; the 8–9-block instances are the coverage wall,
crossed so far only by the restarts-1 arm (43, with 10-block problems in training).

m6-s1 (4150566, seed 1 incl. hash seed, full configuration + restarts 1 + cap 8, modern 4 h, graceful):
**29/95**, 20 training problems ≤8 blocks, pass 36 → 36. Against 33–36 for seed 0 the run-to-run variance is
about ±4 problems; arm differences of that size are noise.

m5-r1-cap8 (4150538, seed 0, full configuration + restarts 1 + `max_plans_per_problem: 8`, modern 4 h,
graceful): **29/95**, 20 training problems ≤8 blocks, 41 rounds, pass 39 → 39. The cap is within the ±4 noise
band of the uncapped arms (33–36) and does not add coverage; the dedupe stays (it only removes plans that add
no states), the cap stays off by default.
Repeatability check of the 43: m9-siwr1-s1 / m9-siwr1-s2 (4153962/3, `hyp/min-count` siw-r1 config, seeds 1
and 2, modern 4 h).
m6-s2 (seed 2) 33/95 and m6-minc3 (`min_complexity: 3`) 33/95, both ≤8 blocks, both killed in the pass
(pre-fix code). Best configuration across seeds 0/1/2: 33–36 / 29 / 33.

## H18: the loop discards near-general policies (from the `hyp/min-train-size` validation)

`min_train_objects` itself was a null result (the 5- and 7-block thresholds never converged in 40 min), but its
*baseline* arm exposed the real issue: on the 30-problem 2–7-block suite (full configuration + restarts 1 +
cap 8, seed 0) the loop learned a **7-rule, cost-8 policy from seven 2–4-object problems in 14 s** that solves
**11/12 held-out** instances (8–30 blocks) and, evaluated on the full suite, **91/95** (failing 010-5, 013-4,
017-5, 018-4). On the 95-problem runs the same loop keeps going because those instances fail: it adds the
failing instance, re-learns on the larger set, and the later policies are 26–40-rule patchworks solving
30–40. The run only reports the last policy, although every success is tested on all problems in the loop.
Fix in progress (`hyp/keep-best`): remember and return the best-coverage policy seen.

Correction to every "last policy solved" figure read from log tails: the in-loop policy test in
`solve_iteratively` breaks at the first failing problem (problems are sorted by size), so the count is the
position of the first failure, not coverage. That is why arms plateau at 33 / 36 / 43 (first failure among
the 8-, 9- or 10-block instances). Only the end-of-run "Policy solves N out of 95" (graceful runs) is true
coverage, and for those runs it matched the tail figure, so the cluster policies were genuinely weaker than the
7-rule policy. No cluster run tested a policy whose first failure came after the 43rd problem. `hyp/keep-best`
removes the early break and keeps the best-coverage policy. Running now on the workstation: the 30-problem
training run repeated for seeds 0–2, with and without `minimize_selected_count: above`, each policy scored on
all 95.
m7-r1-cap8 (4152032, modern 4 h, deadline fix): graceful, true coverage **29/95**, 20 training problems
≤8 blocks, pass 45 → 45 (1 round). Cost 45 versus cost 8 for the 91/95 policy from the small suite.

H18 fix (`hyp/keep-best`, 3d757ec, merged into `hyp/combo`, 204 tests): the in-loop test now scores every
problem (no early break when `keep_best_policy` is on), the best-coverage policy of the run is returned
(ties by cost), stats gain `bestSolved/bestCost/bestRound/lastSolved`. Arms kb-r1-mod (4154621, modern 4 h)
and kb-r1-cpu (4154622, cpu 12 h) with the full configuration + restarts 1.
Note: the 91/95 run used add-problem, budget 300 s, role caps, restarts 1 and cap 8 **without** the final
pass and forced labels. A repeat with those two added hung after the first lazy iteration of round 23 for
two hours (to be isolated). The seed repeat was restarted with the exact 91/95 configuration.
Arms kb-lite-mod (4154630, modern 4 h) and kb-lite-cpu (4154631, cpu 12 h): the exact 91/95 configuration
(add-problem, budget 300 s, role caps, restarts 1, cap 8; no final pass, no forced labels) with keep-best on
the full 95. Workstation: isolating whether `fix_forced_labels` or `final_cost_minimization` causes the hang.
Hang isolation (30-problem suite, seed 0): `fix_forced_labels` alone → 30/30, 7 rules, 14 s (13 rounds);
`final_cost_minimization` alone → 30/30, 8 rules, 29 s (15 rounds). Neither flag hangs by itself; the
combined run took a different trajectory (23 rounds) and stalled inside one solve. Re-running the combination
with seeds 1 and 0 to see whether it is trajectory-dependent.

Seed repeat of the 91/95 configuration on the 30-problem 2–7-block suite (workstation): seeds 0/1/2 give
30/30 in 14 s with 7/6/6 rules; with `minimize_selected_count: above` 7/7/7 rules, also 14 s. Seed 0's policy
re-scores **91/95** on the full suite (the chained scoring in the first pass had failed silently); the other
five are being scored.

Full-95 scores of the six small-suite policies (trained on 2–7 blocks, 14 s each): none seeds 0/1/2 →
**91 / 92 / 93**, count-bias seeds 0/1/2 → 91 / 90 / 91. Failures are "no action found" on a handful of
10–18-block instances (017-5 in every case, 013-4 and 010-5 often). The general policy is found robustly; the
remaining gap is a rule the ≤7-block training set never needs. Running: training suites extended to ≤8, ≤9,
≤10 blocks with the same configuration, scored on all 95.

**Why the 93/95 policy fails.** In blocks-010-5 the tower b0-b3-b2-b1 stands on b8, which is on the table
but belongs on b9; every block above b8 is on its goal support locally, so `c_equal(on, on_g)` marks them as
placed, no clear block is "misplaced", and no rule fires. The needed concept is *well-placed*: on the goal
support and so is everything below (`c_all(r_transitive_reflexive_closure(on), c_equal(on, on_g))`,
complexity 6). ≤7-block training instances never require it; on the full suite the rounds that would need
it sit at complexity 7 with thousands of concepts. H19 (`hyp/extra-features`): allow hand-given elements
into the synthesised pool regardless of the round's complexity limit, then train on the small suite plus the
two failing instances.
m8-r1-cap8 (4153234, modern 4 h, bounded pass): graceful, **28/95**, 19 training problems ≤8 blocks, pass
ran 2 levels (33 → 33). Same story as m7: the full-suite trajectory yields patchworks.
Suite extension (seed 2, same configuration): train ≤8 blocks (35 problems) → 35/35 in 16 s, 7 rules,
**92/95**; ≤9 blocks (40) → 40/40 in 20 s, 7 rules, **93/95** (fails 017-5, 018-4); ≤10 blocks (45, incl.
010-5) → 44/45 after 40 min timeout, 18 rounds, 92/95. Adding the instance that needs the well-placed concept
stalls the loop exactly as predicted; H19 (extra features) is the test.
fc-cpu (4144318, cpu 12 h, final pass, no graceful budget): killed; 29 training problems ≤9 blocks,
first failure at position 35 (a lower bound on coverage), still adding problems at the end.

## H19: hand-given elements alongside synthesis (`hyp/extra-features`, merged into `hyp/combo`, 207 tests)

`extra_features` (same shape as `preset_features`) appends parsed elements to the synthesised lists after
generation, so they are exempt from the per-round complexity limits, keep DLPlan's complexity, and go through
the usual pruning. Validation (seed 2, 2–7 blocks + 010-5 + 017-5, 40 min cap): with the well-placed concept
available the run drifted to a 63-rule policy, **26/95**; without it, 7 rules, **91/95**. The concept was
reachable and used in intermediate candidates but never survived cost minimisation: at complexity 6 it loses
to cheap combinations that fit the training set. H20 (`hyp/extra-cost`): `extra_features_complexity` overrides
the emitted cost of hand-given elements; validated with seeds 0–2 at override 2, plus 4 and unset.
full-cpu (4144771, cpu 12 h, combo before forced labels, graceful): **36/95**, 27 training problems ≤9 blocks,
pass 46 → 46. Cost 46 versus 8 for the small-suite policy.
full2-cpu (4144792, cpu 12 h, full configuration incl. forced labels, graceful): **37/95**, 28 training
problems ≤9 blocks, pass 51 → 51. Best 12-hour full-suite figure so far, still a cost-51 patchwork.

**Framing (Till):** preset and hand-given features are diagnostics of expressiveness, not solutions. H9 showed
the policy language can express a general blocks3ops policy; the synthesised 93/95 policy shows the grammar
can too. H19/H20 (extra elements, cost override) only test whether the learner would select the well-placed
concept if reachability and cost were not obstacles; they do not count as results. The synthesised path:
make the complexity-6 pool tractable (mined grammar) and break cost ties toward few general elements
(selected-count bias). Running on the workstation (seed 2, 2–7 blocks + 010-5 + 017-5, 45 min each):
mined-count, mined-only, count-only, with concept offset 0 so the round can reach complexity 6.

## Result: keep-best on the full suite

kb-r1-mod (4154621, modern 4 h, full configuration + restarts 1 + keep-best, graceful): round 14 produced a
**cost-8 policy solving 89/95**; the run's last policy solved 33, the pass changed nothing (40 → 40), and the
best policy was returned (`bestSolved=89 bestRound=14 solved=89`). First full-suite run with purely
synthesised features at this level; the earlier plateau at 33 was the loop discarding this policy.
That policy has 5 rules over three elements (`b_empty(clear ∧ clear_g)`, `c_equal(on, on_g)`, role `on_g`) and
solves **10/12 held-out** instances (fails 012-1, 020-2).
m9-siwr1-s1/s2 (4153962/3, seeds 1–2 of the restarts-1 arm, pre-keep-best): both 4 h TIMEOUT with first
failure at position 43, matching seed 0; the number was a first-failure position, superseded by keep-best.
Combined flags re-run (small suite): seed 1 → 36 rounds, 99-rule policy, 27/30, timed out; seed 0 → 30 rounds,
no policy, timed out. Either flag alone: 13–15 rounds, 7–8 rules, 30/30. The combination reliably derails the
small-suite trajectory (though the kb-r1-mod cluster arm with both on still found its 89/95 policy at round
14). Forced labels showed no gain with the solve budget, so the recommended configuration drops them:
add-problem, budget 300 s, role offset 2 / concept offset 1, restarts 1, cap 8, keep-best, final pass optional.

## H20 (diagnostic only): cost override for hand-given elements (`hyp/extra-cost`)

`extra_features_complexity` replaces the emitted cost of hand-given elements. 32-problem training set
(2–7 blocks + 010-5 + 017-5): override 2 → 3-rule policy on the well-placed concept, 93/95 (seeds 0, 2;
seed 1 lost its policy file to a double SIGTERM); override 4 → converges in **22 s, 3 rules, 95/95**; unset
(true complexity 6) → 94-rule patchwork, 26/95. Reading (per Till: presets are expressiveness tests, not
solutions): the language and the grammar contain a complete 3-rule blocks3ops policy; what synthesis lacks is
a way to make that complexity-6 concept reachable and to let it win against cheaper patchworks. The
synthesised-only experiment (mined grammar, selected-count bias, concept offset 0) is the test of that.
kb-lite-mod (4154630, modern 4 h, exact 93/95 configuration + keep-best, graceful): **87/95** from round 13
(cost 8), last policy 33. Keep-best turns every full-suite run into a high-80s result.
full2-r1-cpu (4148283, cpu 12 h, pre-deadline-fix): killed in the final pass, no stats; 34 training problems incl. 10-block, first failure at position 40.

Synthesised-only test (seed 2, 2–7 blocks + 010-5 + 017-5, concept offset 0, 45 min each): mined grammar +
count bias → 25/32, 94 rules; mined grammar only → 30/32, 141 rules; count bias only → 26/32, 88 rules; all
timed out, none used a closure concept. With purely synthesised features the loop cannot reach or select the
complexity-6 well-placed concept once the instances that need it are in training; the pool-shrinking and
tie-breaking levers do not change that. Open decision (Till): a goal-closure generator rule, the analogue of
DLPlan's goal-comparison rule, at complexity 4.

## Goal-closure generator rule: applicability across the 56 benchmark domains

Rule: for each primitive role R with goal counterpart R_g, generate `c_all(r_transitive_reflexive_closure(R),
c_equal(R, R_g))` (proposed complexity 4; syntactic 6). Trigger and semantics are domain-independent.

- **Fires with the intended meaning (10 domains):** the tower suites blocks3ops, blocks4ops, blocks,
  blocks-multiple, blocks3ops-fond, d2l/blocks, d2l/blocks3ops, plus hanoi (disc/peg chains) and d2l/depot
  (crate/pallet chains): "on my goal support and so is everything below me".
- **Generated but denotationally equal to `c_equal(R, R_g)`, hence deduplicated (≈10 domains):** cross-type
  goals — gripper, gripper-m, logistics, logistics98, delivery, grid, storage, barman, satellite, floortile.
- **Generated, likely idle (3):** single-atom same-type goals blocks-on, blocks4ops-fond-on, d2l/blocks-on.
- **No-op (≈25):** unary/nullary goals (miconic, visitall, spanner, sokoban, reward, childsnack, doors,
  tireworld variants, islands, miner, acrobatics, beam-walk, graph-traversal, blocks-clear variants); static
  same-type relations (adjacent, road, connected, next, above, smaller) have no goal version.
- **Outside the supported goal form:** d2l/blocks-tower (functional), d2l/gridworld (numeric).

Cost: at most one concept per goal role. Open decision: the assigned complexity.

## H21: `c_equal_closure` constructor in DLPlan (Till's decision)

Decision: implement the goal-closure rule in DLPlan as a new grammar constructor rather than a cost override.
`c_equal_closure(R1, R2)` = objects x such that every y reachable from x via the reflexive-transitive closure
of R1 has equal R1- and R2-successor sets (semantically `c_all(r_transitive_reflexive_closure(R1),
c_equal(R1, R2))`). Its complexity follows from the grammar (1 + the two roles = 3 for primitive roles), no
number is hardcoded. A generator rule applies it to goal pairs exactly as the existing `c_equal` rule does.
Branch `equal-closure` on the dlplan fork (from the pinned rev cfd4561); validation through genfond on the
small suite + the two failing instances, the pool dump, and the regression suites. Image rebuild and re-pin
follow if it holds up.
c-r1-cap8 (4150539, cpu 12 h, pre-deadline-fix): killed in the final pass, no stats row; superseded by the
keep-best arms.

H21 result: dlplan `equal-closure` @ b2df1699 (pushed): new element `EqualClosureConcept`, parser syntax
`c_equal_closure(R1,R2)`, factory/bindings/stubs, generator rule at complexity 3 for goal pairs, kwarg
`generate_equal_closure_concept` (default true), unit test; build clean, core tests 43/43, genfond suite
208 passed. Pool check: the concept is generated at complexity 3. Local 25 min training on 2–7 blocks +
010-5 + 017-5 (graceful stop): 30/32, 6 rules using plain `c_equal`, **90/95**; the closure concept was not
selected before the cap (its negation and boolean sit at complexity 4–5). Regression suites reproduce the
baselines exactly under the default config. genfond pinned to b2df1699 (learn-from-examples and hyp/combo);
image rebuild running locally for the cluster runs.
Image rebuilt locally in 10 min (`make genfond_env.sif`, 1.1 GB) from the re-pinned lock; it parses
`c_equal_closure(...)` at complexity 3 and exposes `generate_equal_closure_concept`. Copied to
`/work/.../genfond_env_ec.sif` (new name so running jobs keep their open image).
Runs with the new grammar (cluster worktree combo7, image `genfond_env_ec.sif`, config: add-problem, budget
300 s, role offset 2 / concept offset 1, restarts 1, cap 8, keep-best; default `generate_equal_closure_concept`):
ec-mod (modern 4 h) and ec-cpu (cpu 12 h) on the full 95; workstation: 2–7 blocks + 010-5 + 017-5 with a
2 h wall budget, scored on all 95.

## RESULT: 95/95 with synthesised features and the extended grammar

Workstation, image with dlplan b2df1699 (`c_equal_closure`), config: add-problem, budget 300 s, role offset 2 /
concept offset 1, restarts 1, cap 8, keep-best, seed 2. Training on 2–7 blocks + 010-5 + 017-5 (32 problems):
32/32 in **63 s**, **6 rules**, five of them on `c_equal_closure(on, on_g)` (the well-placed concept), the
boolean `b_empty(c_some(on, clear_g))`, and the roles `on_g` / `r_not(on_g)`. Scores: **95/95** on the full
suite, **12/12** on the held-out 8–30-block set. No hand-given features; the concept came out of the grammar
at complexity 3. Robustness runs (other seeds, ≤7-block training only) and the full-95 loop arms follow.

Robustness (workstation, new image): training on **2–7 blocks only** (30 problems) → seed 0: 30/30, 6 rules,
closure concept selected, **95/95**; seed 1: 30/30, 6 rules, plain `c_equal` chosen (equal cost), **92/95**;
seed 2: 30/30, 6 rules, closure, **95/95**; all in ~33 s. Regression suites in the new image, default config:
gripper 5/5 c6, miconic 4/4 c12, blocks4ops-clear 4/4 c2, delivery 4/4 c9 — identical to the baselines.

## RESULT: the full 95-problem loop completes with 95/95

ec-cpu (4161148, `rleap_cpu`, new image, recommended configuration, seed 0, single thread): the iterative
loop over all 95 problems **completed normally after 18 min** (`COMPLETED`, not a timeout): 6 training
problems (≤4 blocks), best policy at round 12, **95/95 solved**, 6 rules on `c_equal_closure(on, on_g)`,
`c_equal(on, on_g)` as boolean, and `r_not(on_g)`. No curated training set, no hand-given features.
That policy also solves **12/12 held-out** instances (8–30 blocks).

## Cross-domain test of the working configuration (Till's request)

Submitted 2026-09-16 ~12:45: the recommended configuration (worktree combo7, image `genfond_env_ec.sif`,
`claude-experiments/rec.yaml`: add-problem, budget 300 s, role offset 2 / concept offset 1, restarts 1, cap 8,
keep-best; graceful 12 h budget) on all 21 other deterministic domains, one job each on `rleap_cpu` (tag
`rec`), and the original system (worktree base @ 238ae61, old image, datalog-sig defaults, seed 0) on the
same domains on `rleap_cpu_modern` (tag `old`, plain 12 h limit). Domains: deterministic/{barman 30,
blocks4ops 95, blocks4ops-clear 95, blocks4ops-on 190, gripper 30, hanoi 30, miconic 25, sokoban, storage,
visitall}, deterministic-new/{blocks 12, blocks-multiple 22, delivery 225, grid 14, gripper 10, logistics,
logistics_dp, miconic, reward, spanner, visitall}. Note: gripper and miconic exist in both directories and
share job names; tell rows apart by problem count and job id.
Seed-1 tie (new grammar): with 010-5 + 017-5 in training → 30/32 at the 25 min budget, plain `c_equal`
policy, 90/95; with `minimize_selected_count: above` on the 2–7-block suite → 30/30, 91/95, still `c_equal`.
`c_equal` and `c_equal_closure` tie on complexity (3) and element count (1), so the model choice decides;
2 of 3 seeds pick the closure and reach 95/95. A cheap outer restart over seeds with keep-best would make the
result seed-independent (each attempt is ~30 s); not implemented.

Cross-domain snapshot after 73 min (rec = working configuration, old = original system):

| domain | old | rec |
|---|---|---|
| blocks4ops-clear (95) | 95/95 in 16 s | 95/95 in 47 s |
| blocks4ops-on (190) | 190/190 in 33 s | 190/190 in 4.5 min |
| gripper (30) | 30/30 | 30/30 in 25 s |
| blocks4ops (95) | round 1 after 73 min | round 7, first failure at 62 |
| delivery (225) | round 1 | round 22, first failure at 116 |
| blocks (12) | round 1 | round 20, first failure at 17 |
| hanoi (30) | round 1 | round 11, first failure at 15 |
| miconic (25) | round 1 | round 9, first failure at 14 |
| spanner | round 4, first failure at 86 | round 4, first failure at 9 |
| barman, grid, logistics, logistics_dp, reward | round 1 | rounds 1–9 |
| blocks-multiple, deterministic-new/miconic | crash within seconds | crash within seconds |

The original system sits in its first round on every hard domain; the working configuration is 7–22 rounds
in. Two domains crash under both systems (domain-file issue, investigated). Spanner is the one domain where
the baseline's early policy covers more.
Crash causes: deterministic-new/miconic fails to parse under both systems (`PDDLMissingRequirementError:
:typing not found`, a domain-file issue, not a system difference).
blocks-multiple: problem files reuse names (BW-rand-4 ×4, …); the original system asserts
("Problem names must be unique"), the new one runs but its per-name bookkeeping is confused (8/48 with
duplicate unsolved entries). Excluded from the comparison; the suite needs unique problem names.
Cancelled as superseded by the new-grammar result (Till, 2026-09-16): 4147766 (full2-long, 36 h),
4154631 (kb-lite-cpu), 4154622 (kb-r1-cpu); all pre-grammar full-suite arms.

**Caveat on the full-suite loop:** ec-mod (4161147, same configuration and seed as ec-cpu but on
`rleap_cpu_modern`, graceful 4 h): **28/95** (best round 12, 19 training problems ≤7 blocks) against 95/95
in 18 min on `rleap_cpu`. With a per-solve time budget the trajectory depends on the machine's speed, so a
single full-suite run is not a reliable reading. Repeating the full loop with seeds 1–3 on `rleap_cpu`
(4 h, graceful). The small-suite path (≤7 blocks, ~33 s) remains the robust and cheap route: 2 of 3 seeds
give 95/95, and an outer restart over seeds with keep-best would make it seed-independent.

Cross-domain snapshot at 8 h 20 (finished rows are final; "running" rows give rounds / training-set size /
position of the first failing problem):

| domain | old | rec |
|---|---|---|
| blocks4ops-clear (95) | 95/95, 16 s | 95/95, 47 s |
| blocks4ops-on (190) | 190/190, 33 s | 190/190, 4.5 min |
| gripper ×2 | 30/30, 30/30 | 30/30, 30/30 |
| visitall (500) | 500/500, 43 min | 500/500, 3 h |
| **visitall (36)** | **36/36, 4.6 min** | **8/36, ended after 2 h** (regression, investigated) |
| miconic (25) | 14/25, ended after 2 h | running: 12 training problems |
| logistics (47) | 9/47, ended after 6.9 h | running: round 10 |
| blocks4ops (95) | round 1, 10 problems | round 7, 21 problems, first failure 63 |
| delivery (225) | round 1, 11 problems | round 6, 33 problems, first failure 109 |
| blocks (12) | round 1, 3 problems | round 5, 27 problems (?), first failure 25 |
| hanoi (30) | round 1, 4 problems | round 1, 8 problems |
| logistics_dp | round 1, 5 | round 17, 8, first failure 15 |
| reward | round 1, 5 | round 10, 7, first failure 6 |
| spanner | first failure 34 | round 4, 3 problems, first failure 9 |
| storage | round 1, 6 | round 1, 3, first failure 3 |
| barman, grid, sokoban | round 0–1 | round 0–9 |
| blocks-multiple, miconic-new | invalid suites (see above) | – |
visitall-36 regression cause: the baseline's 1-rule policy uses the complexity-6 role
`r_til_c(connected, visited_g ∧ ¬visited)`; with `role_complexity_offset: 2` that role only enters the pool
at complexity 8, so the rec run climbed to complexity 6 (7 120 concepts / 3 862 roles) and stalled at 8/36.
The offsets trade expressiveness for tractability; with `c_equal_closure` in the grammar blocks3ops may no
longer need them. Testing the recommended configuration without offsets on visitall-36 and on the blocks3ops
≤7-block suite (3 seeds, scored on 95).

**Why spanner (and other large suites) take hours — Till's question.** `deterministic-new/spanner` has 141
problems (up to 22 spanners / 11 nuts / 10 locations). The rec run's four learning rounds took milliseconds
each; at 12:35 it entered the in-loop policy test and was still in it 8.5 h later: every candidate is executed
on all 141 problems × `policy_iterations` = 10, with a DL evaluation per step (~50 s per execution on the
large instances) and a `DEBUG eval to cond` line per condition (21 825 lines). `keep_best_policy` removed the
early break at the first failure, so each candidate pays the whole suite. Learning from example plans is not
the slow part; execution-based validation is. Fix in progress (`hyp/cheap-validation`): one execution per
problem in the loop, size order with early stop after consecutive failures, per-problem time cap, full
verification only at the end, per-step logging off.
Till: ten executions per candidate were for FOND; for deterministic problems repetition only matters
because the datalog executor samples among applicable rule instantiations. Decision: in-loop validation
executes each problem once; the datalog configs' final verification drops `policy_iterations` from 10 to 3.

**Full-suite loop, new grammar, five runs:** ec-cpu (seed 2, hash 0) 95/95 in 18 min; ec-s3 95/95 in 19.5 min;
ec-mod (seed 2, modern node), ec-s1, ec-s2 all **28/95**, each with best policy at round 12 from 19 training
problems and stopped by the 4 h budget. The successful runs also peak at round 12 (6 training problems). The
outcome is decided at round 12 by which equal-cost optimal model clingo returns: `c_equal_closure` (→ 95) or
`c_equal` (→ 28), the same tie as on the small suite. Planned fix (H22): enumerate several optimal models per
round and let the coverage test choose (keep-best already ranks by coverage).

**Offsets are a trade-off, not a free win:** without role/concept offsets, blocks3ops ≤7-block training gives
5-rule `c_equal` policies at 89–90/95 for all three seeds (with offsets: 2 of 3 seeds 95/95), and visitall-500
reaches 26/500 in 50 min (with offsets: 500/500 in 3 h). visitall-36 (the `deterministic-new` suite) needs a
complexity-6 role that the role offset delays to complexity 8; the offsets help where the needed elements are
cheap concepts and hurt where a deep role is needed.

### Cross-domain A/B, final (12 h, seed 0, single thread; rec = working configuration with graceful budget,
old = original system)

| domain (problems) | old | rec | verdict |
|---|---|---|---|
| blocks4ops-clear (95) | 95/95, 16 s | 95/95, 44 s | equal |
| blocks4ops-on (190) | 190/190, 33 s | 190/190, 4.5 min | equal |
| gripper (30) ×2 | 30/30, 30/30 | 30/30, 30/30 | equal |
| visitall (500) | 500/500, 43 min | 500/500, 3 h | equal, slower |
| **blocks4ops (113)** | 52/113 | **106/113** (19 training problems ≤9) | rec |
| **blocks (35)** | killed in round 1 | **29/35** (27 problems ≤13) | rec |
| **hanoi (30)** | killed in round 1 | **30/30** in 10 h | rec |
| **delivery (225)** | killed in round 1 | **139/225** (31 problems ≤9) | rec |
| **miconic (25)** | 14/25 | **18/25** | rec |
| **logistics_dp (47)** | 9/47 | **17/47** | rec |
| logistics (47) | 9/47 | 9/47 | equal |
| **visitall (36)** | **36/36**, 4.6 min | 8/36 | **old** (needs a complexity-6 role; role offset delays it) |
| barman, grid, reward, spanner | killed in round 0–1 | killed at the limit inside validation, no final line (reward best 5/35) | no result |
| sokoban, storage | still running | still running | pending |
| blocks-multiple, miconic-new | invalid suites | – | – |

Reading: no domain regresses except visitall-36, which is the offset trade-off; where the original system
could not finish its first round, the working configuration reaches 8–13-object training sets and much
higher coverage. Two infrastructure limits showed: the validation loop dominates large suites (fix in
progress) and the graceful stop cannot interrupt it (the four killed rec runs), so those runs lost their
stats and best policies.

## H23: cheap in-loop validation (`hyp/cheap-validation`, 299d7fe, 218 tests)

`validation_iterations: 1`, `validation_max_consecutive_failures: 3` (size order, lower-bound count),
`validation_time_limit: 60` (per execution, monotonic clock inside the executors, `ExecutionTimeout`),
deadline/stop checks between problems so the graceful stop works inside validation; datalog configs'
`policy_iterations` 10 → 3; the leaked per-step `eval to cond` lines went to the `genfond.execution` loggers
(they came from `generate_rule_policy`'s logger, which the log map did not silence). Workstation validation
(regression suites + spanner 141 with a 50 min budget) running. H22 (`hyp/opt-enum`, on top of it):
enumerate up to k optimal models per round, discard those violating lazy pairs, validate each cheaply, keep
the best coverage — the tie-breaker for the round-12 fork.

H22 implemented (`hyp/opt-enum`, 3d08ff7, 231 tests): `optimal_model_limit: k` enumerates up to k proven-optimal
models on the final lazy-pairs program (`opt_mode=optN`), discards models violating any pair, validates each
cheaply and keeps the highest coverage (ties: fewer rules, then clingo's order); default 1 = old behaviour.
Local suites: identical solved/cost; 2–3 optimal models per round everywhere, blocks4ops-clear picks a
higher-coverage sibling and converges one round earlier. Workstation: ≤7-block training, seeds 0–2, limit 1 vs
3, scored on 95; cluster: full loop with limit 3 for seeds 1 and 2 (the ones that forked to 28).
H23 result (workstation, new image): regression suites unchanged (gripper 5/5, miconic 4/4, blocks4ops-clear
4/4, delivery 4/4 at the rec config's costs); **spanner 140/140 in 17 min** (8 rounds, 3 training problems,
validation 1 019 s of 1 027 s, 2 early stops, no per-step log lines) where the previous run sat in one
validation block for 12 h. Merged into `hyp/combo`.
Resubmitted under the working configuration with cheap validation (worktree combo8 @ 299d7fe, `rec2`, 4 h
graceful): barman, sokoban, storage, miconic, blocks4ops, grid, reward, spanner, logistics, logistics_dp,
delivery, blocks.

### First batch, last two jobs (sokoban, storage; 12 h, rec)

Both hit the SLURM limit without a final line. sokoban never left its first solve: the log ends at
"Starting solver for p032-microban-sequential with max complexity 2" at 14:32 and nothing follows for 12 h,
so the time went into state-space expansion / feature generation for a single 8-object problem, where
neither the solve budget nor the graceful stop can interrupt. storage learned a policy in the first minutes
and then spent the remaining 11.5 h validating it: every execution on storage-15/16 ran 20–60 min before
detecting a length-2 cycle, ten times per problem. Cheap validation (H23) addresses storage; sokoban needs a
budget on the state-space/feature stage (not implemented; the rec2 run will show whether it is the same
stage again).

## H24: offset compromise (`hyp/combo`, workstation, seed 0, 1 thread)

visitall-36 lost to the original system because the role offset 2 delays the complexity-6 role. Arms with
smaller offsets, each on the 12-problem `deterministic-new/visitall` suite (40 min budget) and on the
≤7-block blocks3ops suite scored on all 95:

| arm | visitall (12) | blocks3ops ≤7 → on 95 | closure rules |
|---|---|---|---|
| r2c1 (current) | 8/36 in the 12 h run | 30/30 → 95/95 (2 of 3 seeds) | yes |
| r1c1 | 7/12, 36 min | 30/30 → **95/95** | 7 |
| **r0c1** | **12/12 in 2.4 s** | 30/30 → **95/95** | 8 |
| r2c0 | 8/12 at the 35 min budget | 26/30 at the 30 min budget → 26/95 | no |

Reading: the concept offset (1) is what makes `c_equal_closure` affordable early; the role offset buys
nothing on blocks3ops at this suite size and costs visitall its complexity-6 role. r0c1 is the new
candidate default (one seed so far; seeds 1–2 and the rec2 domains still to confirm).

## H22 result, part 1 (workstation, ≤7-block suite, scored on 95, `hyp/opt-enum` 3d08ff7)

| seed | limit 1 | limit 3 |
|---|---|---|
| 0 | 30/95 (no closure, 1 418 s: the fork trajectory) | **91/95**, 10 s, 2 model switches |
| 1 | **95/95**, 14 s | 85/95, 1 502 s, 5 switches |
| 2 | 90/95, 13 s | **95/95**, 16 s, 3 switches |

Enumeration turned seed 0's fork trajectory (30/95 after 24 min) into a 10 s run at 91/95 and seed 2 into
95/95, but sent seed 1 down a 25 min trajectory ending at 85/95. Mean 72 → 90 of 95, yet not a clean win:
the coverage tie-break on the ≤7-block training set does not identify the general policy reliably, because
several siblings tie on the small instances too. Combined with role offset 0 (H24) it is being repeated. Cluster: full 95-problem loop with limit 3 for seeds 1 and 2 running (`L3-s1`, `L3-s2`, 4 h).

**Sokoban diagnosed** (desk-03, 12 min verbose run on the first problem): SIW finds no plan for
p032-microban-sequential, so the round has no example plans and `StateSpaceGraph` falls back to the
unrestricted expansion of an 8-object sokoban instance; the feature-pool stage never returns and the graceful
stop cannot interrupt it. Same stage in rec2. Candidate H25 (not started): when the planner yields no plan for
a problem, defer it instead of expanding unrestrictedly, and/or put a state budget on the expansion.

**H24, seeds 1–2:** r0c1 keeps visitall at 12/12 in 2.5 s for all seeds, but on blocks3ops ≤7 seeds 1 and 2
give 77/95 and 71/95 without the closure concept (seed 0: 95/95). r2c0 loses visitall (8/12 at the 35 min
budget). So the role offset is what costs visitall, and without it blocks3ops depends on the equal-cost
tie-break — exactly H22's target. Running: r0c1 + `optimal_model_limit: 3`, seeds 0–2, ≤7 → 95.

## RESULT: role offset 0 + concept offset 1 + optimal-model enumeration (`hyp/opt-enum`, cluster, full 95-problem loop)

Config = recommended config with `role_complexity_offset: 0`, `concept_complexity_offset: 1`,
`optimal_model_limit: 3` (`claude-experiments/r0c1-L3-s*.yaml` in the cluster worktree `optenum`), 1 thread,
modern partition:

| seed | full 95-problem loop | ≤7-block training → on 95 (workstation) |
|---|---|---|
| 0 | **95/95 in 157 s**, best at round 11 | 95/95, 25 s, 1 switch |
| 1 | best 91/95 at round 11, final combination 92/95 at the 3.6 h limit | 95/95, 24 s, 1 switch |
| 2 | **95/95 in 20 s**, best at round 11 | 27/30 after 25 min → 26/95 |

Against the previous full-loop figure (2 of 5 runs 95/95 in ~19 min, the rest at 28/95), the loop now
converges in seconds: without the role offset the closure concept and its siblings tie at equal cost early,
and the coverage tie-break over all 95 problems picks the general one. visitall-12 is 12/12 in 2.5 s under
the same offsets (H24). Pending: seed 1, the rec2 domains under this setting, and the regression suites.

`hyp/opt-enum` merged into `hyp/combo` (fast-forward to 3d08ff7; black/isort/mypy clean, 231 tests pass).
Launched under the new setting (`claude-experiments/rec3.yaml` in the cluster worktree `optenum` = rec +
role offset 0 + limit 3): the twelve rec2 domains as `rec3` (modern partition, 4 h graceful), visitall-500
(`r0c1-L3-vis500`), the regression suites and the held-out set on the workstation.

Early rec3 / regression numbers under the new setting: **blocks (35) 35/35 in 23 s** (rec: 29/35 after 12 h);
regression suites unchanged (gripper-local 5/5, miconic-local 4/4, blocks4ops-clear-local 4/4, delivery-local
4/4, all ≤4 s). Workstation ≤7-block training, seed 2: 27/30 after 25 min → 26/95 (cost climbs 18 → 23 across
rounds 19–22 with 4 feasible siblings each time), while the full loop with the same seed converges in 20 s.
The small-suite training remains trajectory-sensitive; the full loop is the configuration to report.
Held-out (12 instances, 8–30 blocks): the full-loop policies of seeds 0 and 2 both solve **12/12**.
**blocks4ops (113): 113/113 in 3.8 min** (rec: 106/113 at 12 h; original system 52/113).

**H22 with the role offset kept (cluster, full loop, r2c1 + limit 3, 3.6 h graceful):** seed 1 30/95, seed 2
33/95 — the same plateau as without enumeration. Enumeration alone does not fix the fork; only together with
role offset 0 (above). Seeds 3–5 of the r0c1 + limit 3 full loop submitted for a robustness estimate.

**rec2 (cheap validation, r2c1), first two finished at the 3.6 h graceful limit:** miconic 19/25 (rec 18,
old 14), blocks4ops 108/113 (rec 106). Both keep-best policies from rounds 12 / 21 with later rounds patchworking.
**visitall (500) under the new setting: 500/500 in 10.5 min** (rec 3 h, original system 43 min).
Full-loop seed 3: 95/95 in 126 s.
rec2 logistics 9/47 and logistics_dp 9/47 at the 3.75 h graceful limit (12 h rec: 9 and 17; the dp run's
17 came after the 4 h mark, so this is a budget effect, not a regression of cheap validation).

### rec2 batch complete (cheap validation, r2c1, 4 h SLURM limit, 3.6 h graceful)

| domain | rec2 | rec (12 h) | note |
|---|---|---|---|
| miconic (25) | 19 | 18 | best policy from round 12 |
| blocks4ops (113) | 108 | 106 | best from round 21 |
| logistics (47) | 9 | 9 | budget-bound |
| logistics_dp (47) | 9 | 17 | budget-bound (17 came after 6.7 h) |
| reward (35) | best 16 in the log, no stats row | best 5 | killed by SLURM |
| spanner (162) | best 130 in the log, no stats row | killed | killed by SLURM |
| barman (30) | 0, climbing complexity 7 on two problems | killed | killed by SLURM |
| grid, storage | no policy, complexity 12+ / 15+ on two problems | killed | killed by SLURM |
| sokoban | stuck in the first feature pool (no SIW plan) | same | killed by SLURM |
| blocks (35) | 27 | 29 | rec3: 35/35 in 23 s |
| delivery (225) | 140 | 139 | |

Infrastructure finding: five runs were killed by SLURM at 4 h although the graceful deadline was 3.6 h,
because the deadline is only checked between phases and a grounding at complexity ≥ 7 on multi-object
instances outlasts the reserve. When that happens the keep-best policy and the stats row are lost (reward's
16/35, spanner's 130/162 exist only in the log). Fix in progress (H26): write the best policy to the output
path whenever it improves, and write the stats row at the graceful request, not only at exit.

## H26: checkpoint the best policy and the stats row (`hyp/checkpoint-best`, b531de1, merged into `hyp/combo`; 249 tests)

`genfond/checkpoint.py`: whenever the best coverage improves the policy is pickled atomically to the output
path; on a graceful-stop request a provisional stats row (keyed by a per-run id) is appended and replaced by
the final row at exit. `checkpoint_best_policy: true` by default. A kill that lands inside `clingo.ground()`
still loses the stats row, but never the best policy any more. Cluster worktree `combo9` = `hyp/combo` @
b531de1 with `claude-experiments/rec3.yaml` (the new setting) for the next batches.
rec3 grid (26): 1/26, stopped by max complexity after 2.9 h on two 11-object training problems (every
solve at complexity ≥ 9 hits the 300 s budget, 81 GB); rec2 grid was killed at 4 h with no policy. grid is
grounding-bound like the pre-fix blocks3ops and needs a different idea, not a budget.
rec3 logistics_dp (47): **18/47 in 3.3 h** (rec 17 at 12 h, rec2 9 at the 3.75 h limit; the final round
combined a 15/47 keep-best with the last policy).

**Seed 1 diagnosis (full loop):** round 11 enumerates 4 optimal models of cost 5, 2 survive the pair check,
coverages [91, 87]; the general cost-8 policy is more expensive and so never enumerated there. Round 12 adds
a problem the cost-5 policy fails on, and the minimum cost jumps to 16 (with frontier transitions, cost
vectors `[1, 16]`): the general policy is *not feasible* on the enlarged instance although it solves every
problem when executed — the newly added problem's sampled SIW plans (restarts 1, cap 8) do not contain the
trajectory the general policy takes, so under the plan-restricted state space it is infeasible. Whether the
sample happens to contain that trajectory is what the seed decides. H27 (in progress): add the trajectory of
the best policy on every problem it solves as an example plan (policy-conformant plans), so a general policy
found once stays feasible in every later round.
rec3 miconic (25): 14/25 (rec2 19, rec 18, original 14) — the only domain so far where the new setting is
worse than the role-offset setting; rec3 delivery (225): **171/225** (rec2 140, rec 139).

## H27: policy-conformant example plans (`hyp/policy-plans`, 1479325, on `hyp/combo`; 258 tests)

The executors can now return the ground actions they applied; for every problem the round's policy solves
during validation the trajectory is stored as an example plan in front of the problem's SIW plans (exempt
from `max_plans_per_problem` and from the `min_number_of_plans` floor; deduplicated by state set), and the
refutation bookkeeping is invalidated like after a plan addition (the add-a-problem branch drops its carried
bound when plans were added). `policy_conformant_plans: true` by default. Found on the way: the
add-a-problem branch reset the problem's plan list, which would have discarded any plan recorded before the
problem joined the training set (now `setdefault`). A/B launched: full 95-problem loop, seeds 0–5 (`pp-s*`,
cluster worktree `policyplans`, 4 h graceful) against the r0c1-L3 arms; regression and cross-domain quick
suites on the workstation (`results-ab/pp-reg.summary`).
rec3 logistics (47): **15/47** (rec 9, rec2 9).
H27 first numbers: full loop pp-s2 95/95 in 47 s (81 policy plans recorded), pp-s5 95/95 in 96 s; regression
suites unchanged (1–3 policy plans each); blocks-new (12) 12/12 in 43 s.
rec4 = the twelve rec2/rec3 domains under H27 + the rec3 setting (cluster worktree `policyplans`, standard
partition, 4 h graceful) submitted for the three-way comparison rec2 / rec3 / rec4.
rec3 endings at the 4 h kill (no checkpointing on that worktree, best policies lost): barman 0/30 (climbing
complexity 7 on two problems), reward best 15/35 at round 27 (rec2 16), spanner-162 best **137/162** at
round 6 (rec2 130) — its log grew to 32 GB of DEBUG state dumps under `VERBOSE=1` on the large instances,
which is where the run's time went; sokoban stuck in the first feature pool again.
H27 full loop pp-s4: 95/95 in 389 s. rec4 (H27): blocks4ops 113/113 in 231 s, blocks 35/35 in 85 s.
The flood is the per-transition DEBUG line in `state_space_generator` (one line per successor, each with two
full state strings); future batch configs carry `log: {state_space_generator: INFO}`.
rec3 storage: killed at 4 h without a policy (complexity climb on two problems), as in rec2. rec3 complete.

### Cross-domain summary so far (4 h graceful, seed 0, 1 thread)

rec2 = cheap validation, role offset 2 / concept offset 1; rec3 = rec2 + role offset 0 + enumeration limit 3;
rec4 = rec3 + H27 policy-conformant plans (running).

| domain (problems) | rec (12 h) | rec2 | rec3 | rec4 |
|---|---|---|---|---|
| blocks3ops (95), full loop, seeds 0–5 | 95 in 19 min (2 of 5 runs) | – | 95, 92, 95, 95, 36, 30 | 93, 95, 95, 40, 95, 95 |
| blocks4ops (113) | 106 | 108 | **113** in 3.8 min | **113** in 3.9 min |
| blocks (35) | 29 | 27 | **35** in 23 s | **35** in 85 s |
| visitall (500) | 500 in 3 h | – | **500** in 10.5 min | |
| visitall (12/36) | 8 | – | 12 in 2.5 s | |
| delivery (225) | 139 | 140 | **171** | 162 |
| miconic (25) | 18 | 19 | 14 | **21** |
| logistics (47) | 9 | 9 | 15 | **19** |
| logistics_dp (47) | 17 (12 h) | 9 | 18 | **23** |
| reward (35) | 5 | 16 (log only) | 15 (log only) | **20** (checkpointed) |
| spanner (162) | killed | 130 (log only) | 137 (log only) | 129 (checkpointed) |
| barman (30) | killed | 0 | 0 | 0 |
| grid (26) | killed | killed | 1 (max complexity) | 1 |
| storage (30) | killed | killed | killed | 3 (checkpointed) |
| sokoban | killed | killed | killed | killed (no SIW plan) |
H27 workstation regressions: visitall-12 12/12 in 1.4 s, visitall-500 500/500 in 10 min (2 policy plans),
blocks4ops on the flat 95-problem directory (not the 113-problem cluster suite, which includes `old/`):
78/95 at the 55 min budget (best single policy 43/95 at round 13, final combination 78) — a different, harder
problem set than the cluster's, so not comparable with the 113/113 above; worth a cluster run of that set.
Housekeeping: seven stale local waiter loops from 2026-09-14/15 were still alive because their `pgrep -f`
pattern matched the waiting shell itself; killed.

### H27 result (full loop, seeds 0–5, 4 h graceful)

| seed | rec3 setting | + H27 policy plans |
|---|---|---|
| 0 | 95 in 157 s | best 91 at round 13, final combination 93 |
| 1 | best 91, final 92 | best 89 at round 13, final combination **95** at 3.6 h |
| 2 | 95 in 20 s | 95 in 47 s |
| 3 | 95 in 126 s | 40 (best 40 at round 47) |
| 4 | 36 | 95 in 389 s |
| 5 | 30 | 95 in 96 s |

Policy plans lift the tail (two seeds from 30–36 to 95, seed 1 to 95) but do not remove the dependence on
the early plan sample: seed 3 never sees a general candidate and seed 0 loses one round to a sibling. Mean
80.5 → 85.5 of 95; 3 → 4 seeds at 95. Cross-domain (rec4 column above): miconic 21/25 (best of all
settings), logistics 19 and logistics_dp 23 (best), reward 20/35 and storage 3/30 checkpointed for the first
time, delivery 162 (rec3 171), spanner 129 (rec3 137). H26 checkpointing worked: every rec4 job left a
policy file, including the four killed at 4 h.
Running (H28, diversity for the plateau seeds 3–5 under H27): `planners.siw.restarts: 2` (`pp-r2-s*`) and
`optimal_model_limit: 6` (`pp-L6-s*`); the flat 95-problem blocks4ops set under rec3 and H27.
H27 workstation miconic (25): 24/25 at the 55 min budget.

## Benchmark consolidation (Till's request, 2026-09-17)

`domains/` reorganised on `learn-from-examples` (88189e7): one directory per domain encoding under
`deterministic/`, held-out instances in `test/`, FOND tree untouched, `d2l/` and `deterministic-new/` gone
(unparseable, duplicate or redundant), problem names unique, `run_benchmarks.bash` non-recursive with the
deterministic tree as default. Full map in `docs/benchmarks.md`. Result directories and the hyp/* branches on
the cluster still use the old paths; the mapping is: deterministic-new/X → deterministic/X (delivery, grid,
logistics, reward, spanner), blocks3ops-heldout → blocks3ops/test, d2l/blocks3ops → blocks-atomic.

**blocks4ops, flat 95-problem set (2–20 blocks, arbitrary goals; the 113-set adds `old/`):** rec3 setting
36/95 at the 3.6 h limit (best 35 at round 32) versus 78/95 in 55 min with H27 on the workstation; the
H27 cluster arm is running. So the 113/113 figures above rest on the extra 18 instances steering the
trajectory; on the generated set alone blocks4ops behaves like blocks3ops before the fixes and H27 is what
moves it. H28 seed 5: 95/95 under both arms (restarts 2: 318 s, limit 6: 423 s; H27 alone 96 s).
blocks4ops flat, H27 cluster arm (3.6 h): best single policy 48/95 at round 16, final combination **63/95**
(rec3 36; workstation H27 run 78 at 55 min — a different trajectory, same picture). blocks4ops with arbitrary
goals is not solved by the current loop; its general policy needs the same kind of goal-relative concepts as
blocks3ops but the loop does not find it on either set. Open, next candidate after H28.
Correction: the flat 95 instances are a subset of the 113-problem run (95 + 18 in `old/`), so the 113/113
policy solves every flat instance. The loop *can* express and find the blocks4ops policy; on the flat set alone
it does not, because the 18 small `old/` instances (2–15 blocks, another generator) steer the early plan
sample. Same trajectory effect as blocks3ops seeds 3–5, not an expressivity gap — H28's diversity arms are
the right test for it too.

### H28 result (H27 + diversity, full loop, 4 h graceful)

| seed | H27 (restarts 1, limit 3) | restarts 2 | limit 6 |
|---|---|---|---|
| 3 | 40 | 95 (best single 86 at round 13, final combination at 3.6 h) | 95 (best 93 at round 13, final at 3.6 h) |
| 4 | 95 in 389 s | 93 (best 88 at round 13) | 32 (best 31 at round 34) |
| 5 | 95 in 96 s | 95 in 318 s | 95 in 423 s |

Neither arm is a clean improvement: each moves which seed plateaus rather than removing the plateau (seed 4
under limit 6 falls to 32). The invariant across all runs is that a near-general policy (86–93/95) appears
around round 11–13 and the loop then drifts for hours; where 95 is reached late it is the final combination
that recovers it. The plateau is therefore in what happens *after* the near-general policy: the next added
problem enlarges the instance until the general policy is no longer the cheapest model. H27 keeps it
feasible; nothing yet keeps it *preferred*. Candidate H29: after a best-coverage improvement, re-solve the
round with the failing problems' policy-plans only (or cap plans on newly added problems to the
policy-conformant ones) so the enlarged instance stays close to the one the general policy came from.
Not started. blocks4ops-flat under both arms is running.

## H29: policy-anchored labels (`hyp/anchor-labels`, e68be54, on `hyp/policy-plans`; 271 tests)

`anchor_policy_labels: true` (default false): every transition along the best policy's recorded trajectories
on the problems currently in the training set is emitted as `anchor(I,S,A)` and a separate `#program anchor.`
part adds `:- anchor(I,S,A), not good_action(I,S,A).`, so the round's model must agree with the best policy
wherever it already succeeds (in the signature encoding one anchored occurrence forces the whole signature
class, which is the program's own consequence). Anchors are a preference, never evidence: an anchored
UNSAT/UNKNOWN re-solves the identical instance without anchors (`anchorFallbacks`) and only that solve can
refute a complexity level. Anchors whose step is not in the round's graph are dropped and counted. Launched:
full 95-problem loop, seeds 0–5 (`an-s*`, cluster worktree `anchor`, rec3 setting + H27 + anchors, 4 h
graceful) and the flat blocks4ops set (`an-b4flat`).
Note: a push with the branch's inherited upstream briefly fast-forwarded `origin/hyp/policy-plans` to
e68be54; Till restored it to 1479325.
H29 first seeds (full loop): seed 0 **95/95 in 67 s**, seed 1 **95/95 in 73 s** (seed 1 was 92 under rec3 and
95 only at 3.6 h under H27). Anchors are active in the decisive rounds (6–11 transitions from 2–3 problems)
with an occasional fallback.
H29 seeds 3 and 4: **95/95 in 259 s and 117 s** (seed 3 was 40 under H27, 95 only at 3.6 h under H28). Four
of four finished seeds at 95 within 5 min; seeds 2 and 5 are at 90/91 after their first ten rounds and running.
blocks4ops flat under H28 (H27 + diversity, 3.6 h): restarts 2 → 57/95 (best single 40 at round 16);
enumeration limit 6 → **80/95** (best single 80 at round 14). Together with 36 (rec3) and 63 (H27): every arm
finds its best policy around round 14–16 and drifts afterwards, the same signature as blocks3ops. H29 on the
flat set is running (33/95 at round 30 after 40 min); H29 seeds 2 and 5 sit at 90/91 from round 9/11.

**H29 seed 2 diagnosis:** round 9 gives the cost-8 near-general policy P (90/95, 3 training problems). Round
10 adds blocks-010-5, which P fails on. With anchors (10 transitions, 3 problems) the cheapest models are
frontier models of cost 18–19 (`[1,19]`, `[3,18]`, `[2,18]`): on the enlarged instance no anchored model near
cost 8 exists, because 010-5's example plans (SIW, restarts 1, cap 8) do not contain the trajectory a general
policy takes on it, and H27 cannot supply one — P fails on 010-5, so it has no trajectory there. Three
frontier rounds later the loop is at complexity 5 with patchwork costs. The missing piece is therefore the
*new* problem's plans: H30 (in progress) executes P on the newly added problem as far as it gets, re-roots
the problem at the last state before the failure (cycle or no applicable rule) and asks SIW for the rest,
splicing the two into a root-anchored example plan ("policy-prefix plans", reusing `frontier.py`), so a small
modification of P is feasible on the enlarged instance.

## H30: policy-prefix example plans (`hyp/prefix-plans`, 975deab, on `hyp/anchor-labels`; 289 tests)

`policy_prefix_plans: true` (default, inert without example plans): when a problem is added to the training
set and a best policy exists, the policy is executed on it, the trajectory is cut at the first repeated state
(or at the failure), the problem is re-rooted there and SIW plans the rest (`prefix_plan_count: 2`, backoff
dropping 1/2/4 trailing actions when the full prefix has no plan, `prefix_plan_max_length: 200`); the spliced
root-anchored plans join the problem's example plans in front, exempt from the caps, and the refutation is
dropped. Launched: full loop seeds 0–5 (`px-s*`, cluster worktree `prefix`, = H29 setting + H30) and the flat
blocks4ops set (`px-b4flat`).
H30 first seeds (full loop): seed 2 **95/95 in 84 s**, seed 3 95/95 in 77 s, seed 5 **95/95 in 44 s** — seeds
2 and 5 are exactly the two that sit at 90/91 for hours under H29 alone (still running there). Seeds 0, 1, 4
are at 91/95 after 11–15 rounds with prefix plans being added for the 10–11-block problems; running.

### H29 result (full loop, seeds 0–5, 4 h graceful)

| seed | rec3 | H27 | H29 (H27 + anchors) | H30 (+ prefix plans) |
|---|---|---|---|---|
| 0 | 95 in 157 s | 93 | **95 in 67 s** | running (best 91 at round 15) |
| 1 | 92 | 95 at 3.6 h | **95 in 73 s** | running (91 at round 11) |
| 2 | 95 in 20 s | 95 in 47 s | 90 (stuck from round 9) | **95 in 84 s** |
| 3 | 95 in 126 s | 40 | **95 in 259 s** | **95 in 77 s** |
| 4 | 36 | 95 in 389 s | **95 in 117 s** | running (91 at round 11) |
| 5 | 30 | 95 in 96 s | 91 (stuck from round 11) | **95 in 44 s** |

H29 alone: 4 of 6 seeds at 95 within 5 min, the other two frozen at the near-general policy — anchoring keeps
the best policy preferred but cannot help on the added problem it fails on. blocks4ops flat under H29: 39/95
(best 34), no better than rec3 (36). H30 so far: 3 of 3 finished seeds at 95 within 90 s, including the two
H29 stragglers.
H30 seeds 0/1/4 (running) stall differently: after the prefix plan for the 10–11-block problem is added, the
first anchored solve is OPTIMAL with cost `[6, 0]` — six frontier transitions are *required* (no zero-frontier
model exists on the enlarged instance) — and every later solve of that round family hits the 300 s budget
non-optimally (`[7,42]`, `[6,31]`, …, timed out). Two things to check (diagnostic agent running): why the
plan-restricted space of the new problem has alive states without an on-plan good successor although a
spliced root-anchored plan was added, and whether the budget rather than expressivity now binds at 11 blocks.

**H30 stall diagnosed (seed 1 log, agent report):** the prefix plan is fine (root-anchored, goal-reaching,
accepted, no dead ends, all frontier states in the new problem). The first lazy-pairs iteration grounds no
separation facts, so its proven-optimal frontier count is a *lower bound* for the whole round: `[6,0]` at
complexity 4, `[5,0]` after expansion, `[1,0]` at complexity 5, `[0,0]` at complexity 6 — a zero-frontier
policy for the 11-block instance provably needs complexity 6 (the same climb happened at 4 blocks in rounds
5–11). Nothing reads that proof: each of rounds 12–14 then spends 9–10 further 300 s solves refining the
separation layer of a model that cannot be a policy (≈105 of 121 min), and round 15 (complexity 6) finds the
optimum `[0,17]` at 295 s but the lazy loop keeps running 300 s timeouts whose incumbents regress
(`[43,70]`…) until the wall budget ends. Anchors never fall back because a frontier-bearing SAT model is
neither UNSAT nor UNKNOWN. Also: frontier-expansion plans are silently dropped once `max_plans_per_problem`
is hit. H31 (in progress): (1) stop the lazy loop as soon as a proven-optimal relaxation has a non-zero
frontier count and return it; (2) widen the anchor fallback to that case; (3) a round-level solve budget with
warm-started bounds so timed-out iterations cannot regress; (4) log/skip capped frontier plans.

### H30 result (full loop, seeds 0–5, 4 h graceful) and H31 launch

| seed | H29 | H30 (+ prefix plans) |
|---|---|---|
| 0 | 95 in 67 s | 91 (best 91 at round 15; stalled on 010-5 as diagnosed) |
| 1 | 95 in 73 s | 91 (stalled on 011-5) |
| 2 | 90 | **95 in 84 s** |
| 3 | 95 in 259 s | **95 in 77 s** |
| 4 | 95 in 117 s | 93 (best 91 at round 11) |
| 5 | 91 | **95 in 44 s** |

H30 fixes the two H29 stragglers and breaks three seeds H29 had: the prefix plan makes the 10–11-block
problem's instance large enough that complexity 4–5 provably needs frontier transitions and the round budget
is burnt before complexity 6 (the diagnosis above). blocks4ops flat under H30: 41/95. Union over H29/H30:
every seed reaches 95 under one of the two, none of the failures is an expressivity limit.

H31 (`hyp/frontier-bound`, 19e2452, on `hyp/prefix-plans`; 309 tests): the lazy-pairs loop stops as soon as
a proven-optimal iteration has a non-zero frontier count (a valid lower bound at every iteration, since each
grounded program is a relaxation of the full round) and returns FRONTIER; the anchor fallback also triggers
on that proof; `lazy_pairs_warm_start` passes the previous incumbent as clingo's initial bound (a search
heuristic, dropped on UNSAT — a grounded bound would be unsound because added pairs raise the optimum);
optional `round_time_limit`; capped frontier plans are logged and their planner calls skipped
(`frontierPlansDropped`). Launched: full loop seeds 0–5 (`fb-s*`, cluster worktree `fbound`, H30 setting +
H31 defaults) and blocks4ops flat (`fb-b4flat`).
H31 first seeds: seed 2 95/95 in 116 s, seed 5 95/95 in 98 s; the lower-bound abort fires as designed in the
running seeds (bounds 1–7 proven after lazy iteration 1). Seeds 0/1/3/4 and blocks4ops flat running.
H31 seeds 3 and 4: **95/95 in 246 s and 495 s** (both 95 under H29, 95/93 under H30). Four of four finished
seeds at 95 within 9 min. Seeds 0 (best 83 at round 13) and 1 (best 28 at round 2) running.

### H31 result (full loop, seeds 0–5, 4 h graceful) and the picture across H27–H31

| seed | rec3 | H27 | H29 | H30 | H31 |
|---|---|---|---|---|---|
| 0 | 95 (157 s) | 93 | **95 (67 s)** | 91 | 83 (best at round 13; 34 fast rounds) |
| 1 | 92 | 95 (3.6 h) | **95 (73 s)** | 91 | 34 (bad trajectory from round 2; 41 rounds) |
| 2 | 95 (20 s) | 95 (47 s) | 90 | **95 (84 s)** | **95 (116 s)** |
| 3 | 95 (126 s) | 40 | 95 (259 s) | **95 (77 s)** | **95 (246 s)** |
| 4 | 36 | 95 (389 s) | **95 (117 s)** | 93 | **95 (495 s)** |
| 5 | 30 | 95 (96 s) | 91 | **95 (44 s)** | **95 (98 s)** |

H31 does what it was built for: the two failing seeds no longer burn hours in doomed rounds (34 and 41
rounds, every solve under 5 s, 25–43 lower-bound aborts), they simply converge to patchworks of cost 13–16
on a 7–8-problem training set after an unlucky early sample (seed 1's best is 34/95 from a cost-4 policy
in round 2). blocks4ops flat: 41/95. Every seed reaches 95/95 under at least one of H29/H30/H31, and every
remaining failure is an early-sample trajectory, not a budget or an expressivity limit any more. Since rounds
are now seconds, the natural next step is in-loop diversity on stall rather than a bigger budget: H32 (in
progress) resamples the training problems' example plans with a fresh planner seed when the best coverage
has not improved for N rounds (the in-loop form of a seed portfolio), keeping the policy-conformant and
prefix plans. rec5 = the twelve cross-domain suites under the H31 stack submitted.
rec5 (H31 stack) first results: blocks4ops **113/113 in 82 s** (rec3 3.8 min, rec4 3.9 min), blocks 35/35 in
64 s. Ten rec5 jobs running.

## H32: resample example plans on stall (`hyp/resample-on-stall`, e1584fb, on `hyp/frontier-bound`; 323 tests)

`resample_on_stall: true`, `stall_rounds: 6`, `resample_max: 3`: when the best coverage has not improved
for six rounds, every training problem's planner plans are replaced (policy-conformant and prefix plans are
kept) by continuing the problem's lazy SIW stream or, when exhausted, a fresh stream with
`planners.siw.seed += k·1000003` and `restarts ≥ 2`; refutations are invalidated, `max_cost` and the training
set stay. Finding on the way (agent): SIW's diversity RNG is `random.Random(f"{siw.seed}:{restart}")`, and
restart 1 is the identity permutation — so with `restarts: 1` **all six seeds drew the same SIW plans**;
`--seed` only varies policy execution (rule order, bindings, successor draws), hence which problems count as
solved, which are added and in which order, and which H27/H30 plans the instance carries. The seed
dependence is therefore in the execution/addition order, one step removed from the plan sample; the resample
still acts at the right level (the plan set). Launched: full loop seeds 0–5 (`rs-s*`, cluster worktree
`resample`, H31 stack + H32) and blocks4ops flat (`rs-b4flat`).
H32 first seeds: seed 0 **95/95 in 110 s**, seed 1 **95/95 in 71 s** — the two seeds H31 left at 83 and 34
(seed 0: 1 resample(s), best at round 12
seed 1: 1 resample(s), best at round 12). Seed 2 at best 39 after 16+ rounds, seed 3 resampling; running.
H32 seed 3: 95/95 in 254 s (one resample). Seed 2 sits at 39 after two resamples, seed 4 at 84 after one;
seed 5 and blocks4ops flat running.
rec5 logistics_dp: **13/47, stopped by max complexity after 83 min** (rec4 23/47 at the 3.7 h limit, rec3
18). With H31 every doomed round ends in seconds, so the complexity sweep on the 6-problem training set
reaches `max_complexity` within the hour and the run terminates instead of spending its remaining budget —
the abort exposes that "max complexity reached" is treated as the end of the run rather than as a stall.
H32's resample-on-stall is the intended answer; logistics_dp resubmitted under the H32 stack (`rs-logdp`).
H32 seed 5: 95/95 in 103 s (one resample). Seeds 2 and 4 have used all three resamples and sit at 39 and
84; blocks4ops flat best 81 at round 15 (one resample); running.
H32 seed 2 trace: this seed never sees a near-general policy (best 27 at round 6, 39 at round 16) and keeps
adding 2–5-block problems, so by the second resample the training set has 13 problems and 57 planner plans
are replaced at once — a resample of a large patchwork set does not recover. The resample must come earlier
and/or reset the sweep: arms `rs2-*` = `stall_rounds: 3, resample_max: 6` and `rsc-*` = `resample_reset_complexity: true`
submitted for seeds 2 and 4.
rec5 grid: 1/26, stopped by max complexity after 1.9 h (rec3 1/26 at 2.9 h) — unchanged; grid remains open.
logistics_dp under the H32 stack: 15/47, again stopped by max complexity (84 min, 28 rounds, all three
resamples used). The termination itself is the defect: `max_complexity` ends the run although 2.6 h of
budget remain, whereas the slower stacks (rec4: 23/47) simply never got there. H33 (in progress): reaching
max complexity with unsolved problems and budget left is treated as a stall — resample if H32 still may,
otherwise add the next unsolved problem and restart the sweep at `min_complexity` with all accumulated
plans — so the run ends only by budget or full coverage.

## H33: continue past max complexity (`hyp/no-maxc-stop`, e9bbdac, on `hyp/resample-on-stall`; 335 tests)

`continue_after_max_complexity: true`: when the sweep is exhausted with unsolved problems and budget left,
the iterator first resamples (if H32 still may), else adds the next unsolved problem, and restarts the sweep
(`sweep_target` raised so it climbs). Note from the agent: for a *cleanly refuted* sweep the add-problem
restart is provably futile (monotonicity), so it only pays where the sweep ended by timeout / resources /
restricted rounds — the common case on large instances — which is why the resample goes first. Launched
under the H33 stack: logistics_dp, logistics, grid (`nm-*`, cluster worktree `nomaxc`, 4 h graceful).
rec5 miconic **22/25** (rec4 21, rec3 14, rec2 19), delivery **172/225** (rec3 171, rec4 162) — both the best
figures so far under the H31 stack.
rec5 logistics **22/47** at the 3.75 h limit (rec4 19, rec3 15, rec 9) — best so far; this run did not hit
max complexity within the budget.

### rec5 complete (H31 stack, 4 h graceful, seed 0) — cross-domain table

| domain (problems) | rec (12 h) | rec2 | rec3 | rec4 (H27) | rec5 (H31 stack) |
|---|---|---|---|---|---|
| blocks4ops (113) | 106 | 108 | 113 (3.8 min) | 113 (3.9 min) | **113 (82 s)** |
| blocks (35) | 29 | 27 | 35 (23 s) | 35 (85 s) | 35 (64 s) |
| delivery (225) | 139 | 140 | 171 | 162 | **172** |
| miconic (25) | 18 | 19 | 14 | 21 | **22** |
| logistics (47) | 9 | 9 | 15 | 19 | **22** |
| logistics_dp (47) | 17 | 9 | 18 | 23 | 13 (max complexity at 83 min; H33 running) |
| reward (35) | 5 | 16 | 15 | 20 | 15 |
| spanner (162) | – | 130 | 137 | 129 | 130 |
| barman (30) | – | 0 | 0 | 0 | 0 |
| grid (26) | – | – | 1 | 1 | 1 (max complexity at 1.9 h; H33 running) |
| storage (30) | – | – | – | 3 | 3 |
| sokoban | – | – | – | – | – (no SIW plan; H25 still open) |

H32 seed 2 final: 40/95 (three resamples). Seed 4 and the follow-up arms running.

### H32 result (full loop, seeds 0–5, 4 h graceful)

| seed | H31 | H32 (+ resample on stall) | H32 early (`stall_rounds 3`, 6 resamples) | H32 reset (`resample_reset_complexity`) |
|---|---|---|---|---|
| 0 | 83 | **95 in 110 s** (1 resample) | | |
| 1 | 34 | **95 in 71 s** (1 resample) | | |
| 2 | 95 | 40 (3 resamples; never near-general) | best 88 at round 13, then drift (running) | best 88 at round 14, then drift (running) |
| 3 | 95 | **95 in 254 s** (1 resample) | | |
| 4 | 95 | 84 (3 resamples) | best 93 at round 28, then drift (running) | best 93 at round 29, then drift (running) |
| 5 | 95 | **95 in 103 s** (1 resample) | | |

Resampling fixes the two H31 failures and breaks two other seeds; the follow-up arms show why: they reach
a near-general policy (88/93) and then the resample fires *during* the legitimate complexity climb on the
enlarged training set (H30 diagnosis: the 10–11-block instance needs complexity 6, and with H31 those
doomed rounds are seconds long), replacing the plans the climb depends on. The stall counter must not count
sweep progress as a stall. H32b (in progress): count only rounds that produced a policy without improving
the best, never lower-bound-abort / NO_SOLUTION rounds of an ongoing sweep, and never resample while the
sweep on the current training set is still climbing. blocks4ops flat under H32: best 81 (running).
