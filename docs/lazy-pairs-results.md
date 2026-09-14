# Counterexample-guided (lazy) separation pairs

## Mechanism

`--type datalog-sig` states the separation constraint as a hitting set per unordered pair of
signature classes with the same action name, and `hyp/dist-sets` made `Sum |D|` over those pairs
the dominant term of the instance (689k `dist/2` facts at blocks3ops p005-1, 55M at p006-1, where
the run dies building the instance text). Almost none of those pairs constrain the optimum.

They are now added on demand, in the classic lazy-constraint loop (`genfond/lazy_pairs.py`):

1. Ground the base program and the instance *without* any `sig_pair`/`dist` facts and solve to
   optimality. That is a relaxation: every model of the full problem is a model of it.
2. Read the selection and the good/bad labelling off the model and ask `LazyPairs` which pairs it
   violates. **None means the relaxed optimum is feasible for the full problem, hence optimal for
   it as well** — the loop is done and policy extraction runs on that model, unchanged.
3. Otherwise ground one more batch of violated pairs into the same `clingo.Control` and re-solve.

The counterexample scan (`action_signatures.LazyPairs.violated_pairs`) does *not* enumerate pairs:
a pair is violated iff it is a good/bad pair that no selected element separates, and "no selected
element separates it" is a masked equality of the packed bit vectors. Bucketing the classes of one
action name by their *masked* vector yields the violated pairs directly as the good x bad cross
product inside each bucket. Each iteration adds at most `lazy_pairs_batch` of them, smallest |D|
first: small sets are the hardest to hit and constrain the selection most, large ones get
separated for free by whatever the solver picks next.

The facts carry the batch index (`sig_pair(k,K1,K2,D)`, `dist(k,D,E)`) and the hitting-set rules
live in `#program pairs(k)` in `solve_datalog_sig.lp`, so grounding a new batch grounds the rules
against that batch only instead of re-grounding them over everything added before. A `dist/3` fact
is emitted only in the batch that first needs its set; `sig_hit/1` is a plain atom, so the rule
deriving it stays valid for later batches that reuse the id.

Termination: each iteration adds at least one pair the previous model violated, and the pair set
is finite. An UNSAT relaxation proves the full problem UNSAT. The loop wraps the solve of a single
round, so `max_cost` / `limit_feature_cost` / `min_feature_complexity` / `limit_prune_cost` and the
whole frontier machinery are untouched.

Config: `lazy_pairs` (true in `default_datalog-sig.yaml`, false in `default.yaml`) and
`lazy_pairs_batch` (5000). `--lazy-pairs/--no-lazy-pairs` make the A/B one flag; with
`lazy_pairs: false` the instance and the solve are byte-for-byte the previous behaviour.

## Ground program, blocks3ops one-shot

`--one-shot --max-complexity 4 -n 1 --seed 0 --max-memory 5000`, one problem per run.

| instance | mode | atoms | rules | pairs grounded | `dist` facts | iter | clingo CPU | wall | peak mem | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| p004-1 | eager | 124,633 | 233,499 | 2,128 / 2,128 | 105,685 | 1 | 1.30 s | 2.6 s | 108 MB | solved, cost 2 |
| p004-1 | lazy | 61,102 | 106,070 | 789 / 2,128 | 43,630 | 6 | 0.26 s | 2.7 s | 87 MB | solved, cost 2 |
| p005-1 | eager | 831,756 | 1,551,082 | 17,083 / 17,083 | 689,082 | 1 | 5.84 s | 11.7 s | 380 MB | solved, cost 2 |
| p005-1 | lazy | 337,167 | 563,262 | 5,000 / 17,083 | 212,917 | 2 | 4.49 s | 11.8 s | 201 MB | solved, cost 2 |
| p006-1 | eager | — | — | — | (55.0M) | — | — | 104 s | 3842 MB | **MemoryError, 0/1** |
| p006-1 | lazy | 1,580,273 | 1,949,104 | 17,056 / 1,150,646 | 249,659 | 7 | 7.71 s | 244 s | 605 MB | **solved, cost 4** |
| p007-1 | lazy | PLACEHOLDER | | | | | | | | |

The headline is p006-1: 1.5% of the pairs and 0.45% of the `dist` facts the eager encoding would
need, and the complexity-4 ceiling for blocks3ops moves from 5 blocks to 6. Note where the 244 s
go — 7.7 s of clingo and ~230 s of state-space expansion and feature evaluation. Past 5 blocks the
separation layer is no longer the bottleneck at all.

`lazyPairIterations`, `lazyPairsGrounded`, `lazyPairsTotal` and `lazyDistFacts` are recorded in the
stats file.

## End to end

`--type datalog-sig -n 1 --seed 0 --max-memory 5000`, `timeout 40m`.

| suite | mode | solved | cost | wall | peak mem |
|---|---|---|---|---|---|
| `domains/suites/blocks3ops-local` (10) | eager | PLACEHOLDER | | | |
| `domains/suites/blocks3ops-local` (10) | lazy | PLACEHOLDER | | | |

## Equivalence

* `tests/test_lazy_pairs.py` pins the loop against the eager encoding. The interesting one is
  `test_the_lazy_selection_is_feasible_for_the_eagerly_grounded_program`: stopping when *its own*
  scan finds no violated pair makes any test against that scan circular, so the selection the loop
  returns is instead forced onto the eagerly grounded program, which carries every pair as an ASP
  constraint — it must stay satisfiable and at the same cost. Checked on gripper and blocks-clear.
* A hand-built four-class instance forces several iterations at `batch_size=1` and must end at the
  eager cost while grounding strictly fewer than all pairs.
* `test_violated_pairs_agree_with_the_eager_relation` pins the masked-equality scan against the
  distinguishing sets for five selections.
* `tests/test_datalog_signatures.py` runs its cost equivalence against the plain `datalog`
  encoding for both modes.

## Caveats

* **Same cost, not always the same model.** The lazy loop solves a different (growing) program, so
  it may return a different equally-optimal model; on the `gripper` fixture it does, which is why
  `test_both_encodings_find_the_same_policy` stays on the eager mode. The dist-sets write-up
  reports the same effect for its own change. Optimality itself is pinned by the feasibility test.
* **Batching is a guess, not a policy.** Smallest-|D|-first and a flat 5000 came out of the
  reasoning above, not out of a sweep; see below.
* **The Python scan is not free**, but it is cheap compared to what it replaces: bucketing plus a
  `np.bitwise_count` over the candidate pairs. At p006-1 the seven scans together are under 2 s of
  the 244 s run.
* **The bottleneck has moved again.** At 6 blocks the run is dominated by state-space expansion and
  feature evaluation, and at 7 blocks (p007-1) that is where it stops. Attacking the separation
  layer further buys nothing on this domain.

## What I would try next

1. **Warm-start the loop.** Iteration 1 always solves a selection-free relaxation and always
   produces a useless cost-0 model. Seeding the first batch with the globally smallest-|D| pairs
   (computable without any model) would skip that round, at the price of grounding pairs that may
   turn out not to be good/bad pairs.
2. **Tune the batch.** p005-1 needed two iterations at batch 5000 and p006-1 seven; a batch that
   adapts (grow while the cost is still climbing, shrink once it is stable) is the obvious knob.
3. **Attack the state space**, not the separation layer — that is what now bounds blocks3ops. The
   feature-vector and symmetry quotients recorded in the memory notes are the candidates.
4. **Drop pairs again.** Nothing ever removes a grounded batch. A pair separated by a cheap element
   the final model keeps could be retired between outer iterations, which matters more for the
   iterative solver (many rounds on one growing problem set) than for one-shot.
