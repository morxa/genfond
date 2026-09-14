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
`lazy_pairs: false` the instance is byte-for-byte the one `hyp/dist-sets` emitted and the solve is
the single `Control.solve` it did.

## Ground program, blocks3ops one-shot

`--one-shot --max-complexity 4 -n 1 --seed 0 --max-memory 5000`, one problem per run.

| instance | mode | atoms | rules | pairs grounded | `dist` facts | iter | clingo solve | round wall | peak mem | outcome |
|---|---|---|---|---|---|---|---|---|---|---|
| p004-1 | eager | 124,633 | 233,499 | 2,128 / 2,128 | 105,685 | 1 | 1.30 s | 2.22 s | 108 MB | solved, cost 2 |
| p004-1 | lazy | 61,102 | 106,070 | 789 / 2,128 | 43,630 | 6 | 1.15 s | 2.27 s | 87 MB | solved, cost 2 |
| p005-1 | eager | 831,756 | 1,551,082 | 17,083 / 17,083 | 689,082 | 1 | 5.84 s | 11.27 s | 380 MB | solved, cost 2 |
| p005-1 | lazy | 337,167 | 563,262 | 5,000 / 17,083 | 212,917 | 2 | 4.46 s | 11.36 s | 201 MB | solved, cost 2 |
| p006-1 | eager | — | — | — / 1,150,646 | (55.0M) | — | — | 103 s | 3842 MB | **MemoryError, 0/1** |
| p006-1 | lazy | 1,580,273 | 1,949,104 | 17,056 / 1,150,646 | 249,659 | 7 | 141 s | 243 s | 605 MB | **solved, cost 4** |
| p007-1 | lazy | — | — | 0 / — | 0 | 0 | — | 2013 s | 3111 MB | **`bad_alloc`, 0/1** |

Atoms and rules are cumulative over all ground calls. "clingo solve" is the single solve for the
eager runs (CPU; at `-n 1` that is wall) and the sum of the per-iteration solve walls for the lazy
ones; "round wall" is the whole round, i.e. state space + feature evaluation + instance +
grounding + solving.

The headline is p006-1: 1.5% of the pairs and 0.45% of the `dist` facts the eager encoding would
need, and the complexity-4 ceiling for blocks3ops moves from 5 blocks to 6. Note where its 243 s
go — 95 s building the state space and evaluating features, 141 s in clingo, and ~7 s for all
seven counterexample scans (up to 460k violated pairs each), their fact text and the incremental
grounding. Past 5 blocks the separation layer is no longer what bounds the domain.

p007-1 makes that concrete: 37,633 states, 20 minutes to expand them, 13 more to evaluate the
features, and then `bad_alloc` on the *first* grounding — the graph layer and the `eval`/`c_eval`/
`r_eval` facts alone exceed the cap. The lazy loop never ran an iteration; there was nothing for
it to do.

### Batch size, p006-1

| `lazy_pairs_batch` | iterations | pairs grounded | `dist` facts | clingo solve | round wall | peak mem |
|---|---|---|---|---|---|---|
| 500 | 20 | 8,985 | 104,561 | 728 s | 827 s | 590 MB |
| 5,000 (default) | 7 | 17,056 | 249,659 | 141 s | 243 s | 605 MB |
| 50,000 | 9 | 51,439 | 1,064,801 | 152 s | 240 s | 992 MB |

All three end at cost 4. A small batch is the expensive mistake: 20 rounds of re-solving a program
that is still missing the constraint that matters spend 728 s in clingo instead of 141 s, 3.4x the
wall clock, to save 60% of the grounded facts. A large batch buys nothing over the default in time
and pays 4x the facts and 1.6x the memory. The interesting column is "pairs grounded": even the
most generous setting touches 4.5% of the 1.15M pairs.

`lazyPairIterations`, `lazyPairsGrounded`, `lazyPairsTotal` and `lazyDistFacts` are recorded in the
stats file.

## End to end

`--type datalog-sig -n 1 --seed 0 --max-memory 5000`, `timeout 40m`.

| suite | mode | solved | cost | wall | peak mem | training problems | max complexity |
|---|---|---|---|---|---|---|---|
| `blocks3ops-local` (10) | eager | 10/10 | 9 | 18.6 s | 185 MB | 5 | 3 |
| `blocks3ops-local` (10) | lazy | 10/10 | 5 | 8.8 s | 68 MB | 4 | 2 |
| `gripper-local` (5) | eager | 5/5 | 6 | 2.7 s | 64 MB | 2 | 4 |
| `gripper-local` (5) | lazy | 5/5 | 6 | 3.3 s | 64 MB | 2 | 4 |
| `blocks4ops-clear-local` (4) | eager | 4/4 | 2 | 0.5 s | 62 MB | 2 | 2 |
| `blocks4ops-clear-local` (4) | lazy | 4/4 | 2 | 0.6 s | 62 MB | 2 | 2 |
| blocks3ops p002-1/p003-1/p004-1 | eager | 3/3 | 4 | 0.6 s | 61 MB | 2 | 2 |
| blocks3ops p002-1/p003-1/p004-1 | lazy | 3/3 | 4 | 0.5 s | 61 MB | 2 | 2 |

The three small suites are the requested A/B: identical coverage and identical cost, so the flag
changes nothing where the eager encoding was never under pressure.

`blocks3ops-local` diverges, and in the lazy run's favour: it finds a complexity-2 policy of cost
5 from four training problems where the eager run escalates to complexity 3, a fifth training
problem and cost 9. Both policies are re-validated by `execute_policy` on all ten problems, so
both are correct. This is *not* evidence that lazy solving is better: the two runs solve different
programs, an equally optimal model in an early round differs, and from there the escalation paths
diverge. It is the same model non-determinism the dist-sets change already showed, amplified by
the iterative loop. One suite is one sample.

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
  `np.bitwise_count` over the candidate pairs. At p006-1 all seven scans, the fact text and the
  incremental grounding together cost ~7 s of a 243 s round.
* **The bottleneck has moved again.** At 6 blocks the run is dominated by state-space expansion and
  feature evaluation (95 s of 243 s), and at 7 blocks it dies there, before the first solve.
  Attacking the separation layer further buys nothing on this domain.
* **Nothing shrinks.** Grounded batches are never retired, and `LazyPairs` is rebuilt per round, so
  in the iterative solver each round re-discovers its pairs from scratch.

## What I would try next

1. **Warm-start the loop.** Iteration 1 always solves a selection-free relaxation and always
   produces a useless cost-0 model. Seeding the first batch with the globally smallest-|D| pairs
   (computable without any model) would skip that round, at the price of grounding pairs that may
   turn out not to be good/bad pairs.
2. **Adapt the batch.** The sweep above says 5000 is a reasonable default but not that it is right
   for every instance: batch 500 wastes rounds while the cost is still climbing from 0, and by the
   time the cost is stable a large batch is pure waste. Growing the batch while the cost still
   moves and shrinking it once it settles would plausibly beat both ends.
3. **Attack the state space**, not the separation layer — that is what now bounds blocks3ops. The
   feature-vector and symmetry quotients recorded in the memory notes are the candidates.
4. **Drop pairs again.** Nothing ever removes a grounded batch. A pair separated by a cheap element
   the final model keeps could be retired between outer iterations, which matters more for the
   iterative solver (many rounds on one growing problem set) than for one-shot.
