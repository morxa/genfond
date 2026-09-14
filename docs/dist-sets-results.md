# Precomputed distinguishing sets for the datalog separation layer

## Approach

`--type datalog-sig` already quotients the separation constraint onto action signature classes,
but it still derived *which* elements distinguish a pair of classes inside ASP: `sig_c_dist/3`,
`sig_r_dist/3` and `sig_f_dist/3` paired every two classes with the same action name with every
concept, role and feature, and the `#show` rules that fed the policy generator did it a second
time. That is `|K|^2 * (|F| + |C|*a + |R|*a^2)` ground rules.

Which elements distinguish a pair is a property of the instance, not of the model, so it is now
computed in Python (`genfond/action_signatures.py`) with packed bit vectors: one row per class
holding its boolean feature vector, its concept memberships per argument position and its role
memberships per position pair. Per pair the three blocks are XOR-ed, the argument positions are
or-reduced away, and the resulting element bit vector is hashed to deduplicate. The instance
carries `sig_pair(K1, K2, D)` per unordered pair and `dist(D, f("...")|c("...")|r("..."))` per
element of each *distinct* set, and the ASP side is reduced to a hitting-set condition:

```
sig_hit(D) :- dist(D, c(C)), c_selected(C).          % and f/r
:- good_sig(K), bad_sig(K).                          % the K1 = K2 case
:- good_sig(K1), bad_sig(K2), sig_pair(K1,K2,D), not sig_hit(D).   % and the mirrored one
```

The rule conditions are reconstructed in Python too (`generate_datalog_policy`), from the
selected elements plus the good (`sig_action/2`) and bad (`bad_sig/1`) classes, mirroring the
layered `#show` rules exactly: a bad class already separated by a selected feature contributes
nothing, one separated by a selected concept contributes only concept conditions, the rest
contribute role conditions. `good_sig`/`bad_sig`, the graph layer, the minimize statements and
the `limit_*`/`min_feature_complexity` programs are untouched, so frontier expansion and the
cost bookkeeping behave exactly as before.

## Equivalence

The reconstruction was checked against the old `#show`-based one **on identical models**: the
new `FeaturePool` output plus a regenerated copy of the old `sig_*_eval` facts, solved with the
pre-change `solve_datalog_sig.lp`, then both generators run on the same `solution`.

| instance | legacy rules | new rules | identical |
|---|---|---|---|
| gripper problem-2-1 | 4 | 4 | yes |
| blocks4ops-clear p002-1 | 2 | 2 | yes |
| blocks3ops p003-1 | 9 | 9 | yes |
| blocks3ops p004-1 | 7 | 7 | yes |

`tests/test_action_signatures.py` additionally pins the emitted `sig_pair`/`dist` relation
against a naive transcription of the ASP rules it replaces, and the pre-existing
`tests/test_datalog_signatures.py` still pins the quotiented encoding against the unquotiented
`datalog` one (same policy, same cost on gripper and blocks4ops-clear).

## Ground program, blocks3ops, `--one-shot --max-complexity 4 -n 1 --seed 0 --max-memory 4000`

| instance | encoding | clingo atoms | clingo rules | clingo CPU | total wall | peak mem | outcome |
|---|---|---|---|---|---|---|---|
| p004-1 | old | 1,817,708 | 1,192,222 | 15.0 s | 15.9 s | 1316 MB | solved, cost 2 |
| p004-1 | new | 124,633 | 233,499 | 0.85 s | 1.8 s | 107 MB | solved, cost 2 |
| p005-1 | old | — | — | — | 30.4 s | 3552 MB | **`bad_alloc`, 0/1** |
| p005-1 | new | 831,756 | 1,551,082 | 7.5 s | 14.6 s | 377 MB | solved, cost 2 |
| p006-1 | old | — | — | — | — | >4 GB | memory |
| p006-1 | new | — | — | — | — | >4 GB | memory |

14.6x fewer atoms and 5.1x fewer rules on p004-1, and 4 blocks of solving time turns into 2.
p005-1 moves from *not solvable at a 4 GB cap* to solved in 15 s; the ceiling for blocks3ops at
complexity 4 therefore moves from 4 to 5 blocks. p006-1 still does not fit (see caveats).

The *instance* file grows while the *ground program* shrinks — that is the whole trade:

| p004-1 instance | old | new |
|---|---|---|
| lines total | 27,007 | 120,557 |
| separation-layer facts | 14,263 (`sig_aname`/`sig_pos`/`sig_c_eval`/`sig_r_eval`/`sig_bool_eval`) | 107,813 (`sig_pair` + `dist`) |

## End to end (iterative solver, `-n 1 --seed 0 --max-memory 4000`)

| suite | old solved | new solved | old cost | new cost | old wall | new wall |
|---|---|---|---|---|---|---|
| `domains/suites/gripper-local` (5) | 5/5 | 5/5 | 6 | 6 | 3.85 s | 3.32 s |
| `domains/suites/blocks4ops-clear-local` (4) | 4/4 | 4/4 | 2 | 2 | 0.47 s | 0.75 s |
| `blocks3ops` p002-1/p003-1/p004-1 (3) | 3/3 | 3/3 | 4 | 4 | 1.00 s | 0.92 s |

Coverage and cost are identical everywhere. These suites are small enough that the old encoding
never hit its wall, so the wall-clock differences are noise; the payoff is on the instances the
old encoding could not ground at all.

## Python-side cost

The pair scan is numpy over `uint64` bit vectors, one chunk per left-hand class, and costs
roughly 0.5M pairs/s: 17k pairs (p005-1) in 0.1 s, 1.15M pairs (p006-1) in 2.3 s. Memory for the
bit vectors is negligible (`|K| * (1 + a + a^2)` rows of a few words each). The dominant Python
cost is the *text* of the emitted facts, see caveats.

## Separation-layer sizes (new encoding)

| instance | classes | pairs | distinct sets | `dist/2` facts |
|---|---|---|---|---|
| blocks3ops p004-1 | 114 | 2,128 | 1,905 | 105,685 |
| blocks3ops p005-1 | 319 | 17,083 | 12,811 | 689,082 |
| blocks3ops p006-1 | 2,468 | 1,150,646 | 752,962 | 55,041,047 |

## Caveats

* **The bottleneck moved, it did not disappear.** `Sum |D|` over the distinct sets is now the
  dominant term, and at 6 blocks it is 55M facts: the run dies with a `MemoryError` inside
  `iter_dist_set_facts`, building the instance text, before clingo is ever called. Deduplication
  helps less than one would hope (2,128 pairs -> 1,905 distinct sets on p004-1) because almost
  every pair differs in a slightly different set of ~50 elements.
* **A measured but rejected refinement.** The three components of `D` (features, concepts, roles)
  are independent — a pair is separated iff the selection hits any one of them — so they can be
  deduplicated separately and emitted as `sig_pair(K1,K2,Df,Dc,Dr)` + `dist_f/dist_c/dist_r`.
  Measured: 689k -> 337k facts on p005-1 and 55.0M -> 12.9M on p006-1. It was implemented and
  then reverted, because p006-1 still exceeds 4 GB with it (it dies a little later, while
  handing the instance to clingo) and it perturbs which of the equally-optimal models clingo
  returns, breaking `test_both_encodings_find_the_same_policy[gripper]`. It is the obvious next
  step if the instance size is attacked again.
* **Same cost, not always the same model.** The constraint is equivalent, but clingo now searches
  a different program, so an end-to-end run may return a different optimal policy at the same
  cost (on blocks3ops p004-1 one-shot: 5 rules instead of 6, same selected concept and role).
  This is model non-determinism, not a semantic difference — with the *model* held fixed the two
  reconstructions agree exactly (table above).
* **numpy** is used but is not a direct dependency in `pyproject.toml`; it arrives transitively
  (`up-symk` -> `configspace`) in the `main` group, so `poetry install` always provides it.
* `r_selected/1` had to be added to the `#show` list — the old encoding leaked role selection
  only through the per-pair `sig_r_dist` shows.
