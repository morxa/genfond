# H10: synthesise and deduplicate features over a richer state sample (`hyp/rich-sample`)

**Verdict: null result on blocks3ops, and the premise it was built on (H9) is wrong.**
The mechanism works and is cheap, but on the domains measured here it changes the pool only in
the degenerate case of a *single* training problem, and it changes nothing end to end.

## The idea

dlplan's `generate_features` deduplicates the concepts, roles and features it generates by their
denotation on the states it is handed, and genfond's own `prune_redundant_concepts/roles/features`
does the same over the training states. Both only ever see the (plan-restricted) states of the
current training problems, which are few and small. The H9 entry in `docs/experiments-log.md`
concluded from this that a *general* element coincides on those states with a shallower accidental
one and is discarded before the solver ever sees it, and named
`c_equal(r_primitive(on,0,1),r_primitive(on_G,0,1))` -- "a block sits on its goal support",
complexity 3, the concept the hand-crafted preset policy of H9 used to reach 5/12 held-out -- as
the example.

`feature_sample` adds random-walk states from every problem on the command line, including the
large ones that are not in the training set, to the state set used *for synthesis and
deduplication only*. The ASP instance is untouched: sample states never become `state/2` facts, so
the grounded program does not grow with the sample.

## What was implemented

- `state_space_generator.random_walk_states(domain, problem, walks, length, rng)`: grounds the
  actions once and follows `walks` independent trajectories of at most `length` steps, picking a
  random applicable action and (for nondeterministic actions) a random outcome. It never
  enumerates a state space, so a 30-object instance costs the same as a 3-object one.
- `FeaturePool` builds its own `InstanceInfo` for problems outside the training set, numbered
  above the training instance ids and kept out of `self.instances` (which is what
  `evaluate_*_from_problem`, and therefore the ASP emission, resolves problem names through).
  A state carrying facts the instance cannot map (numeric fluents) is dropped and counted.
- The sample states are appended to the state list passed to `dlplan_gen.generate_features` and to
  the state list `compute_redundant_features/concepts/roles` compare denotations over. The
  concept/role versions now compare raw dlplan object indices over a fixed state list instead of
  object names over a per-(problem, state) dict -- the same equivalence relation, minus the
  quadratic `obj_id_to_obj` scan.
- `compute_uninformative_*` deliberately stays on the training states. An element that
  distinguishes no action argument on any training state cannot appear in any rule the *current*
  ASP instance can express, and the `#minimize` over complexity would never select it even if it
  were emitted, so keeping it would be cost without effect.
- Config block `feature_sample` (`enabled`, `walks_per_problem`, `walk_length`, `problems:
  all|active`, `seed`) plus `--feature-sample` / `--no-feature-sample`. The CLI flag uses
  `dest="feature_sample_enabled"` and is applied by hand in `__main__`, because `ConfigHandler`
  would otherwise merge the bare bool over the whole config block.
- Walks are cached per (domain, problem, walks, length, seed) at module level: `FeaturePool` is
  rebuilt every round, and re-grounding 95 problems per round would dominate the run.

## Finding 1: `c_equal` is never generated, and never was deduplicated away

Training on `blocks3ops` p003-1 + p004-1 (full state spaces, 86 states), sampling 9403 states from
all 95 instances:

| complexity | generators | concepts w/o sample | with sample | `c_equal(on, on_G)` |
|---|---|---|---|---|
| 3 | restricted (`datalog-sig`) | 29 | 29 | absent both |
| 4 | restricted | 88 | 88 | absent both |
| 5 | restricted | 253 | 257 | absent both |
| 3 | unrestricted | 50 | 50 | absent both |
| 4 | unrestricted | 158 | 163 | absent both |
| 5 | unrestricted | 744 | 783 | absent both |

Evaluating the parsed `c_equal(r_primitive(on,0,1),r_primitive(on_G,0,1))` directly over the 86
training states shows it has the same denotation as **no** generated concept -- not over the
training states, not over the 9489 training + sample states. It is therefore not a deduplication
victim at all. A minimal check outside genfond settles it: `dlplan.generator.generate_features` on
a synthetic vocabulary with two binary predicates, 40 random states and all complexity limits at 5
(`generate_equal_concept` defaults to `True`) returns 26 concepts and **no** `c_equal` either.
dlplan's EqualConcept rule does not fire in this build.

So the H9 explanation of the 0/12 held-out result -- "the general representative is discarded
before the solver ever sees it" -- is wrong for the concept it named. The preset policy's key
concept is outside the synthesised pool because the *generator* never produces it, not because the
deduplication removes it. Making the pool available would need the preset/parse path (which
already works) or a fix in dlplan, not a richer sample.

## Finding 2: the sample only matters with a single training problem

The sample earns its keep exactly where the training set cannot vary something the sample can.
With one training problem, every `*_G` (goal) concept is static, so the goal structure is
indistinguishable from an accidental coincidence. Training on p003-1 alone, sampling p003-1 and
p005-2 (80 sample states):

| complexity | concepts w/o | with | roles w/o | with |
|---|---|---|---|---|
| 3 | 26 | **29** | 28 | 28 |
| 4 | 80 | **88** | 70 | **73** |

The three extra complexity-3 concepts are `c_some(r_primitive(on_G,0,1),c_top)`,
`c_all(r_primitive(on_G,0,1),c_bot)` and `c_all(r_primitive(on_G,0,1),c_primitive(ontable_G,0))` --
all of them goal concepts, all of them collapsed by p003-1's single goal. This case is the one the
focused tests in `tests/test_feature_generator.py` pin down.

Add a second training problem with a different goal and the effect is gone (the table in Finding 1:
identical pools up to complexity 4). In the iterative loop the training set reaches two problems
within a couple of rounds, so the sample is a no-op from then on.

`prune_redundant_concepts` / `prune_redundant_roles` found **zero** redundant elements in every
configuration measured here, with or without the sample: dlplan's own deduplication has already
removed them. Item 3 of the hypothesis is therefore correct but inert on these domains.

## Finding 3: no end-to-end effect -- and the suite is hash-seed sensitive

`domains/suites/blocks3ops-local` (10 problems), `--type datalog-sig -n 1 --seed 0 --max-memory
6000 --add-problem-after-success`, held-out evaluated with
`scripts/eval_policy.py --seed 0 -i 3` on the 12 instances of
`domains/deterministic/blocks3ops-heldout`:

| `PYTHONHASHSEED` | arm | solved | wall | cost | rules | complexity | held-out |
|---|---|---|---|---|---|---|---|
| 0 | off | 10/10 | 5.8 s | 5 | 30 | 2 | 1/12 |
| 0 | **on** | 10/10 | 13.3 s | 9 | 40 | 3 | 0/12 |
| 1 | off | 10/10 | 12.5 s | 9 | 40 | 3 | 0/12 |
| 1 | **on** | 10/10 | 6.1 s | 5 | 30 | 2 | 1/12 |
| 2 | off | 10/10 | 5.9 s | 5 | 30 | 2 | 1/12 |
| 2 | **on** | 10/10 | 6.2 s | 5 | 30 | 2 | 1/12 |

Both arms produce exactly two outcomes -- (cost 5, complexity 2, 30 rules, 1/12) and (cost 9,
complexity 3, 40 rules, 0/12) -- in the same 2:1 proportion. The first pair looks like a clear
regression from the sample and is not: flipping `PYTHONHASHSEED` flips it. `State` is a
`frozenset` of `pddl` atoms whose hashes are string hashes, so set and dict iteration order --
and with it the order the frontier loop picks plans and training problems in -- varies per
process. **`--seed 0` and `-n 1` are not enough to make two runs of this suite comparable;
`PYTHONHASHSEED` has to be fixed too.** That affects every A/B in `docs/experiments-log.md` that
was decided on a single local run.

Regression suites (same flags, sample on vs off): `gripper-local` 5/5 cost 6, `miconic-local` 4/4
cost 12, `blocks4ops-clear-local` 4/4 cost 2, `delivery-local` 4/4 cost 9 -- identical in both
arms, and the logged pool sizes are identical round for round (e.g. gripper 4/8 features, 30/32
concepts, 30/30 roles with 49 sample states and with none).

## Cost

| | |
|---|---|
| walks, 95 blocks3ops problems, 5 x 30 steps | 9403 states, **45.3 s** once (grounding dominates), then cached |
| walks, 10-problem local suite | 289 states, < 0.1 s |
| redundancy pruning per round, 9403 sample states | 1.0 s (cx 3) / 3.4 s (cx 4) / 11.4 s (cx 5), against < 0.05 s without |
| `blocks3ops-local` peak RSS | 67 MB -> 113 MB |
| ASP instance | **unchanged** (972 lines both ways on the p003-1 + p004-1 one-shot) |

The instance size is the point worth keeping: the sample buys whatever pool breadth it buys
without touching the grounded program, so it does not interact with the grounding ceiling that
bounds blocks3ops.

## Caveats

- Walk states are sampled from the *initial* state forward. States near a goal of a large
  instance are rarely reached, so the sample is biased towards the early part of the state space.
  A goal-regression sample, or walks from the states an example plan visits, would be a different
  experiment.
- `problems: all` re-grounds every problem on the command line once. The cache makes that a
  one-off, but a suite of very large instances would pay it up front.
- With `max_complexity` above 4 the sample does grow the pool (Finding 1, complexity 5), so on a
  domain where the loop actually climbs that high the effect may not be zero; blocks3ops never
  gets there before the grounding ceiling.
