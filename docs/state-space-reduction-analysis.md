# Pruning objects from states: analysis, measurements, and a shelved plan

Date: 2026-08-05. Branch: `learn-from-examples`.

Two ideas were proposed for shrinking the ASP instance by discarding information about objects
in a state:

1. If an object occurs in no action applicable in a state, keep no information about it.
2. If two objects cannot be distinguished by any concept or role, keep only one of them.

**Summary of the outcome.** Idea 1 is already implemented in its only sound form; its stronger
reading is unsound. Idea 2 is unsound as stated but becomes sound when reformulated as
*symmetry reduction* — and measurement then shows it gives large reductions exactly on the easy
domains and **provably nothing** on the goal-pinned domains we are trying to improve. A second
candidate reduction (quotienting by feature vector) was measured and shows the same inverse
correlation. Both are documented here so the reasoning does not have to be redone.

## Why the ASP size is worth attacking

Both encodings are quadratic in the size of the state space. In `solve_constraints.lp`,
`trans_diff/7` ranges over *pairs* of transitions and `bool_dist/4` over pairs of states; in
`solve_datalog.lp` the final integrity constraint and `c_distinguished`/`r_distinguished` range
over pairs of (state, action). Grounding is therefore ~|T|²·|F| and ~|SA|²·|C|. A factor *k*
cut from |S| buys roughly *k*². For blocks3ops `p005-2` (5 blocks, |S| = 501, |T| = 2140) at
complexity 5, `trans_diff` alone grounds on the order of 2140² · 110 ≈ 5·10⁸ instances. This is
consistent with blocks3ops being grounding-bound rather than search-bound.

## Idea 1 — objects that occur in no applicable action

### Reading (a): do not emit object-indexed ASP facts for such objects

Sound, and **already implemented**. `feature_generator.py:575` computes `all_action_args`, and
`c_eval` (`:597-602`) and `r_eval` (`:623-628`) are emitted only for objects and pairs drawn
from it. It is sound precisely because `solve_datalog.lp` consults `c_eval`/`r_eval` *only* at
`aparam` positions, so a concept's extension elsewhere is unobservable to the solver. Nothing
left to do.

### Reading (b): delete those objects' atoms from the state, so states merge

**Unsound.** Gripper, goal "all balls in roomb", robot in rooma. In state s₁ all five balls are
in rooma; in s₂ balls 1–4 are already in roomb and only ball 5 is in rooma. `pick(b, roomb, g)`
is inapplicable because the robot is at rooma, so balls 1–4 occur in no applicable action of s₂
and would be deleted. The numerical feature *number of balls not at roomb* — exactly the
descending measure the policy needs — then takes the same value on s₁ and s₂.

Two independent defects:

1. Deleting an object changes every **cardinality** feature, and cardinality features are where
   the progress measures live. `trans_delta` and `safe_state` depend on them directly.
2. The same object is deleted in some states and kept in others, so this is not even a
   consistent projection. Feature values jump non-monotonically along a trajectory and the
   deltas become meaningless.

## Idea 2 — objects indistinguishable by concepts and roles

As stated it fails for the same reason as 1(b): *keeping only one* deletes an object and
changes cardinalities. The criterion is also wrong on its own terms. "Indistinguishable by any
concept or role" is evaluated against the *generated pool*, which grows with `max_complexity`
every round; and pairwise indistinguishability of two objects does not imply the states are
indistinguishable, since a role can separate the pair even when no concept separates the
members.

Both defects are repaired by the same reformulation: replace "indistinguishable" with a
**structural symmetry**, and replace "delete one object" with "keep one representative per
**orbit**".

### Definitions

Let `Ct` be the domain constants, `G` the goal atom set, and `σ(s) = s ∪ {p_G(t) : p(t) ∈ G}`
the dlplan structure that `feature_generator._get_state_from_goal` builds. A permutation γ of
the objects is a **symmetry of s** if it fixes `Ct` pointwise, preserves type tags, and
satisfies `γ(G) = G` and `γ(s) = s`. Write `Aut(s)` for this group, and `s ≅ t` when some
type/constant/goal-preserving γ has `γ(s) = t`.

**Prop 1 (feature invariance).** Every dlplan concept, role and feature is defined by a formula
over the primitive predicates and named constants, so for `γ ∈ Aut(s)` the extensions are
γ-invariant, and for `s ≅ t` via γ we get `C^σ(t) = γ(C^σ(s))` and `f(σ(t)) = f(σ(s))` for every
Boolean/Numerical feature. Induction over the dlplan grammar: and, or, not, some, all, diff,
projection, count, concept-distance, role composition and transitive closure are all
automorphism-invariant. Constants must be fixed because `c_one_of` names them.

**Prop 2 (transition equivariance).** For *any* object permutation γ, ground action `a` and
state `s`: `a` is applicable in `s` iff `γ(a)` is applicable in `γ(s)`, and
`succ(γ(s), γ(a)) = γ(succ(s, a))` as a set, `oneof` included. Grounding in `ground.py` commutes
with object renaming, and `check_formula`/`apply_effects` are purely syntactic on atoms.

**Corollary.** For `γ ∈ Aut(s)`, the actions `a` and `γ(a)` share an action name and source
state, have ≅-successors, and therefore identical feature values and identical `trans_delta`. No
policy in either encoding can distinguish them. Symmetrically, `s ≅ t` implies identical
`eval`/`c_eval`/`r_eval` profiles and identical goal/alive status.

### The sound reductions

* **R1 — orbit-space expansion.** Key `StateSpaceGraph.nodes` by a canonical form of `≅` rather
  than by the raw state, so isomorphic states reached along different paths become one node.
* **R2 — action-orbit pruning.** At each expanded node, partition the applicable ground actions
  into `Aut(s)`-orbits and expand one representative per orbit.

`≅` is a bisimulation for everything the encodings observe — goal-hood, aliveness, feature
vectors, action names, feature deltas — and every expressible policy is `≅`-invariant. The
"good" relation is *already* closed under feature-equivalence: that is literally the D2
constraint in `solve_constraints.lp` and the final integrity constraint in `solve_datalog.lp`.
So every π-trace in the full state space is mirrored state-by-state, up to `≅`, by a π-trace in
the reduced graph, and goal-reachability, dead-end status and strong-cyclicity transfer back.
The separation constraints are not weakened either: a pruned transition is
feature-indistinguishable from a kept one, so the constraint it would have generated is
subsumed. `execute_policy` re-validates the final policy on every original problem regardless,
so a residual error could only cost a failed round, never an incorrect policy.

## Measurements

### The problem-level gate

`Γ₀` is the automorphism group of the structure built from *static atoms ∪ goal ∪ types ∪
constants* (a static predicate is one that never occurs in any action effect). Since static
atoms are contained in every reachable state, `γ(s) = s` implies `γ(static) = static`, so
`Aut(s) ⊆ Γ₀` for every `s`, and `s ≅ t` requires an element of `Γ₀`. **If `Γ₀` is trivial, no
state has any symmetry and both reductions are provably no-ops.** This makes the gate a sound,
one-off, per-problem test.

### Orbit reduction, complete automorphism groups

Groups computed by colour-preserving graph automorphism over the description graph (objects
coloured by type with constants pinned, atom nodes coloured by predicate, argument order
encoded by position nodes); state orbits by minimising over the whole group.

| problem | objects | \|Γ₀\| | \|S\| | orbits | reduction |
|---|---|---|---|---|---|
| blocks4ops-clear p005-1 | 5 | 24 | 866 | 45 | **19.2×** |
| gripper problem-4-1 | 8 | 48 | 384 | 36 | **10.7×** |
| visitall 3×3 | 9 | 8 | 712 | 161 | **4.4×** |
| blocks4ops-on p004-01 | 4 | 2 | 125 | 65 | 1.9× |
| miconic problem-3-1 | 9 | 2 | 384 | 240 | 1.6× |
| blocks3ops p005-2 | 5 | **1** | 501 | 501 | **none** |
| miconic problem-2-1 | 6 | **1** | 64 | 64 | **none** |

A first pass using only single transpositions reported visitall as having no symmetry. That was
a false negative: a grid reflection is a *product* of disjoint transpositions and is never a
single swap. Any implementation must use a real automorphism/canonical-labelling backend
(nauty, or BLISS via igraph), not a transposition scan. All numbers above use complete groups.

### Why the zeros are structural, not accidental

`Γ₀` is the automorphism group of *static ∪ goal*, and every goal atom pins the objects it
mentions. blocks3ops `p005-2` has five goal atoms over five objects, so nothing is left to
permute — `|Γ₀| = 1` is a theorem about that instance, not a measurement artifact. The same
holds for miconic instances whose passengers have distinct origin/destination pairs. Symmetry
survives only where the goal is *unspecific*: `clear(a)` (blocks4ops-clear), "all balls in
roomb" (gripper), "visit everything" (visitall).

Goal specificity is also roughly what makes an instance hard for this system. **Symmetry
reduction is therefore anti-correlated with difficulty**: it pays most where we already
succeed, and provably nothing on blocks3ops, the domain currently limited by ASP grounding.

### The obvious alternative fails the same way

Quotienting by the *feature vector* — merging states that agree on goal/alive flags and on every
feature in the current pool — needs no symmetry at all, so it looked like the right tool for
goal-pinned domains. Measured on blocks3ops `p005-2` with the `state` configuration:

| max complexity | \|F\| | \|S\| → classes | \|T\| → classes | pair reduction |
|---|---|---|---|---|
| 3 | 12 | 501 → 8 (62.6×) | 2140 → 27 (79.3×) | ~6280× |
| 4 | 47 | 501 → 330 (1.5×) | 2140 → 1700 (1.3×) | ~2× |
| 5 | 110 | 501 → 384 (1.3×) | 2140 → 2028 (1.06×) | ~1× |

The collapse is enormous at low complexity and gone by complexity 5. This is exactly backwards:
`ProblemIterator` escalates complexity precisely when a round fails, so the pool is richest —
and the quotient emptiest — at the moment the instance is largest. The reduction is big only
when the instance is already small.

## Conclusion

For the goal-pinned domains that currently limit the system, pruning object information from
states cannot help, in any of its sound forms:

* symmetry-based reduction is provably empty (`|Γ₀| = 1`);
* feature-vector-based reduction vanishes at the complexities where the instance is large.

The lever that does work on those domains is the one already on this branch: avoid building the
full state space at all (example plans plus frontier expansion, see
`frontier-expansion-results.md`). The remaining untried lever is the *encoding* rather than the
state space — `trans_diff` grounding at |T|²·|F| is the dominant term and is not addressed by
any state-space reduction.

Symmetry reduction remains worth implementing if the goal is to extend coverage on
gripper/blocks-clear/visitall-style domains, where 4–19× on |S| translates to two orders of
magnitude on the quadratic constraints. The plan below is kept for that case.

## Shelved implementation plan

Scope chosen at the time: the datalog path first (`use_example_plans` + `frontier_expansion`),
with a complete automorphism backend. The unrestricted `state`/`trans`/`d2l` path follows for
free once `StateSpaceGraph` is symmetry-aware.

### The central difficulty: frames

Once nodes are canonical representatives, an edge no longer leads to "the state you get by
applying `a`" but to a symmetric copy of it. Every edge must carry the permutation `γ_e` with
`γ_e(actual successor) = child.state`, and everything that replays or reconstructs action
sequences must compose them: plan-suffix matching in `StateSpaceGraph.__init__` (store
`γ_e(suffix)` at the child — exact by Prop 2), `action_path_from_root` (must return actions in
the *original* frame, since plans are always replayed from `problem.init`), and
`frontier._root_anchored_plan`/`reroot` (the node's stored state is a genuine state but not the
one the path from init reaches).

### Steps

0. **Dependency.** `pynauty` publishes only a `cp312` wheel plus an sdist, so on Python 3.14 it
   builds nauty from source and needs `cc` + `make`; verify `poetry install` before building on
   it. Fallback: `igraph` (BLISS backend, `canonical_permutation(color=…)` and
   `automorphism_group(color=…)`, ships a `cp39-abi3` wheel that runs on 3.14). Keep the backend
   behind one private function so the choice is a single edit. Add to the mypy
   `ignore_missing_imports` list.

1. **New `genfond/symmetry.py`**, self-contained:
   - `state_description_graph(problem, state, goal_atoms, constants)` — object vertices coloured
     by type tag with each domain constant given its own singleton colour (they are named in the
     dlplan vocabulary via `one_of` and must be fixed); one vertex per state atom coloured by
     predicate; one per goal atom coloured `(predicate, "goal")`; arity 1 joins directly, arity
     ≥ 2 gets a position vertex per argument coloured `(predicate, i)`. Numeric fluents
     (`FunctionEqualTo`, cf. `state_space_generator.get_num_vals`) get an atom vertex coloured
     `(function name, value)`.
   - `canonical_form(problem, state) -> (key, relabelling)`.
   - `symmetry_between(rho_s, rho_t)` = `rho_t⁻¹ ∘ rho_s`, the genuine symmetry taking `s` to
     `t`; both structures embed the same goal atoms, so the composite preserves goal, types and
     constants. Assert `γ(s) == t` behind a debug flag.
   - `state_symmetries(problem, state)` — generators projected onto object vertices.
   - `action_orbits(actions, generators)` — union-find over applicable ground actions.
   - `perm_atom`/`perm_state`/`perm_action`/`compose`/`invert` helpers.
   - `problem_symmetry_gate(domain, problem)` — the `Γ₀` test above. This is what keeps
     blocks3ops and miconic at zero cost.

2. **`genfond/state_space_generator.py`**, behind `config["symmetry_reduction"]` (default off):
   - `StateSpaceNode` gains `relabelling`; `add_node` keys `self.nodes` by the canonical form and
     computes the edge permutation.
   - Store edge permutations in a **new** `node.edge_perm: dict[tuple[Action, int], Perm]` rather
     than changing the shape of `node.children` — `children` has many readers
     (`feature_generator.py:211,225,322,348,575,629`, `state_space_vis.py:35`,
     `iterative_solver.py:93`, `can_reach`, `find_nodes_leading_to_dead`, `prune_nodes`) and none
     of them need the permutation.
   - `prune_nodes` (`:390-397`) tests `child.state in self.nodes`; switch to the canonical key.
     The `dead_states` membership test (`:278`) and the `selected_states` set must be
     canonicalised too, or they silently stop matching.
   - Plan replay: store `γ_e(suffix)` at the child, so `add_plan_suffixes` and the re-expansion
     fixpoint operate in the child's own frame.
   - Action-orbit pruning behind a separate `config["symmetry_action_pruning"]`, and **never
     prune an action that one of the node's plan suffixes prescribes** — pruning is optional, so
     this keeps plan matching untouched.

3. **`genfond/frontier.py`.** `action_path_from_root` composes the stored edge permutations while
   walking back and maps each edge action by the accumulated permutation, so the sequence is
   replayable from `problem.init`. `collect_frontier_states` then reports the *actual* state that
   path reaches (`γ⁻¹(node.state)`), because `reroot` builds a problem from it and
   `_root_anchored_plan` requires root-anchoring. Assert that replaying the returned path from
   `problem.init` yields the returned state — this single check catches every frame bug.

4. **Config.** Add `symmetry_reduction: false`, `symmetry_action_pruning: false`,
   `symmetry_max_objects: 30` to `genfond/config/default.yaml` (required — a CLI flag with no
   default there is silently ignored). Thread through `FeaturePool.__init__` →
   `generate_state_space` (`feature_generator.py:154-161`). Log the gate result, group size, and
   |S| before/after per problem.

5. **Tests** (`tests/test_symmetry.py`, plus `test_generate_state_space.py`, `test_frontier.py`):
   gate trivial for blocks3ops `p005-2` and miconic `problem-2-1`, non-trivial for gripper,
   blocks4ops-clear and visitall; `canonical_form` agrees on a state and a hand-applied symmetry
   and differs under a non-symmetric perturbation; `symmetry_between` really maps one state onto
   the other; reduced state counts match the table above; the **replay invariant** for every
   node; the existing "every state on an example plan is expanded" test (commit `c9219e9`) still
   passes; end-to-end, the set of solved problems must not shrink with reduction on. Note that
   `tests/test_generate_state_space*.py` and `test_frontier.py:42-43` index `nodes[...]` by raw
   state and would need the canonical key.

6. **Benchmark.** `THREADS=1`, fixed seed, `--max-memory` always set. gripper, blocks4ops-clear,
   blocks4ops-fond-clear, visitall against blocks3ops and miconic `problem-2-1` as the
   zero-symmetry negative controls, which must show no regression.

## Reproducing the measurements

The numbers above came from throwaway scripts, not committed code. To redo them: build the
description graph as described in step 1, enumerate colour-preserving automorphisms (networkx's
`GraphMatcher` is adequate at these sizes), and minimise each state of
`StateSpaceGraph(domain, problem).nodes` over the group. For the feature-vector table, build a
`FeaturePool(domain, [problem], ConfigHandler(type="state"), max_complexity=cx)` and count
distinct `(node.goal, node.alive == Alive.ALIVE, feature vector)` tuples over
`fp.node_id_to_state_ids`.
