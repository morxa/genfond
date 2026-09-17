# Policy-anchored labels (H29): implementation

Date: 2026-09-17. Branch: `hyp/anchor-labels` (parent: `hyp/policy-plans`, commit `1479325`).

**Status: implemented and unit-tested. No benchmark numbers yet** — this document describes the
mechanism, not a measured result. A/B it per `docs/experiments-log.md` before drawing any
conclusion about coverage.

## The problem

H27 (`policy_conformant_plans`, `docs/policy-plans-results.md`) records the trajectory the best
policy P took on every problem it solved as an example plan, so the plan-restricted
`StateSpaceGraph` of every later round still *contains* that trajectory and P stays **feasible**
on the ASP instance.

Feasible is not the same as preferred. The measured failure is: a near-general P appears
(86–93 of 95 problems); the loop adds a problem P fails on to the training set; on that enlarged
instance a cheaper *patchwork* model — one fitted to the training set, not to the family — wins
the `#minimize` over feature complexity, and the run drifts for hours optimising away from the
policy that already almost worked.

H29 makes P preferred: the round's model must **agree with P** on the problems P already solved,
which leaves the optimisation only the question of how to additionally cover the new problem.

## The mechanism

Config key `anchor_policy_labels` (default `false`, documented in `genfond/config/default.yaml`
next to `policy_conformant_plans`). Per round, when a best policy exists:

1. `solve_iteratively` keeps `best_trajectories` — the trajectories of the *best* policy,
   replaced wholesale each time a new best is adopted, so a rejected candidate's trajectories
   never pin anything. It hands `solve_step` the subset for the problems in
   `iter_kwargs["active_problems"]`.
2. `FeaturePool._emit_anchors` replays each trajectory from `problem.init` through the very
   state graph the instance was built from and emits `anchor(I, S, A).` for every step.
3. `solve_datalog_sig.lp` and `solve_datalog.lp` carry `#defined anchor/3.` plus a separate
   `#program anchor.` part holding one constraint:
   `:- anchor(I, S, A), not good_action(I, S, A).`
4. `Solver(..., anchors=True)` grounds that part. `iterative_solver.solve` runs the round
   anchored; if the anchored solve comes back `UNSATISFIABLE` or `UNKNOWN`, it logs
   `Anchored solve unsatisfiable; retrying without anchors` and re-solves the identical instance
   with `anchors=False`. Only that second solve is ever reported as the round's answer.

## Files and functions

| File | Change |
|---|---|
| `genfond/config/default.yaml` | new key `anchor_policy_labels: false` with the rationale |
| `genfond/feature_generator.py` | `FeaturePool(anchor_plans=...)`, `_emit_anchors`, `_children_of`, and `anchored_transitions` / `anchored_problems` / `anchors_dropped` |
| `genfond/solve_datalog_sig.lp` | `#defined anchor/3.` + `#program anchor.` |
| `genfond/solve_datalog.lp` | same |
| `genfond/solver.py` | `Solver(anchors=...)` grounds the `anchor` part |
| `genfond/iterative_solver.py` | `solve(anchor_plans=...)`, the `_run_solver` helper and the fallback, `solve_step(anchor_plans=...)`, `best_trajectories` in `solve_iteratively`, the solve_prog guard, the stats |
| `tests/test_anchor_labels.py` | 13 tests |

## Decisions

**How a transition is identified.** As `(instance id, node id, action string)` — the first three
arguments of `trans/4`, and exactly the shape `forced_good/3` uses. The anchor constrains
`good_action(I, S, A)`, i.e. the *action* at that state, not a particular outcome: a FOND action
has several outcomes and the recorded trajectory only shows the one execution happened to sample.
Replay therefore carries a *set* of nodes, exactly like `plan_visited_states` and like
`StateSpaceGraph`'s propagation of a plan suffix to all matching successors.

**Signature classes.** With `--type datalog-sig` goodness is a function of the action signature
(`good_sig`/`bad_sig` and `:- good_sig(K), bad_sig(K).`), so anchoring one occurrence already
forces every occurrence of its class good, wherever it occurs — the cross-state consequence
`genfond/forced_labels.py` spells out as its step 4. Nothing about the class is emitted, and
nothing may be: the class is anchored *because the program says so*. Stating it as a fact of its
own would be assertion rather than consequence, which is the unsound half of that argument.
`tests/test_anchor_labels.py::test_anchoring_one_occurrence_forces_its_signature_class_good_everywhere`
pins this behaviour down.

This is also what makes an anchor genuinely contradictory rather than merely redundant: P's
action at one state may share a signature class with an action that must be bad at another state
(its outcome is a dead end, say), and then no model satisfies the anchor even though policies
exist. Hence the fallback.

**Fallback semantics.** The anchored attempt is a *preference*, never evidence. A round's answer
always comes from a solve without anchors whenever the anchored one did not produce a model, so:

- an anchored `UNSATISFIABLE` can never reach `ProblemIterator.set_last_result` as
  `Result.NO_SOLUTION`, and therefore can never set `refuted_complexity` or enable
  `min_feature_complexity` for later rounds (see the "Refuted complexity levels" section of
  `AGENTS.md`);
- `UNKNOWN` (the solve budget ran out before any model) is retried for the same reason: the
  anchored attempt proves nothing, and the unanchored program may still find a model;
- when the anchored solve *does* produce a model, that model satisfies every constraint of the
  unanchored program too — anchors only remove models — so the round is sound as it stands, and
  the optimum it reports is the optimum *among models agreeing with P*. That is the intended
  change in what the round optimises.

Cost: at most one extra clingo solve on the rounds that hit the fallback, counted as
`anchorFallbacks`.

**Anchors combine with `optimal_model_limit` for free.** Anchors are constraints, so the
enumeration in `Solver.enumerate_optimal` simply ranges over the models that satisfy them;
`_choose_candidate` then picks by coverage among policies that all agree with P.

**Dropped anchors.** A step is skipped (and counted in `anchorsDropped`) when

- its problem is not in this round's training set — requirement 4: the instance holds no states
  for it;
- its source state is missing from the graph, or is dead or a goal there — the `good_trans`
  choice only ranges over alive non-goal states, so there is no `good_action` to force;
- its action has an outcome that is neither alive nor pruned — `solve_datalog*.lp`'s second
  constraint forbids selecting it, so the anchor would make the round unsatisfiable on its own
  and *every* round would pay for a fallback.

Dropping only weakens a preference, so this cannot make a policy unreachable. It does mean the
mechanism is close to inert without `policy_conformant_plans`: without H27 putting P's
trajectories into the state space, most steps are simply not in the graph. `solve_iteratively`
logs a warning for that combination.

**Scope.** Anchors are applied to the rounds of the main loop only.
`_final_cost_minimization_pass` passes no `anchor_plans`: that pass exists to shave the cost of
an already-chosen policy and validates every candidate itself, so constraining it to agree with
the incumbent would defeat its purpose.

**Guards.** `anchor_policy_labels` requires `solve_prog` to be `solve_datalog.lp` or
`solve_datalog_sig.lp` (`ValueError` otherwise, like the `fix_forced_labels` guard) — the
`anchor` program part exists nowhere else. It is also inert without `use_example_plans` and is
switched off with a warning when `keep_best_policy` is false, since "the best policy" is then
undefined.

## Stats

| Key | Meaning |
|---|---|
| `anchoredTransitions` | anchors emitted this round |
| `anchoredProblems` | problems they came from |
| `anchorsDropped` | steps skipped for the reasons above |
| `anchorFallbacks` | cumulative count of rounds that re-solved without anchors |

Plus one INFO line per round from `_emit_anchors`:
`Anchoring N transition(s) from the best policy (K problems)`.

## Off switch

`anchor_policy_labels: false` (the default) emits no `anchor/3` fact, never grounds the `anchor`
part, and never passes `anchor_plans` to `solve_step` — the ground program and the round
structure are byte-for-byte what they were before this change. The only code that runs at all is
the flag test itself.

## Tests

`tests/test_anchor_labels.py` (13 tests, all fast; the whole file runs in under a second):

- (a) the constraint: an anchored transition is labelled good in both encodings; a contradicting
  anchor is `UNSATISFIABLE` while the *same instance text* solves without the `anchor` part;
  anchoring one occurrence forces its signature class good at another state.
- (b) the fallback: with an injected unsatisfiable anchored solve, `solve_step` runs exactly one
  anchored and one plain solve, logs the fallback line, increments `anchorFallbacks`, and returns
  `Result.SUCCESS` with the unanchored solve's status — so nothing refutes the complexity level.
  A round without a contradiction solves exactly once.
- (c) the off switch: no `anchor(` facts in the instance and `anchor_plans=None` at the
  `solve_step` call site; a trajectory for a problem outside the training set is ignored; steps
  outside the plan-restricted state space are dropped, not emitted.
- plus: the anchored transitions are exactly the trajectory's own steps and each names a `trans/4`
  the instance really has; only the *best* policy's trajectories, restricted to the round's
  active problems, reach `solve_step`; and the anchors combine with the `lazy_pairs` loop.

Gates: `black`, `isort`, `mypy genfond tests` clean; `pytest --import-mode importlib -q` →
271 passed, 1 skipped (baseline 258 passed, 1 skipped, plus the 13 new tests).

## What to measure next

A/B `anchor_policy_labels: true` against the H27 baseline on the suite where the drift was
observed (the 95-problem blocks set), per `docs/experiments-log.md`: coverage, rounds to the best
policy, `anchorFallbacks`, and whether the best policy's cost stops climbing after it appears.
