# Genfond Workspace Instructions

## Project Overview

**Genfond** is an iterative learning system for generalizable policies — one policy that solves a whole family of PDDL problems. It uses a combinatorial approach that works as follows:
1. Expand the state space of training problems
2. Use ASP to find a policy:
   * For each alive non-goal state, select at least one transition leading towards the goal.
   * Select features that distinguish the good from the non-good transitions.
3. Extract the policy from the ASP solution
4. Iterate by testing the policy on new problems, add new problems to the training set if not already solved.

Features are generated using the DLPlan library.

**Main entry point:** `python -m genfond --type <policy_type> domains/<domain>/{domain.pddl,p*.pddl}`

### Supported problem types
The framework supports multiple types of problems:
* Deterministic planning, where each action has exactly one outcome.
* Fully Observable Nondeterministic (FOND) planning, where actions may have multiple outcomes.
* Some preliminary support for numeric planning, although this is incomplete.

### Solution forms
Multiple types of problems and solution forms are supported:
* Rule-based policies define the actions to take by the qualitative change occurring in the features when taking the action. They focus on FOND problems. There are multiple variants:
    * Rule-based policies with state constraints: State constraints characterize dead-end states that the policy must not visit.
    * Rule-based policies with transition constraints: Transition constraints characterize the transitions leading to dead-end states.
    * Exact rule-based policies: The policy rule must characterize all possible outcomes. This is deprecated and not used anymore.

  The configurations for these are `genfond/config/default_state.yaml`, `default_trans.yaml`, `default_exact.yaml`. The most commonly used configuration is `default_state.yaml`.
* Rule-based policies for deterministic problems, which are similar to the rule-based policies above, but do not require specifying bad alternative outcomes, and therefore the three variants collapse to a single configuration: `genfond/config/default_d2l.yaml`.
* Datalog policies define the actions using state features, DL concepts, and DL roles. They do not require the action model, because the action is only characterized given the current state. They focus on deterministic problems. There are multiple variants configured in `genfond/config/default_datalog-*`. These are different attempts but do not currently work well. The most relevant configuration is `default_datalog.yaml`.

## Commands

Dependencies are managed by Poetry and require **Python 3.14+** (not 3.13 — dependency constraint). Run everything through the venv (`poetry run …` or inside `poetry shell`).

```bash
poetry install --no-root --with=dev
python -m genfond -h                     # verify installation

pytest                                   # all tests
pytest tests/test_solver.py              # one file
pytest tests/test_solver.py::test_solver_equiv  # one test
pytest -k siw -v                         # by name pattern

mypy genfond tests                       # CI checks both dirs (tox.ini only checks genfond)
isort --check genfond tests
black --check genfond tests
tox                                      # runs the whole CI check set locally
```

Formatting is Black/isort with **line length 119** (`pyproject.toml`, `setup.cfg`). Run `black genfond tests` and `isort genfond tests` before committing.

Running the solver:

```bash
# Iterative solver (main entry point); --type selects the policy formulation
python -m genfond --type state domains/non-deterministic/acrobatics/{domain.pddl,p*.pddl}

# One-shot: solve all given problems once at max_complexity, no iteration
python -m genfond --one-shot --max-complexity 6 domains/non-deterministic/acrobatics/{domain.pddl,p0002*}

# Execute a pickled policy
python execute_policy.py domains/.../{domain.pddl,p0005.pddl} out.policy
python print_policy.py out.policy
python check_solvable.py domains/.../{domain.pddl,p*.pddl}
```

`Makefile` + `Dockerfile` + `genfond.bash` + `run_benchmarks.bash` exist only for building an Apptainer image and submitting SLURM benchmark jobs for large-scale experiments; they are not part of the local dev loop.

## Architecture

```
iterative_solver.py (orchestrator)
├─ problem_iterator.py      → iteration over problems, feature complexities, plans
├─ feature_generator.py     → DLPlan feature synthesis → ASP instance
│  └─ state_space_generator.py → reachable state computation
├─ solver.py                → clingo ASP solver wrapper (+ genfond/solve*.lp)
├─ siw_planner.py / topk_planner.py → example plans for supervised learning
└─ generate_policy.py       → rule_policy.py | datalog_policy.py
   └─ execute_policy.py     → execute_rule_policy.py | execute_datalog_policy.py
```

Core loop (`iterative_solver.solve_iteratively`):

1. `ProblemIterator` (`problem_iterator.py`) yields a configuration to try: a growing set of `active_problems` (sorted by object count), a max feature `complexity`, `max_cost`, whether to use the unrestricted feature generators, and the currently active example plans.
2. `solve()` builds a `FeaturePool` (`feature_generator.py`), which expands each problem's state space (`state_space_generator.StateSpaceGraph`) and runs DLPlan feature synthesis over the collected states, then serializes everything to an ASP instance (`to_clingo()`).
3. `Solver` (`solver.py`) grounds `genfond/<solve_prog>.lp` plus that instance and solves it. `max_cost` and `enforce_highest_complexity` are passed as extra `#program` parts (`limit_feature_cost`, `min_feature_complexity`) so each round must beat the previous policy's cost.
4. `generate_policy.py` turns the clingo model into a `Policy` (`rule_policy.py`) or `DatalogPolicy` (`datalog_policy.py`).
5. The policy is executed on all problems (`execute_policy.py` → `execute_rule_policy.py` / `execute_datalog_policy.py`). Solved problems are recorded; unsolved ones drive the next iteration.

**Key insight:** the system solves easier problems first, then adds larger problems and increases feature complexity. `ProblemIterator.__next__` is the state machine that decides *how* to escalate when a round fails, in priority order: add another example plan → enable unrestricted feature generators → increment complexity → add the next unsolved problem to the training set. Read it before changing anything about iteration behaviour; `set_last_result()` **must** be called between iterations (asserted).

### Refuted complexity levels

`min_feature_complexity(c)` rests on exactly one fact: complexity `c-1` was **refuted** — no solution exists that uses only features of complexity ≤ `c-1`. That is a claim about one particular ASP instance, so `ProblemIterator` tracks it explicitly in `refuted_complexity` and returns `enforce_highest_complexity` per round; never hardcode it in the caller.

- Only a round over the **full** feature pool refutes a level. A restricted-generator round proves nothing about the unrestricted pool, which is a superset.
- A `SUCCESS` refutes its own level too: clingo minimises the feature cost, so `cost[-1]` is optimal for that pool and nothing there beats the new `max_cost`. This is what keeps the `max_cost < complexity` short circuit in `solve()` available — that short circuit is only valid while enforcement is on, and is guarded on it.
- Every branch that changes the state space calls `_invalidate_refutations()`: `INC_PLANS`, `EXPAND_FRONTIER` (when `frontier_progress`), and the add-a-problem branch. Adding a plan or a dead end can make a *simpler* policy possible, so a stale bound would exclude it. Without this the search enforced a bound that, after the first plan addition, was never re-established under the current state space — only one complexity level is ever tried per plan set.
- The add-a-problem branch is the exception that keeps its bound: adding an instance is monotone (a selection solving the larger set solves every subset), so it resumes at `succ_complexity` with `refuted_complexity = succ_complexity - 1`. `unselect_problems` *replaces* the training set instead of extending it, and then nothing carries over.
- Refutations are relative to the current `max_cost`. Both branches that loosen `max_cost` back to `MAX_COST` also reset `refuted_complexity`, so the two never get out of step.

`reset_complexity_on_state_space_change` (default off) additionally restarts the sweep at `min_complexity` whenever the state space changes. `sweep_target` guards the interaction: the next plan is only added once the sweep has climbed *past* the level the last one was added at — with `>=` the restart and the addition alternate at one fixed level and the search never reaches the higher complexities at all.

It does **not** change which policies are reachable. The feature pool at complexity `c` contains every feature of complexity `< c`, and the round right after a state-space change runs without `min_feature_complexity`, so a policy expressible at complexity 2 is found by the round at complexity 4 as well — and preferred, since the `#minimize` is over feature cost. What the restart changes is the cost of getting there: the low-complexity instances are far smaller (roles alone contribute `n·m²` grounded values per state) and may still ground where the high-complexity one hits `Id out of range` or exhausts memory. The trade is a full re-sweep per added plan against a chance to succeed on a round whose larger instance cannot be grounded at all.

### Policy types

`--type X` maps to `genfond/config/default_X.yaml`, which sets `policy_type` (the `PolicyType` enum in `policy.py`: `EXACT`, `CONSTRAINED`, `DATALOG`) and `solve_prog` (which `.lp` file is loaded):

| `--type` | policy_type | solve_prog | notes |
|---|---|---|---|
| `state` (default) | CONSTRAINED | `solve_state_constraints.lp` | FOND, dead-end *states* constrained. Most used. |
| `trans` | CONSTRAINED | `solve_trans_constraints.lp` | FOND, dead-end *transitions* constrained |
| `d2l` | EXACT | `solve_d2l.lp` | deterministic problems |
| `datalog` | DATALOG | `solve_datalog.lp` | Datalog policies over DL concepts/roles; enables `use_example_plans` |
| `exact` | EXACT | `solve.lp` | deprecated |
| `datalog-actions`, `datalog-action-params` | DATALOG | resp. `.lp` | experimental variants |

The `--type` choices are **discovered at import time** by globbing `genfond/config/default_*.yaml` (`config_handler._discover_type_configs`). Adding a config file adds a policy type; there is no registry to update.

The ASP programs share `solve_constraints.lp` (via `#include`), which defines `trans_delta/6`, `good_action`, `bool_dist`, feature selection and the `#minimize` over feature complexity. Predicate vocabulary: `state/2`, `alive/2`, `goal/2`, `trans/4`, `eval/4`, `feature/1`, `feature_complexity/2`, `selected/1`.

### Config layering

`ConfigHandler` (a `dict` subclass) merges, in order: `config/default.yaml` → `config/default_<type>.yaml` → user `--config` file → CLI `vars(args)`. Merging is deep (`mergedeep`).

Important gotcha: CLI overrides are applied **only for keys that already exist in the merged config and whose value is not None**. A new `--foo` CLI flag has no effect unless `foo` also has a default in `default.yaml`.

### Supervised learning / example plans

For rule-based policies a supervised learning variant is implemented: a planner creates example plans, which are used to build a *partial* state space that feeds the solver. When `use_example_plans` is true (the `datalog` config), `StateSpaceGraph` is restricted to states reachable along those plans — off-plan states are not expanded, and a state is "revived" if some plan reaches it. This keeps the state space tractable for larger problems.

An off-plan successor that already satisfies the goal is classified as `Alive.ALIVE` at creation time and still not expanded: the goal check in the expansion loop only runs on states that are popped from the queue, so it would otherwise never see them.

Plan suffixes are propagated to a **fixpoint**. States are expanded in a single LIFO pass, so a plan can reach a node after that node was already expanded; `StateSpaceNode.add_plan_suffixes` deduplicates and reports whether the node gained anything, and a node that gains a suffix is queued again. Without this, the actions the late-arriving plan prescribes at that node are never matched and its successors are wrongly treated as off-plan — states demonstrably lying on an example plan were left unexpanded. Termination holds because a node only re-enters the queue when its suffix set grows, and that set is finite.

Planner selection is `config["planner"]` with per-planner settings under `config["planners"]` (`iterative_solver._get_example_plan_computer`):
- `siw` (default) — `siw_planner.py`, wraps the external `siw` package. Diversity comes from `branch` (branching over SIW serializations) and `restarts` (permuted action/goal orders). Plans are pulled **lazily** from a generator, so the iterator only pays for the plans it consumes.
- `topk_planner` — `topk_planner.py`, symk via unified-planning.

Both expose the same `compute_plans(domain_str, problem_str, planner_config) -> Iterator[Plan]` signature; a new planner must match it and be added to the `match` in `_get_example_plan_computer`.

### Frontier expansion

`frontier_expansion` (on by default for `datalog`) relaxes the plan restriction. Off-plan successors become `Alive.PRUNED` instead of `Alive.DEAD`, which `feature_generator` emits as `pruned/2`. `solve_datalog.lp` may then select a transition into such a state — `safe_state(I,S) :- pruned(I,S)` treats reaching it as success — but pays `1@2`, a *higher* priority than the feature-complexity minimize at `@0`, so a frontier transition is only used when the round is otherwise unsatisfiable.

The loop is closed in `frontier.py`:

1. The model reports the states it relied on as `frontier/2`; `collect_frontier_states` resolves those `(instance, state)` ids back to `State`s via `FeaturePool.lookup_node`, plus the action path from the root (`StateSpaceGraph.action_path_from_root`).
2. `expand_frontier` re-roots the problem at each state (`reroot`, which must pass `domain_name=`, not `domain=`) and asks the planner for a plan.
3. A plan is spliced onto the root path (`_root_anchored_plan`) and added as a new example plan — `StateSpaceGraph` always replays plans from `problem.init`, so plans must stay root-anchored.
4. No plan marks the state as a dead end. It then emits neither `alive/2` nor `pruned/2`, and the existing constraint at `solve_datalog.lp:22` stops the solver selecting it.

The frontier is closed off once a policy exists: `iterative_solver.max_prune_cost` returns `0` whenever `max_cost < MAX_COST`, because a tightened budget means the round is only trying to beat an existing policy's cost. Without this gate the solver reaches for the frontier on every post-success round and spends planner calls and state-space growth on shaving feature complexity rather than on gaining solvability — measured on `blocks3ops`, that was the difference between 5/7 and 4/7 problems solved.

`solve_step` returns `Result.FRONTIER` (and **no** policy) whenever the model uses any frontier transition; only a zero-frontier model is an acceptable policy. `Result.FRONTIER` must not tighten `max_cost` or mark problems solved — a frontier model's feature cost is artificially low. `ProblemIterator.__next__` has a top-priority branch that retries the identical configuration with the enlarged plan/dead-end sets, gated by `frontier_progress` (something new was learned) and `max_frontier_expansions` (loop guard).

Caveat: SIW is incomplete, so "no plan" does not prove a dead end. A false dead end can only prevent a policy from being found, never produce an incorrect one, because the final policy is re-validated by `execute_policy` on every problem.

Cost-vector gotcha: the `@2` level is absent from `model.cost` when no `pruned/2` fact grounds, so the vector is length 1 or 2 depending on the instance. Always index `cost[-1]` for the feature complexity; never `cost[0]`.

## Key Files

| File | Purpose | Start here? |
|------|---------|---|
| `solver.py` | clingo ASP wrapper | ✅ minimal, clean pattern |
| `config_handler.py` | YAML config merging | ✅ override patterns |
| `iterative_solver.py` | main orchestrator | ✅ full flow + error handling |
| `problem_iterator.py` | iteration state machine | ✅ `Result` / `LastStep` enums |
| `feature_generator.py` | DLPlan integration | ⚠ complex feature synthesis |
| `state_space_generator.py` | reachable state graph, plan-restricted expansion | ⚠ state explosion handling |
| `frontier.py` | frontier states → re-rooted planning → new example plans | ✅ small, self-contained |
| `state_space_vis.py` | GraphViz rendering of state graphs | debugging aid |
| `tests/conftest.py` | test fixtures | ✅ both PDDL and raw-ASP fixtures |

For feature work, start with `solver.py` → `config_handler.py` → `iterative_solver.py`.

## Conventions

- Module naming: `generate_*_policy.py` (model → policy object), `execute_*_policy.py` (policy → actions), `*_generator.py` (features / state spaces), `solve*.lp` (ASP programs).
- Logging uses named loggers (`genfond.<module>`); per-component levels are configurable via the `log:` map in the config (e.g. execution is silenced to `CRITICAL` by default).
- `State` is `frozenset[Formula]` — ground `pddl` atoms. `pddl` objects are hashable and used as dict keys throughout.
- Policies are persisted as **pickles** — changing the shape of `Policy`/`DatalogPolicy` invalidates saved `.policy` files.
- Type hints are expected; mypy ignores missing stubs only for the external libs listed in `pyproject.toml` (`pddl`, `dlplan`, `unified_planning`, `pygraphviz`, `mergedeep`).

### Adding a feature
1. Extend the appropriate `*_generator.py` module.
2. Add the corresponding clingo predicate in the relevant `.lp` file.
3. Add tests in `tests/test_*.py`.
4. Add a default to `genfond/config/default.yaml` if it is configurable (see the CLI-override gotcha above).
5. Run `black`, `isort`, `mypy`, `pytest` before committing.

## Tests

`tests/conftest.py` provides two kinds of fixtures:
- PDDL fixtures loaded from `tests/fixtures/pddl_files/<domain>/` — `simple_blocks`, `fond_blocks`, `typed_blocks`, `blocks_clear`, `doors`, `blocks3ops`, `childsnack`, … Each returns a **`(domain, problem)` tuple**, not a bare domain.
- Raw ASP program strings (`simple_program`, `program_with_nontriv_equiv`) for testing `Solver` without any PDDL parsing.

Solver-level tests should prefer the raw-ASP fixtures; they are far faster. `tests/helpers.py` holds shared assertions (`get_action`).

`domains/` holds the benchmark suites (`non-deterministic/`, `deterministic/`, `deterministic-new/`, `d2l/`) used for manual runs and benchmarking, not by the test suite.

## Common Pitfalls

- **Import errors in tests:** use `pytest --import-mode importlib` (what `tox.ini` does).
- **ASP grounding takes forever or OOMs:** the state space is too large. Check sizes via `state_space_generator.py`, lower `max_complexity`, or restrict the problem set. clingo parallelizes over `os.cpu_count()` by default; override with `-n/--num-threads`. `--max-memory` sets an `RLIMIT_AS` cap.
- **mypy errors on external libs:** already handled by the overrides in `pyproject.toml`.

---

`CLAUDE.md` is a symlink to this file — edit `AGENTS.md`, not the symlink.
