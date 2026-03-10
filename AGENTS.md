# Genfond Workspace Instructions

## Project Overview

**Genfond** is an iterative learning system for generalizable policies. It uses a combinatorial approach that works as follows:
1. Expand the state space of training problems
2. Use ASP to find a policy:
   * For each alive non-goal state, select at least one transition leading towards the goal.
   * Select features that distinguish the good from the non-good transitions.
3. Extract the policy from the ASP solution
4. Iterate by testing the policy on new problems, add new problems to the training set if not already solved.
Features are generated using the DLPlan library.

### Supported problem types
The framework supports multiple types of problems:
* Deterministic planning, where each action has exactly one outcome.
* Fully Observable Nondeterministic (FOND) planning, where actions may have multiple outcomes.
* Some preliminary support for numeric planning, although this is incomplete.

### Solution forms
Multiple types of problems and solution forms are supported:
* Rule-based policies define the actions to take by the qualitiative change occurring in the features when taking the action. They focus on FOND problems. There are multiple variants: 
    * Rule-based policies with state constraints: State constraints characterize dead-end states that the policy must not visit.
    * Rule-based policies with transition constraints: Transition constraints characterize the transitions leading to dead-end states.
    * Exact rule-based policies: The policy rule must characterize all possible outcomes. This is deprecated and not used anymore.
  The configurations for these can be found in `genfond/config/default_state.yaml`, `genfond/config/default_trans.yaml`, `genfond/config/default_exact.yaml`.
  The most commonly used configuration is `genfond/config/default_state.yaml`.
* Rule-based policies for deterministic problems, which are similar to the rule-based policies above, but do not require specifiying bad alternative outcomes, and therefore the three variants collapse to a single configuration. The configuration is in `genfond/config/default_d2l.yaml`.
* Datalog policies define the actions using state features, DL concepts, and DL roles. They do not require the action model, because the action is only characterized given the current state. They focus on deterministic problems. There are multiple variants configured in `genfond/config/default_datalog-*`. These are different attempts but do not currently work well. The most relevant configuration is `genfond/config/default_datalog.yaml`.

### Supervised learning
For rule-based policies, a supervised learning variant is implemented. It uses a planner to create example plans, which are then used to create a partial state space. This is then used as input for the datalog solver.


**Main entry point:** `python -m genfond -t <policy_type> domains/<domain>/{domain.pddl,p*.pddl}`

## Architecture

```
iterative_solver.py (orchestrator)
├─ feature_generator.py    → DLPlan feature synthesis
├─ state_space_generator.py → Reachable state computation
├─ problem_iterator.py      → Iteration over problem instances, feature complexities, other configuration options
├─ solver.py               → Clingo ASP solver wrapper
├─ topk_planner.py         → Top-k plan extraction, used for supervised learning
└─ generate_policy.py      → Outputs one of 3 policy types
   ├─ policy.py (execute)
   ├─ rule_policy.py
   └─ datalog_policy.py
```

**Key insight:** The system works iteratively — solving easier problems first, then adding larger prolems and increasing feature complexity. Use `problem_iterator.py` to understand the iteration pattern.

## Development Setup

### Prerequisites
- **Python 3.14+** (required — not 3.13)  
- **Poetry** (not pip): `pip install --user poetry`  
- **Git** (for repo-based dependencies: dlplan, pygraphviz)

### Local Setup
```bash
poetry install --no-root
poetry shell
python -m genfond -h          # Verify installation
```

### Container Setup
The container setup is only useful for benchmarking. It is used to create an Apptainer image with all dependencies pre-installed.
This Apptainer image is used in a SLURM cluster environment to run large-scale experiments. For local development, use the Poetry setup above.

## Testing & Quality Assurance

### Test Commands
```bash
pytest                        # Run all tests
mypy genfond                  # Type checking (external libs skipped via config)
isort --check genfond tests   # Import order
black --check genfond tests   # Code style
```

### Code Style Enforced
- **Line length:** 119 characters (Black, isort)  
- **Type hints:** Required (mypy config ignores external libs: pddl, dlplan, unified_planning)
- **Import order:** Managed by isort (profile: "black")

Use `black` and `isort` to auto-format code before committing. This ensures consistency across the codebase.

## Coding Conventions

### Naming Patterns
- `*_policy.py` → Policy generation/execution  
- `*_generator.py` → Feature/state generators  
- `execute_*.py` → Policy runners  
- Clingo predicates: `feature(X)`, `feature_complexity(X, C)`, `action(...)`, `state(...)`

### Config System
- **Layered YAML:** `default.yaml` → type-specific files (`default_datalog.yaml`, etc.) → user file → CLI
- **Merging:** Uses ConfigHandler + mergedeep library  
- **Policy-specific configs:** Each of 3 policy types has separate config file

### ASP/Clingo Patterns
- Constraint programs stored in `genfond/*.lp`  
- Main solver wrapper: `solver.py` (minimal, exemplary pattern)  
- Clingo auto-parallelizes with `os.cpu_count()` (override via Solver if needed)

## Key Files & Patterns

| File | Purpose | Start Here? |
|------|---------|---|
| `solver.py` | Clingo ASP wrapper | ✅ Minimal, clean pattern |
| `config_handler.py` | YAML config merging | ✅ Override patterns |
| `iterative_solver.py` | Main orchestrator | ✅ Full flow + error handling |
| `feature_generator.py` | DLPlan integration | ⚠ Complex feature synthesis |
| `state_space_generator.py` | Reachable state graph in training problems | ⚠ State explosion handling |
| `problem_iterator.py` | Iteration state machine | ⚠ Result/LastStep enums |
| `tests/conftest.py` | Test fixtures | ✅ Both PDDL + programmatic fixtures |
| `tests/helpers.py` | Test utilities | ✅ shared test code |

**For feature work:** Start with `solver.py` → `config_handler.py` → `iterative_solver.py`.

## Common Pitfalls & Solutions

### Issue: Tests fail with import errors
**Solution:** Use `pytest --import-mode importlib` (configured in tox.ini)

### Issue: ASP grounding takes forever or OOMs
**Solution:** 
- Check state space size with `state_space_generator.py`  
- Reduce problem iteration (in `problem_iterator.py`)  

### Issue: mypy errors on external libs
**Solution:** Already configured in `pyproject.toml` — mypy ignores pddl, dlplan, unified_planning

## Development Workflow

### Adding a New Feature
1. Create feature in appropriate `*_generator.py` module
2. Add corresponding Clingo predicate in `.lp` files  
3. Add tests in `tests/test_*_generator.py` 
4. Update config if needed (defaults in `genfond/config/`)
5. Run: `black`,`isort`, `mypy`, `pytest` before commit

### Modifying Policy Types  
- `policy.py` defines **PolicyType** enum  
- Each type has: `generate_{type}_policy()` + `execute_{type}_policy()`  
- Config: separate YAML per type in `genfond/config/`

### Debugging State Space Issues
- Use `feature_generator.py` to inspect features  
- Use `state_space_generator.py` to visualize transitions  
- Use `state_space_vis.py` for GraphViz rendering

### Profiling Performance
- Clingo auto-threads; use `--max-threads` if CPU-bound  
- Feature generation is expensive; check `feature_generator.py` complexity  
- Problem iteration can skip problems; tune in `problem_iterator.py`

## Performance Considerations

- **Multi-threaded by default** — Clingo uses all CPUs  
- **Memory-heavy** — State space graphs can be large; monitor on large domains  
- **Feature generation is slow** — DLPlan runs feature synthesis; iterative deepening on complexity  
- **Top-k planning overhead** — Required for feature observation; impacts runtime

## Testing Fixtures & Utilities

**Available fixtures** (in `tests/conftest.py`):
- `simple_blocks`, `fond_blocks`, `typed_blocks`, etc. — PDDL domains
- Programmatic domain construction via `pddl` library  
- Helpers in `tests/helpers.py` for common assertions

**Example test pattern:**
```python
def test_something(simple_blocks):
    # simple_blocks is a pddl.Domain object
    assert simple_blocks.name == "blocks"
```

## Related Commands

```bash
# Execute learned policy on test problems
python execute_policy.py domains/.../domain.pddl policy.pickle
```

## References
- [README.md](README.md) — Installation & usage overview
- [Makefile](Makefile) — Build targets (container builds), only relevant for building Apptainer images for large-scale benchmarks
- `pyproject.toml` — Dependencies, Python 3.14+ requirement

---

**Last updated:** March 2026  
**For AI agents:** Focus on `solver.py`, `config_handler.py`, and `iterative_solver.py` for architectural understanding. Respect Python 3.14+ and Poetry setup — these are non-negotiable due to dependency constraints.
