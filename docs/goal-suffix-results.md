# Rename the goal-predicate suffix from `_G` to `_g`: results

Date: 2026-09-15. Branch: `hyp/combo` → `hyp/goal-suffix`.

## Background

`dlplan`'s generator rules `EqualConcept::generate_impl` (`src/generator/rules/concepts/equal.cpp`)
and `AndRole::generate_impl` (`src/generator/rules/roles/and.cpp`) only recognize a role as the
"goal counterpart" of a base role when its predicate name equals `<base>_g` (lowercase). genfond's
`feature_generator.construct_vocabulary_info` / `_get_state_from_goal` / `construct_instance_info`
name goal predicates `<base>_G` (uppercase), so `c_equal` (and the goal-conditioned `r_and`) were
never generated for genfond, even with `generate_equal_concept` enabled.

An earlier attempt patched `dlplan` itself to accept both suffixes (commit
`66a0541aa0ad49180f27ce81175835872ec0fff2` on a since-deleted `fix-equal-goal-suffix` branch on
the `morxa/dlplan` fork). That patch was reverted: the simpler, more maintainable fix is for
genfond to adopt dlplan's existing convention instead of asking dlplan to accommodate genfond's.
**dlplan is unchanged** — still pinned at `cfd45615a8f87a4d92b4ba4a61a8dc9d873cf1c9`, no
reinstall needed.

## The change

Renamed the goal-predicate suffix from `_G` to `_g` in `genfond/feature_generator.py` (vocabulary
construction, `_get_state_from_goal`, `construct_instance_info`) and in the expected-string
fixtures of `tests/test_execute_datalog_policy.py`, `tests/test_feature_generator.py`,
`tests/test_generate_datalog_policy.py`. No other production code referenced the suffix (grepped
`_G` across `genfond/`, `tests/`, `scripts/`; the only other hit was the unrelated `_Group` class
in `action_signatures.py`).

**Breaking change:** pickled `.policy` files produced before this change embed dlplan feature
strings like `r_primitive(on_G,0,1)`; after this change genfond's own vocabulary only registers
`on_g`, so old pickles fail to re-parse (`RuntimeError: Failed parse.`, "undefined predicate"
pointing at the `_G` remnant). Regenerate policies rather than loading old ones.

## Gates

In `/home/thofmann/code/genfond-wt/goal-suffix`: `black`, `isort`, `mypy genfond tests` all clean.
`pytest tests/ --import-mode importlib -q`: **130 passed, 1 skipped, 1 failed** —
`test_datalog_signatures.py::test_both_encodings_find_the_same_policy[gripper]`. Confirmed
pre-existing-clean (passes on `hyp/combo` before this change) and caused by this change: with
`c_equal` now available, the plain (`datalog`) and signature-quotiented (`datalog-sig`) encodings
find two *different* optimal policies for gripper at complexity 4 — both cost `[6]`, 5 rules vs 4
rules, differing only in which of several equal-cost `c_equal`-based formulations the solver
picks. `test_both_encodings_agree_on_feature_cost` (which checks cost only, not the exact rules)
still passes for both fixtures and both `lazy` settings. This is a real, expected side effect of
unlocking a new tied-cost alternative, not a suffix-correctness bug; reported rather than fixed
per house rule (research code: report, don't fix).

## Pool sizes (blocks3ops, p003-1 + p004-1, `datalog-sig`)

| complexity | concepts | roles | `c_equal` concepts |
|---|---|---|---|
| 3 | 30 (was 29) | 29 (was 28) | `c_equal(r_primitive(on,0,1),r_primitive(on_g,0,1))` |
| 4 | 90 (was 88) | 76 (was 73) | + `c_not(c_equal(r_primitive(on,0,1),r_primitive(on_g,0,1)))` |

`c_equal` is generated for the first time; before, the pool sizes were identical but `c_equal`
count was always 0.

## End-to-end: blocks3ops-local (training) + blocks3ops-heldout (12 problems)

`--type datalog-sig -n 1 --seed 0 --max-memory 6000 --add-problem-after-success`, `PYTHONHASHSEED=0`.

- Training: solves 10/10 blocks3ops-local problems, cost `[0, 7]`, final policy has **36 rules**
  and makes heavy use of `c_equal(r_primitive(on,0,1),r_primitive(on_g,0,1))` (and its negation)
  alongside `c_primitive(clear_g,0)`, `c_primitive(ontable_g,0)`, `c_primitive(ontable,0)`, and
  the roles `r_primitive(on_g,0,1)` / `r_not(r_primitive(on_g,0,1))`.
- Held-out (`scripts/eval_policy.py --seed 0 -i 3`, 12 problems, 8-30 blocks): **0/12 solved**
  ("No action found" / "Cycle detected" on all 12) — genuine generalization failures, not parse
  errors.

**Gotcha found and worked around, not a bug in this change:** `scripts/eval_policy.py` run as a
plain script (`python scripts/eval_policy.py`, not `python -m ...`) put `scripts/` at
`sys.path[0]`, which doesn't contain a `genfond` package, so Python fell through to the
**editable-installed** `genfond` in the venv — which points at `/home/thofmann/code/genfond`
(the main repo, still on the `_G` convention), not this worktree. That produced spurious
`Failed parse.` / "undefined predicate ...on_g" errors (the pickled policy embeds `on_g`, but the
main-repo `genfond` build vocabulary with `_G`). Running with `PYTHONPATH=.` (from the worktree
root) forces the local worktree's `genfond` package to resolve first, after which the script
correctly reaches the real (unsolved) held-out outcomes above. `-m genfond` invocations were
never affected (module mode puts cwd on `sys.path[0]`), so the pool-size and end-to-end training
numbers above are unaffected by this.

This 0/12 is worse than the task's reference point of "a hand-written preset with
`c_equal(on,on_G)` gave 5/12" — but that preset was hand-crafted directly into a policy, not
learned by the iterative solver from `blocks3ops-local`'s 10 training problems (up to 5 blocks)
against held-out problems with 8-30 blocks. The iterative solver's cost-minimal 36-rule policy
apparently overfits to the small training regime; `c_equal` being available doesn't by itself
close that gap. Reported as-is, not investigated further (out of scope for this task).

## Regression suites (no held-out), `--type datalog-sig -n 1 --seed 0 --max-memory 6000 --add-problem-after-success`

| suite | solved | cost | before |
|---|---|---|---|
| gripper-local | 5/5 | 6 | 5/5, cost 6 |
| miconic-local | 4/4 | 12 | 4/4, cost 12 |
| blocks4ops-clear-local | 4/4 | 2 | 4/4, cost 2 |
| delivery-local | 4/4 | 9 | 4/4, cost 9 |

No regressions: identical solved counts and costs to the pre-change baseline. None of these four
domains' optimal policies benefit from `c_equal` at their existing complexity budgets.
