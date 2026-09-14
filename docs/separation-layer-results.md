# Cutting the pairwise separation layer

Date: 2026-08-06. Branch: `reduce-asp-separation-layer` (based on `learn-from-examples`).

Follow-on from `state-space-reduction-analysis.md`, which ruled out state-space pruning for the
goal-pinned domains. The remaining lever was the encoding, and both encodings turn out to be
dominated by one thing.

## Where the ground program goes

Counted with `clingo --text`, head predicate by head predicate.

**datalog** (`solve_datalog.lp`, gripper `problem-2-1`, complexity 3 — a **42-state** instance,
20 concepts, 20 roles, 5 features), 1 008 618 ground rules:

| head | rules | share |
|---|---|---|
| `c_distinguished` | 504 392 | 50.0% |
| `r_distinguished` | 475 928 | 47.2% |
| `#show` | 24 609 | 2.4% |
| `c_eval` / `r_eval` | 1 068 / 976 | 0.2% |
| `f_distinguished` | 572 | 0.1% |
| everything else | < 700 | 0.1% |

**rule-based** (`solve_state_constraints.lp`, blocks4ops-on `p004-01`, complexity 4 — 125 states,
272 transitions, 29 features), ~1 030 000 ground rules: `trans_diff` 437 272 (42%),
`distinguished` 437 272 (42%), `bool_dist` 70 304 (7%), everything else < 1%.

97% of the datalog program and 85% of the rule-based one is pairwise separation. In the datalog
case the **concept and role** comparisons dominate and `f_distinguished` is noise.

## D1 — quotient the datalog separation layer by action signature

`solve_datalog.lp`'s separation constraint observes an `(instance, state, action)` triple only
through:

* the action name,
* for each parameter position `N` and concept `C`, whether `c_eval(I,S,C,P_N)`,
* for each ordered position pair `(N1,N2)` and role `R`, whether `r_eval(I,S,R,P_N1,P_N2)`,
* the source state's `bool_eval` vector.

Call that the **action signature**. Two triples with the same signature cannot be told apart by
any selection, so the constraint may range over signature classes instead — and since `K1 = K2`
derives none of `sig_c_dist`/`sig_r_dist`/`sig_f_dist`, the quotiented constraint still forces
goodness to be a function of the signature, which is what the original says. The reduction is
therefore exactly equivalence-preserving, not an approximation.

This is the sound form of "keep one per equivalence class". Objects cannot be deleted — that
breaks every cardinality feature — but the (state, action) pairs whose argument tuples have
identical concept/role profiles are precisely what the encoding compares.

Implemented as `solve_datalog_sig.lp`, reachable as `--type datalog-sig`. `feature_generator`
collects each signature while it is already evaluating that concept/role extension, so nothing
is computed twice, and emits `asig/4` plus class-level `sig_*` facts. **The graph layer is
untouched**, so `frontier/2`, `pruned/2` and `safe_state` behave exactly as before.

### How far the signatures collapse

| instance | complexity | \|SA\| → signatures | quadratic term |
|---|---|---|---|
| blocks4ops-clear p004-1 | 4 | 272 → 24 (11.3×) | ~128× |
| gripper problem-3-1 | 4 | 368 → 37 (9.9×) | ~99× |
| miconic problem-2-1 | 4 | 240 → 124 (1.9×) | ~4× |

### Ground program, gripper `problem-2-1`, complexity 3

| | ground rules |
|---|---|
| `solve_datalog.lp` | 1 008 618 |
| `solve_datalog_sig.lp` | **4 997** |

**202× smaller**, and `c_distinguished`/`r_distinguished` disappear entirely.

### End to end

`-n 1`, `--max-memory` set, fixed planner seed.

| benchmark | encoding | solved | solver CPU | peak RSS | rules |
|---|---|---|---|---|---|
| gripper 2-1, 3-1, 4-1 | datalog | 3/3 | 8.35s | 511 MB | 4 |
| gripper 2-1, 3-1, 4-1 | **datalog-sig** | 3/3 | **0.79s** | **70 MB** | 4 |
| blocks4ops-clear p002-1…p005-1 | datalog | 4/4 | 0.09s | 58 MB | 2 |
| blocks4ops-clear p002-1…p005-1 | **datalog-sig** | 4/4 | **0.01s** | **49 MB** | 2 |
| miconic 2-1, 3-1, 4-1 | datalog | 1/3 | 171.9s | 7.05 GB | 11 |
| miconic 2-1, 3-1, 4-1 | **datalog-sig** | 1/3 | **138.2s** | 7.40 GB | 10 |

Gripper: 10.6× solver CPU and 7.3× memory at identical coverage and an identical policy.
Miconic is the negative case and it is worth being explicit about: its signatures collapse only
1.9×, so the quadratic term shrinks ~4× and the run is only ~1.2× faster with no coverage
change. The two runs return different but equally valid 10- and 11-rule policies — the ASP has
several optimal models and the tie is broken differently.

**The collapse is large exactly where actions are interchangeable** (gripper's balls,
blocks-clear's blocks) and small where they are not (miconic's passengers, each with its own
origin and destination). Same shape as every other reduction measured on this branch.

## R1/R2 — rule-based encoding

`trans_diff/7` existed only to be projected over `selected/1` by `distinguished/6`. Deriving
`distinguished/6` directly grounds the same rule bodies but keeps |T|² atoms rather than
|F|·|T|² + |T|². Separately, every consumer of `trans_delta` requires an alive *source*, so that
guard is free; the target stays unguarded, because a non-good transition into a dead state still
has to be separated from a good one.

blocks4ops-on `p004-01`, complexity 4: ~1 030 000 → ~530 000 ground rules (**1.94×**), with
`trans_diff` gone. `trans_delta` is unchanged on that instance — every state is alive, so R2
gains nothing there and only pays off in domains with dead ends.

`solve_d2l.lp` deliberately keeps both. It uses `trans_diff` inside a conditional literal, so the
atom has to exist, and its constraint ranges over `trans/4` from *any* state, so guarding
`trans_delta` on `alive/2` there would silently weaken it.

### Correctness

* 96 tests pass (88 existing + 8 new in `tests/test_datalog_signatures.py`).
* The new tests pin the two datalog encodings together: same policy and same feature cost on the
  `gripper` and `blocks_clear` fixtures, and the signature program must ground to under a tenth
  of the atoms.
* Rule-based: byte-identical policies on blocks4ops-fond-clear `p003-1`/`p004-1` with
  `--type state`, and the same outcome on blocks4ops-on at complexity 5, 1.8× faster.

## What is left

* **D2** — push the `not goal(I2,S2)` and `good_action` guards into the `sig_*_dist` derivations;
  check whether reflexive role position pairs (`N1 = N2`) are ever needed.
* **D3** — move policy-condition extraction out of ASP; the `#show` rules replicate the join.
* **R3** — the rule-based signature quotient (source boolean vector + delta sign vector). Measured
  at 2–30×, decaying with complexity, so worth less than D1.
* **S1** — replace pairwise separation with a bounded explicit rule set: |T|·|R|·|F| instead of
  |T|²·|F|. The only candidate that removes the quadratic rather than shrinking it, and the only
  one likely to move blocks3ops.

## Reproducing

`ground_size.py`, the throwaway harness used for the tables, is not committed. It builds
`FeaturePool(domain, [problem], ConfigHandler(type=…), max_complexity=cx)`, writes
`pool.to_clingo()` to a temp file, runs `clingo --text <solve_prog> <instance>` and counts output
lines by head predicate.
