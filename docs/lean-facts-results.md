# Skipping unused object-level facts for the signature encoding

## What was dropped, and why it is safe

`feature_generator.node_to_clingo` writes four families of ground facts per (state, action)
combination: `c_eval(I,S,C,obj)` (one per concept, per state, per action-argument object),
`r_eval(I,S,R,o1,o2)` (one per role, per state, per ordered action-argument pair), `aname(A,Name)`
(one per action-in-state) and `aparam(A,i,obj)` (one per action parameter). `solve_datalog.lp` (and
the experimental `solve_datalog_actions.lp` / `solve_datalog_action_params.lp`) read these back
inside the pairwise separation rules -- grep confirms the only ASP consumers of the four
predicates are those three programs. `solve_datalog_sig.lp` (`--type datalog-sig`) declares
`#defined c_eval/4.` and `#defined r_eval/5.` (to silence clingo's "never grounded" warning) but no
rule in the file, in any of its `#program` parts, references `c_eval`, `r_eval`, `aname` or
`aparam` at all -- confirmed by reading the whole file, `#program` parts included.

The Python side does not need the emitted text either. `emit_action_signatures` builds
`ActionSignature` objects (`action_signatures.py`) directly from the same dlplan evaluations that
would otherwise be serialized as `c_eval`/`r_eval` facts -- `sig_concepts`/`sig_roles` are computed
from `extension`/`role_extension` (the Python `set`s returned by
`evaluate_concept_from_problem`/`evaluate_role_from_problem`) independently of whether those sets
are also written out as clingo text. `generate_datalog_policy_from_signatures` (the policy
extractor for `--type datalog-sig`) reads only `sig_action`, `bad_sig`, `f_selected`, `c_selected`,
`r_selected`, `bool_eval`, `frontier` and `good_action` off the *model*, and reconstructs rule
conditions from the `signatures` list passed in from Python -- never from `c_eval`/`r_eval`/
`aname`/`aparam`. `frontier.py` and `lazy_pairs.py` don't touch these four predicates either.

So for `--type datalog-sig`, the four families are pure overhead: grounded ground atoms with an id
each, counted by clingo, occupying memory, contributing nothing to any rule body. On a 7-block
`blocks3ops` instance they are the dominant term (`c_eval` is `n * |C| * m`, `r_eval` is
`n * |R| * m^2`, for `n` states, `m` action-argument objects per state).

## Mechanism

A new config key, `emit_object_facts` (default `true`, documented in `genfond/config/default.yaml`),
gates the four `clingo_program +=` lines in `feature_generator.node_to_clingo`. Everything that
produces `emit_object_facts`-independent side effects -- the `extension`/`role_extension`
evaluations, the `sig_concepts`/`sig_roles` bookkeeping, the stats counters used for the
"N concept evaluations (M skipped)" summary line -- is left untouched; only the text emission is
skipped. `default_datalog-sig.yaml` sets `emit_object_facts: false`; every other `--type`
(including plain `datalog`) keeps the default `true`, since `solve_datalog.lp` and the
`datalog-actions`/`datalog-action-params` variants genuinely consume these facts.

The `#defined c_eval/4.` / `#defined r_eval/5.` lines in `solve_datalog_sig.lp` are kept as-is
(harmless either way, and needed if `emit_object_facts` were ever flipped back to `true` for that
type); `aname`/`aparam` never needed a `#defined` declaration because `solve_datalog_sig.lp` never
mentions them, grounded or not.

The instance-size summary line in `iterative_solver.solve` (`"ASP instance: N lines, M MB"`) was
promoted from `DEBUG` to `INFO` so the size reduction is visible in a normal run without
`--verbose`.

## Correctness A/B

`--type datalog-sig -n 1 --seed 0 --max-memory 6000 --add-problem-after-success`, `emit_object_facts`
true (via a one-line `--config`) vs the new default (`false`):

| suite | solved (false) | solved (true) | cost, round by round |
|---|---|---|---|
| `gripper-local` (5) | 5/5 | 5/5 | identical every round: `[0,6]` |
| `blocks4ops-clear-local` (4) | 4/4 | 4/4 | identical every round: `[2]`, `[0,2]` |
| `miconic-local` (4) | 4/4 | 4/4 | identical every round: `[0,2]`, `[0,6]`, `[0,12]` |
| `delivery-local` (4) | 4/4 | 4/4 | identical every round: `[5]`, `[0,9]`, `[0,9]` |
| `blocks3ops-local` (10) | 10/10 | 10/10 | identical for 3 rounds, **diverges on round 4** |

Four of the five suites are exactly identical -- same solved counts, and the logged
`"Found policy with cost [...] for <problems> with max complexity <c>"` line is byte-identical for
every round of the iterative loop, not just the final one.

`blocks3ops-local` is not: the first three rounds match exactly (`[2]`, `[0,4]`, `[0,4]`, same
training problems added in the same order), but round 4 (training set `blocks-002-1, blocks-003-1,
blocks-004-2, blocks-005-2`) finds `cost [0,5]` at complexity 2 with `emit_object_facts: false` and
escalates to complexity 3 for `cost [0,9]` with it `true`. Tracked down: the `SIW found example
plan` calls that feed round 4 already return different plans between the two runs (different costs,
different counts), i.e. the divergence is upstream of round 4's own solve -- it originates in which
equally-optimal model round 3 happened to return. `emit_object_facts` changes which unrelated,
unreferenced ground atoms are interned, which changes clingo's internal atom ordering, which can
flip which of several tied-cost optimal models is returned; a different (same-cost) model can walk
different states in `frontier_expansion`, request different plans from SIW, and consume the shared
global RNG (`--seed`) differently, from which point the escalation path diverges. This is the exact
"same cost, not always the same model" / iterative-loop-amplification effect already documented for
the `hyp/dist-sets` and `hyp/lazy-pairs` changes (see `docs/dist-sets-results.md` and
`docs/lazy-pairs-results.md`) -- both of those changes reproduced it on `blocks3ops-local` too, for
the same underlying reason (a different but semantically-inert change to the grounded program).

To isolate whether the *encoding itself* is sound (same training set, same complexity -> same
cost) from the iterative loop's sensitivity to tie-breaking, the two blocks3ops instances the size
measurement below also uses were solved with `--one-shot` (single round, no escalation, no
planner calls in the SAT path):

| instance | complexity | solved (false) | solved (true) | cost (false) | cost (true) |
|---|---|---|---|---|---|
| p005-1 | 4 | 1/1 | 1/1 | `[0,2]` | `[0,2]` |
| p006-1 | 4 | 1/1 | 1/1 | `[0,4]` | `[0,4]` |

Same training problem, same complexity, same states (`numStates`/`numTransitions` identical in the
stats CSV), same number of lazy-pair iterations and grounded pairs, identical optimal cost. This is
the correctness property that actually matters -- a round of the solver is unchanged by
`emit_object_facts` -- and it holds. The iterative-loop divergence on `blocks3ops-local` is a
property of the outer loop's sensitivity to solver tie-breaking, already present and documented
before this change, not a defect introduced by it.

## Ground program, blocks3ops, `--one-shot --max-complexity 4 -n 1 --seed 0 --max-memory 6000 --add-problem-after-success`

| instance | `emit_object_facts` | clingo atoms | clingo rules | instance text | wall | peak mem | outcome |
|---|---|---|---|---|---|---|---|
| p005-1 | true | 341,095 | 567,188 | 98,641 lines, 7.4 MB | 12.95 s | 201 MB | solved, cost 2 |
| p005-1 | false | 259,800 | 485,893 | 10,800 lines, 0.6 MB | 12.78 s | 188 MB | solved, cost 2 |
| p006-1 | true | 1,899,049 | 2,519,367 | 1,041,989 lines, 79.6 MB | 3:42.7 | 603 MB | solved, cost 4 |
| p006-1 | false | 1,032,158 | 1,652,476 | 103,392 lines, 6.0 MB | 3:32.1 | 410 MB | solved, cost 4 |

("clingo atoms"/"clingo rules" are `solver.statistics["problem"]["lp"]["atoms"/"rules"]`,
cumulative over all lazy-pair grounding batches, from the `--stats` CSV; "peak mem" is
`/usr/bin/time -v` maximum resident set size; "instance text" is the ASP instance string built in
Python, now logged at INFO.)

p005-1: 1.3x fewer clingo atoms, 1.17x fewer rules, 12.3x smaller instance text, 6.5% less peak
memory. p006-1: 1.84x fewer clingo atoms, 1.52x fewer rules, 13.3x smaller instance text, and 32%
less peak memory (410 MB vs 603 MB) -- the reduction grows with instance size, as expected from the
`n*|C|*m` / `n*|R|*m^2` scaling of the dropped facts against `n` states. Wall clock barely moves
either way (dominated by state-space expansion, feature evaluation and the lazy-pairs solve, none
of which this change touches); the win is memory and, further out, headroom before clingo's 32-bit
ground-fact id ceiling.

## p007-1

`domains/deterministic/blocks3ops/p007-1.pddl`, `--type datalog-sig -n 1 --seed 0 --max-memory
6000 --one-shot --max-complexity 4 --add-problem-after-success`, `emit_object_facts: false`
(the new default), `timeout 40m`.

Outcome: **did not finish within 40 minutes** -- killed by the `timeout` wrapper (exit 124), not by
`bad_alloc`/`MemoryError`. Peak RSS was 3.07 GB (`/usr/bin/time -v`), comfortably under the 6 GB
cap, so it was not memory-bound either; it simply had more work queued than 40 minutes covers.
Stage timings from the run's own log:

| stage | finished at | elapsed from start |
|---|---|---|
| state-space expansion (37,633 states) | 01:11:26 | ~18.0 min |
| feature generation + pruning (uninformative/redundant) | 01:20:31 | ~27.2 min |
| instance built (1,034,680 lines, 60.3 MB) + first grounding + first solve | 01:26:59 | ~33.7 min |
| lazy pairs iteration 1 (cost `[0]`, 24.7M of 65.2M pairs violated, batch of 5000 added) | 01:27:11 | ~33.9 min |
| lazy pairs iteration 2 | killed at 40:00 | did not finish |

This is a materially different outcome from the pre-`hyp/lean-facts` baseline recorded in
`docs/lazy-pairs-results.md`, which died with `bad_alloc` on the *first* grounding at ~2013 s
(33.5 min), before clingo ever solved anything, with `emit_object_facts` implicitly `true` (the
option didn't exist yet) piling `eval`/`c_eval`/`r_eval` facts on top of the graph layer. Here, with
those four families dropped, the first grounding, first solve, and one full lazy-pairs iteration
all completed inside the 40-minute budget and peak memory stayed at half the cap. Whether the run
would go on to solve p007-1 (or die later, e.g. on a later lazy-pairs iteration's grounding) is
**unresolved** -- it was not re-run with a longer timeout, per instructions to wrap up rather than
wait further. What is established is that dropping the unused facts measurably pushed the point of
failure later (from "before the first solve" to "partway through the lazy-pairs refinement"), and
memory is no longer obviously the limiter at this size; state-space expansion and feature
generation (~28 of the 34 minutes before the first solve) now dominate, matching the
`docs/lazy-pairs-results.md` finding that beyond 6 blocks the separation layer is not the
bottleneck.

## Caveats

* **This does not touch the bottleneck the two prior hypotheses already moved past 6 blocks.**
  `docs/lazy-pairs-results.md` found that at 6+ blocks the run is dominated by state-space
  expansion and feature evaluation, not the separation layer. Dropping `c_eval`/`r_eval`/`aname`/
  `aparam` shrinks the ground program and its memory footprint, which buys headroom against the
  32-bit id ceiling and the memory cap, but it does not touch the Python-side state-space/feature
  cost, so it cannot by itself fix a case that dies before grounding starts.
* **Same cost, not always the same model, amplified by the iterative loop.** See the
  `blocks3ops-local` entry above. This is a pre-existing property of the ASP encoding's use of
  tied optimal models to drive the outer loop, not something specific to this change -- both
  `hyp/dist-sets` and `hyp/lazy-pairs` documented the identical phenomenon on the identical suite.
* `emit_object_facts: false` is safe only where the solve program genuinely does not read
  `c_eval`/`r_eval`/`aname`/`aparam` back -- i.e. `solve_datalog_sig.lp`. It must stay `true` for
  `solve_datalog.lp`, `solve_datalog_actions.lp` and `solve_datalog_action_params.lp`, which is
  why the default stays `true` and only `default_datalog-sig.yaml` overrides it.
