# Enumerating equal-cost optimal models and keeping the best (`optimal_model_limit`)

## The observation

On blocks3ops one round decides the whole run. Two optimal models of cost 3, each with a single
selected element, are indistinguishable to the encoding:

| selected element | problems solved (of 95) |
|---|---|
| `c_equal_closure(on, on_g)` | 95 |
| `c_equal(on, on_g)` | 28 |

Same feature cost, same number of selected elements, both satisfy every constraint of the round.
Nothing static tells them apart, so which one comes back is whichever clingo happened to prove
optimal first — and the cheap one that generalizes badly derails the rest of the run (the loop
adds the failing problems, re-learns on a bigger training set, and ends on a large patchwork
policy; the H18 entry in `experiments-log.md` is the same failure mode).

The loop already tests every candidate policy on all problems (cheap in-loop validation,
`keep_best_policy`). Coverage is therefore the natural tie-breaker. It just never had more than
one candidate per round to apply it to.

## The mechanism

`optimal_model_limit: k` (config + `--optimal-model-limit`, default **1** = previous behaviour
exactly, no second solve and no extra validation).

1. **`Solver.enumerate_optimal(k)`** (`genfond/solver.py`). After a solve has *proved* an
   optimum, the same `Control` is re-solved with `configuration.solve.opt_mode = "optN"` and
   `configuration.solve.models = k`. `optN` re-establishes the optimum and then enumerates models
   *of that cost*; clingo flags exactly those with `Model.optimality_proven` (the one
   optimisation-phase model it re-reports has it clear), and the `models` bound counts only the
   enumerated ones. Models are deduplicated by their shown atoms, the incumbent stays first, and
   the enumeration as a whole is bounded by the same `time_limit`/`wall_deadline` a single solve
   gets. `opt_mode` and `models` are restored in a `finally`, because the lazy-pairs loop solves
   the same `Control` over and over. `k <= 1` returns the incumbent without solving at all.
2. **Lazy pairs** (`genfond/lazy_pairs.py`, `_feasible_candidates`). The enumeration runs on the
   *final* grounded program, i.e. after the loop has converged and every counterexample batch is
   in. Even then the program is only a relaxation: the incumbent is known feasible (the loop
   stops exactly when a model violates no pair), but a sibling of the same cost may be optimal
   for the relaxation alone. Every enumerated model is therefore put through the same
   counterexample scan (`LazyPairs.violated_pairs`, which reads the index and emits nothing, so
   it neither grows the ground program nor disturbs the loop's accounting) and dropped if it
   violates anything. Discard rather than repair: adding the pairs and re-solving would restart
   the loop, and the point is a bounded tie-break, not further search.
3. **`solve()`** turns each feasible model into a policy. A candidate that reaches into the
   frontier is dropped (only a zero-frontier model is a policy, and the incumbent already gave
   one, so there is nothing to expand).
4. **`solve_step`** takes a `validate` callback — passed only by `solve_iteratively`, so the
   final cost-minimization pass, which runs its own comparison, never pays for the enumeration.
   `_choose_candidate` validates every candidate and picks by **(coverage, fewer rules, clingo's
   order)**. The last two make a tie reproduce the single-model path's choice exactly; the rule
   count is the H18 lesson (a 7-rule policy out-covered the 26–40-rule ones).
5. **`solve_iteratively`** reuses the winner's validation as the round's own test and replays its
   per-problem outcomes into the `ProblemIterator`, which `_validate_candidate` deliberately
   leaves alone while the candidates are still competing (a loser must never mark a problem
   solved). So a limit of *k* costs *k* validations per successful round, not *k+1*.

### Stats

Per round: `optimalModelsEnumerated`, `optimalModelsFeasible`, `optimalModelChosenCoverage`,
`optimalModelChosenIndex`, `optimalModelCoverages` (cleared at the start of every round, so a
round that enumerates nothing does not report the previous round's numbers).
Cumulative over the run: `optimalModelsEnumeratedTotal`, `optimalModelsFeasibleTotal`,
`optimalModelTieRounds`, and **`optimalModelSwitches`** — how many rounds ended on a model other
than the one clingo returned, i.e. how often the tie-break actually changed the run.

## Local sanity (four `*-local` suites)

`--type datalog-sig -n 1 --seed 0 --add-problem-after-success --solve-time-limit 300
--max-memory 16000`, `max_plans_per_problem: 8`, `planners: {siw: {restarts: 1}}`,
`PYTHONHASHSEED=0`. Limit 1 vs. limit 3, same machine, back to back.

| suite | solved (1 / 3) | cost (1 / 3) | wall s (1 / 3) | rounds→best (1 / 3) | train (1 / 3) | enum / feasible | tie rounds | switches |
|---|---|---|---|---|---|---|---|---|
| gripper-local | 5/5 · 5/5 | 6 · 6 | 0.76 · 2.12 | 6 · 8 | 1 · 2 | 5 / 5 | 2 | 1 |
| blocks4ops-clear-local | 4/4 · 4/4 | 2 · 2 | 0.39 · 0.28 | 2 · **1** | 2 · **1** | 3 / 3 | 1 | 1 |
| miconic-local | 4/4 · 4/4 | 12 · 12 | 0.81 · 0.78 | 5 · 5 | 3 · 3 | 12 / **6** | 1 | 0 |
| delivery-local | 4/4 · 4/4 | 9 · 9 | 0.80 · 0.87 | 7 · 7 | 4 · 4 | 9 / 9 | 4 | 0 |

Solved count and cost are identical everywhere, which is the gate these suites exist for. Beyond
that:

* **The mechanism is live, not inert.** Every suite really had equal-cost siblings: 2–3 optimal
  models per round, and 1–4 rounds per run where more than one feasible candidate had to be
  scored.
* **The tie-break changes outcomes.** `blocks4ops-clear-local` is the miniature of the blocks3ops
  observation: coverages `[2, 4, 2]` on round 1, the second candidate wins, and the run converges
  in **round 1 from a single training problem** instead of round 2 from two — same cost, less
  work.
* **The lazy filter earns its place.** `miconic-local` enumerated 12 models over the run and kept
  6: half the siblings were optimal for the relaxation only. Without the re-check those would
  have become candidate policies violating separation constraints.
* **Cost.** Sub-second in absolute terms on these suites; the honest number is `gripper-local`,
  0.76 s → 2.12 s, where a switch on a coverage tie (`[5, 5]`) sent the run down a different but
  equally successful trajectory (2 training problems instead of 1). On a real suite the price is
  one extra clingo enumeration solve plus one in-loop validation per extra candidate — which is
  precisely what the `validation_*` keys were made cheap for.

## What is still open

The local suites can only show that the mechanism works and costs little; they are all solved
either way. The claim this hypothesis is actually about — that coverage-based tie-breaking picks
`c_equal_closure` over `c_equal` and holds the 95/95 policy — has to be measured on blocks3ops
(training on `p00[2-7]-*.pddl`, seeds 0–2, limit 1 vs. 3, each resulting policy scored on all 95
with `scripts/eval_policy.py`). That run belongs on the workstation.
