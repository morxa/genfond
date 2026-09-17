import enum
import logging
import sys
from typing import TYPE_CHECKING, Any, Callable, Collection, Iterator, Mapping, MutableMapping, Optional, Sequence

from pddl.action import Action
from pddl.core import Domain, Plan, Problem

from .cost_utils import feature_cost
from .ground import state_string
from .state_space_generator import State, plan_visited_states

if TYPE_CHECKING:
    # Only for the type of `on_problem_added`. `genfond.prefix_plans` imports this module (for
    # `plan_from_actions`) and pulls in the planner and the executors on the way, so importing it
    # here at runtime would be a cycle.
    from .prefix_plans import PrefixPlans

log = logging.getLogger("genfond.problem_iterator")

MAX_COST = sys.maxsize


def plan_from_actions(actions: Sequence[Action]) -> Plan:
    """Turn a sequence of ground actions into a `Plan`, the representation every plan consumer
    in this codebase uses (`siw_planner.to_pddl_plan`, `frontier._root_anchored_plan`,
    `StateSpaceGraph.__init__`, which calls `Plan.instantiate(domain)` to get the actions back).

    The caller is responsible for the plan being *root-anchored*: `StateSpaceGraph` always
    replays a plan from `problem.init`, so a trajectory that does not start there is matched
    against the wrong states. Policy executions always start at `problem.init`, so their
    trajectories are root-anchored by construction.
    """
    return Plan([(action.name, list(action.parameters)) for action in actions])


def plan_key(plan: Plan) -> tuple:
    """A hashable identity for a plan. `Plan` defines __eq__ but no __hash__."""
    return tuple((str(name), tuple(str(arg) for arg in args)) for name, args in plan.actions)


class PlanStateCoverage:
    """Tracks, per problem, the set of states already reached by that problem's example plans.

    A plan that reaches only states earlier plans of the same problem already cover cannot
    change the state space `StateSpaceGraph` builds (every state it would revive or expand is
    already reachable), so keeping it around is pure waste: one more plan for the solver to
    replay, one more entry in every log line, no effect on what is expressible. `add()` reports
    whether a plan actually earns its keep, so the caller can drop the ones that do not.
    """

    def __init__(self, domain: Domain, problems: Mapping[str, Problem]):
        self.domain = domain
        self.problems = problems
        self.covered: dict[str, set[State]] = dict()

    def add(self, problem_name: str, plan: Plan) -> bool:
        """Register `plan`'s visited states for `problem_name`; return whether any were new."""
        visited = plan_visited_states(self.domain, self.problems[problem_name], plan)
        covered = self.covered.setdefault(problem_name, set())
        new = not visited <= covered
        covered |= visited
        return new


class Result(enum.Enum):
    UNKNOWN = 0
    SUCCESS = 1
    NO_SOLUTION = 2
    OUT_OF_RESOURCES = 3
    # A model was found, but it relies on transitions into unexpanded frontier states, so it
    # is not a policy yet. The caller expands those states and the same configuration is
    # retried with the enlarged plan set.
    FRONTIER = 4
    # The solve ran out of its wall-clock budget (`solve_time_limit`) without finding a model.
    # Unlike NO_SOLUTION this is *not* a refutation -- the round says nothing about whether a
    # policy exists at this complexity -- so it must not set `refuted_complexity`. Escalation
    # otherwise proceeds exactly as after NO_SOLUTION.
    TIMEOUT = 5


class LastStep(enum.Enum):
    START = 0
    INC_PLANS = 1
    INC_COMPLEXITY = 2
    EXPAND_FRONTIER = 3


class ProblemIterator:

    def __init__(
        self,
        problems: list[Problem],
        config: Mapping,
        plans: Optional[Mapping[str, Iterator[Plan]]] = None,
        plan_coverage: Optional[PlanStateCoverage] = None,
        on_problem_added: Optional[Callable[[Problem], "PrefixPlans"]] = None,
    ):
        self.problems = problems
        self.config = config
        self.plan_iterators = plans
        # None (the default, and what every existing test passes) disables the dedupe entirely:
        # every plan the iterator is handed is kept, exactly as before this was added.
        self.plan_coverage = plan_coverage
        # Called once for each problem that joins the training set, before that problem's first
        # round, and expected to return example plans to seed it with (policy-prefix plans, H30
        # -- see `genfond.prefix_plans`). The iterator knows nothing about how they are built:
        # it only needs somewhere to put them. None (the default, and what every existing test
        # passes) means no problem is ever seeded with anything.
        self.on_problem_added = on_problem_added

    def __iter__(self) -> "ProblemIterator":
        self.active_problems: list[Problem] = []
        self.active_plans: MutableMapping[str, Plan] = dict()
        # Problems for which a max_plans_per_problem cap has already been logged, so the log
        # line appears once per problem instead of once per round for the rest of the run.
        self._plan_cap_logged: set[str] = set()
        # How many of `active_plans[name]` are policy-conformant plans (see
        # `record_policy_plans`). They are stored at the *front* of the list and are exempt
        # from both `max_plans_per_problem` and the `min_number_of_plans` floor, so both counts
        # subtract this. Empty unless `policy_conformant_plans` is on, which makes both
        # subtractions no-ops.
        self.policy_plan_counts: dict[str, int] = dict()
        # Problems the `on_problem_added` hook has already run for, so a problem that leaves the
        # training set (`unselect_problems`) and rejoins it later is not seeded twice.
        self._prefix_plans_done: set[str] = set()
        # How many policy-prefix plans were added over the run, and for how many problems the
        # hook ran and came back empty-handed. Read out by `iterative_solver` as the
        # `prefixPlansAdded` / `prefixPlanFailures` stats.
        self.prefix_plans_added = 0
        self.prefix_plan_failures = 0
        # Whether policy-conformant plans were added since the last success. Such an addition
        # changes the state space of problems already in the training set, which breaks the
        # monotonicity argument `_add_next_problem` uses to carry `succ_complexity - 1` over as
        # a refutation.
        self.plans_added_since_success = False
        self.selected_states: dict[str, set[State]] = dict()
        self.new_states: dict[str, set[State]] = dict()
        self.dead_states: dict[str, set[State]] = dict()
        self.frontier_expansions = 0
        self.frontier_progress = False
        # How many plans the frontier expansion found but could not keep because their problem
        # was already at `max_plans_per_problem`. Read out by `iterative_solver` as the
        # `frontierPlansDropped` stat; every one of them is a planner call that bought nothing,
        # which is what `plan_cap_reached` lets the caller avoid up front.
        self.frontier_plans_dropped = 0
        self.all_features = False
        self.last_step = LastStep.START
        self.complexity = self.config["min_complexity"]
        self.last_result = Result.SUCCESS
        self.succ_complexity = self.complexity
        self.active_problems_solved = True
        self.max_cost = MAX_COST
        # Highest complexity level refuted for the *current* state space and the current
        # `max_cost`: no solution exists that uses only features of at most that complexity.
        # `min_complexity - 1` encodes the config's own assertion that nothing below
        # `min_complexity` is worth trying. See `enforce_highest_complexity`.
        self.refuted_complexity = self.config["min_complexity"] - 1
        # The complexity the sweep has to climb back to before another example plan is added.
        # Only ever differs from `complexity` when `reset_complexity_on_state_space_change`
        # restarts the sweep; without it the guard is vacuous.
        self.sweep_target = self.complexity
        self.solved = {problem.name: False for problem in self.problems}
        return self

    def enforce_highest_complexity(self) -> bool:
        """Whether the solver may require a selected feature of at least `complexity`.

        `min_feature_complexity(c)` is justified by exactly one fact: that complexity `c-1`
        was refuted over the full feature pool. That refutation is a statement about a
        specific ASP instance, so it does not survive a change of the state space -- adding an
        example plan or a dead end can make a *simpler* policy possible, and the constraint
        would exclude it. Every branch that changes the state space therefore calls
        `_invalidate_refutations`.
        """
        return self.refuted_complexity >= self.complexity - 1

    def _invalidate_refutations(self) -> None:
        """Record that the state space changed, so no complexity level is refuted any more.

        With `reset_complexity_on_state_space_change` the sweep also restarts at
        `min_complexity`, which re-establishes the refutations on the way back up and lets the
        solver find a policy that only became expressible in the enlarged state space. Without
        it the search stays where it is and simply drops the enforcement until the current
        level has been refuted again, which is cheaper but never looks below.
        """
        self.refuted_complexity = self.config["min_complexity"] - 1
        if self.config["reset_complexity_on_state_space_change"]:
            self.sweep_target = max(self.sweep_target, self.complexity)
            self.complexity = self.config["min_complexity"]

    def _plan_cap_reached(self, problem_name: str) -> bool:
        """Whether `problem_name` already holds `max_plans_per_problem` example plans.

        null (the default) never caps. Logged once per problem the first time it is hit, not on
        every later round that would otherwise have tried to add another plan.
        """
        cap = self.config.get("max_plans_per_problem")
        if cap is None:
            return False
        # Policy-conformant plans do not count towards the cap: the cap exists to stop runaway
        # *search* growth (INC_PLANS, frontier expansion), and a trajectory a working policy
        # actually took is the most valuable plan there is -- capping those away is exactly
        # what would reintroduce the infeasibility this mechanism removes.
        reached = len(self.active_plans.get(problem_name, [])) - self.policy_plan_counts.get(problem_name, 0) >= cap
        if reached and problem_name not in self._plan_cap_logged:
            self._plan_cap_logged.add(problem_name)
            log.info(
                "Problem %s reached max_plans_per_problem=%d; no more example plans will be added to it",
                problem_name,
                cap,
            )
        return reached

    def plan_cap_reached(self, problem_name: str) -> bool:
        """Whether `problem_name` can still take another example plan.

        The public face of `_plan_cap_reached`, for callers that want to know *before* doing the
        work whose result would be dropped -- `solve_iteratively` skips the planner call for a
        frontier state whose problem is already capped.
        """
        return self._plan_cap_reached(problem_name)

    def _accept_plan(self, problem_name: str, plan: Plan) -> bool:
        """Whether `plan` earns its keep: reaches a state not already covered for this problem.

        With no coverage tracker (the default in every existing caller and test) every plan is
        accepted, so this is a no-op unless the caller opted in.
        """
        if self.plan_coverage is None:
            return True
        if self.plan_coverage.add(problem_name, plan):
            return True
        log.info(
            "Discarding an example plan for %s: it reaches no state beyond the existing example plans",
            problem_name,
        )
        return False

    def record_policy_plans(self, plans: Mapping[str, Plan]) -> tuple[int, int]:
        """Take the trajectories a validated policy took on the problems it solved.

        A policy that solves a problem demonstrates a trajectory through that problem's state
        space; unless an example plan already covers it, the plan-restricted `StateSpaceGraph`
        does not contain that trajectory, and the very policy that produced it is therefore
        *infeasible* on the ASP instance of every later round. Recording it as an example plan
        keeps it feasible, so the next round (which typically adds a failing problem to the
        training set) can still select the policy that already solved everything else instead of
        having to jump to a much more expensive patchwork.

        Plans are keyed by problem name and recorded for *every* problem the policy solved, not
        only the ones currently in the training set: a problem gets added to the training set
        precisely when it is still unsolved, and the plans that matter for it are the ones its
        already-solved neighbours contribute.

        Placement and caps:

        - Policy plans go in *front* of a problem's existing example plans and are counted in
          `policy_plan_counts`, which `_plan_cap_reached` and the `min_number_of_plans` floor
          both subtract. They are therefore never dropped by `max_plans_per_problem`, and never
          suppress the planner's own plans either.
        - They are deduped both against the plans the problem already has (`plan_key`) and by
          the state coverage tracker (`_accept_plan`), so a trajectory that reaches no state the
          existing plans do not already cover is discarded: it cannot change the state space.

        Returns `(plans added, problems affected)`.
        """
        added = 0
        problems_affected = 0
        for problem_name, plan in plans.items():
            if not plan.actions:
                # The goal already held at `problem.init`; nothing to demonstrate.
                continue
            active = self.active_plans.setdefault(problem_name, [])
            if plan_key(plan) in {plan_key(known) for known in active}:
                continue
            if not self._accept_plan(problem_name, plan):
                continue
            self.active_plans[problem_name] = [plan] + active
            self.policy_plan_counts[problem_name] = self.policy_plan_counts.get(problem_name, 0) + 1
            added += 1
            problems_affected += 1
        if added:
            # The state space of the next round changed, so nothing is refuted any more -- the
            # same reasoning as for INC_PLANS and frontier expansion.
            self._invalidate_refutations()
            self.plans_added_since_success = True
        return added, problems_affected

    def _seed_prefix_plans(self, problem: Problem) -> None:
        """Run the `on_problem_added` hook for a problem that just joined the training set.

        The plans it returns (policy-prefix plans, H30) are treated exactly like the
        policy-conformant plans of `record_policy_plans`: stored in *front* of the problem's own
        example plans, counted in `policy_plan_counts` so neither `max_plans_per_problem` nor the
        `min_number_of_plans` floor applies to them, and deduped by plan identity and state
        coverage. They are the reason the problem's instance can express the policy that was
        already almost right, so capping them away would defeat the mechanism.

        Called *after* `_add_next_problem` has set `refuted_complexity`, because adding plans
        drops that bound again: the state space this round is solved over is not the one the
        refutation was established over (the same reasoning as `_invalidate_refutations`, which
        is not called directly here only because its optional complexity restart would undo the
        `complexity = succ_complexity` this branch has just set).
        """
        if self.on_problem_added is None or problem.name in self._prefix_plans_done:
            return
        self._prefix_plans_done.add(problem.name)
        result = self.on_problem_added(problem)
        active = self.active_plans.setdefault(problem.name, [])
        known = {plan_key(plan) for plan in active}
        accepted = []
        for plan in result.plans:
            if not plan.actions or plan_key(plan) in known:
                continue
            if not self._accept_plan(problem.name, plan):
                continue
            known.add(plan_key(plan))
            accepted.append(plan)
        if not accepted:
            if result.attempted:
                self.prefix_plan_failures += 1
            return
        self.active_plans[problem.name] = accepted + active
        self.policy_plan_counts[problem.name] = self.policy_plan_counts.get(problem.name, 0) + len(accepted)
        self.prefix_plans_added += len(accepted)
        self.refuted_complexity = self.config["min_complexity"] - 1
        log.info(
            "Added %d policy-prefix plan(s) for %s (prefix length %d, backoff %d)",
            len(accepted),
            problem.name,
            result.prefix_length,
            result.backoff,
        )

    def record_frontier_expansion(
        self, plans: Mapping[str, list[Plan]], dead_states: Mapping[str, set[State]]
    ) -> None:
        """Take the result of expanding the frontier states the last model relied on.

        Sets `frontier_progress`, which gates the retry: without new plans and without new
        dead ends the next round would be identical, so we must fall through to the normal
        escalation ladder instead of looping forever.
        """
        self.frontier_expansions += 1
        self.frontier_progress = False
        for problem_name, new_plans in plans.items():
            active = self.active_plans.setdefault(problem_name, [])
            known = {plan_key(plan) for plan in active}
            for index, plan in enumerate(new_plans):
                if self._plan_cap_reached(problem_name):
                    # Capped: no more plans go into this problem this round, regardless of
                    # how many more the planner proposed for it. Each of those cost a planner
                    # call, so say how many were thrown away rather than dropping them silently.
                    dropped = len(new_plans) - index
                    self.frontier_plans_dropped += dropped
                    log.info(
                        "Dropping %d frontier plan(s) for %s: it is already at" " max_plans_per_problem=%s",
                        dropped,
                        problem_name,
                        self.config.get("max_plans_per_problem"),
                    )
                    break
                if plan_key(plan) in known:
                    # The planner is deterministic, so a frontier state that the new plan
                    # fails to expand yields the same plan every round. Re-adding it is not
                    # progress; counting it as such spins this loop until the expansion
                    # budget runs out.
                    log.debug("Frontier plan for %s is already an example plan, ignoring it", problem_name)
                    continue
                if not self._accept_plan(problem_name, plan):
                    continue
                known.add(plan_key(plan))
                active.append(plan)
                self.frontier_progress = True
        for problem_name, states in dead_states.items():
            known_dead = self.dead_states.setdefault(problem_name, set())
            if states - known_dead:
                self.frontier_progress = True
            known_dead |= states
        if self.frontier_progress:
            # New plans and new dead ends both change the state space the next round is
            # solved over, so the complexity levels refuted so far no longer apply.
            self._invalidate_refutations()

    def set_last_result(self, result: Result, cost: Optional[tuple[int]] = None, optimal: bool = True) -> None:
        """Record how the last round ended.

        `optimal` says whether the round's model was *proved* to be of minimal cost. It is only
        ever False under a `solve_time_limit`, where a solve can be cancelled while it still
        holds a merely feasible model. Such a model is a valid policy -- every constraint is
        satisfied -- but its cost is an upper bound, not the optimum, so it refutes nothing:
        a cheaper policy may well exist at this very complexity.
        """
        self.last_result = result
        # Only a round over the full feature pool refutes a complexity level; the restricted
        # generators are a subset, so their failure says nothing about the unrestricted ones.
        full_feature_pool = self.all_features or not self.config["use_unrestricted_features"]
        if result == Result.NO_SOLUTION and full_feature_pool:
            self.refuted_complexity = max(self.refuted_complexity, self.complexity)
        # A frontier model's feature cost is artificially low because it was allowed to assume
        # unexpanded states are solvable. Tightening max_cost or marking problems solved from
        # it would make every later round unsatisfiable, so Result.FRONTIER changes nothing.
        if result == Result.SUCCESS:
            assert cost
            self.active_problems_solved = True
            # `succ_complexity` is (re)established below over the state space this round was
            # solved over; policy plans recorded after this point invalidate it again.
            self.plans_added_since_success = False
            # Keeping this as a *preference* is sound either way: the next round is asked to
            # beat the cost we actually achieved, which is a real upper bound whether or not
            # it is the optimum. Only the refutation below depends on optimality. `max_cost`
            # bounds `limit_feature_cost`, i.e. the feature/concept/role complexity sum, not
            # the raw cost vector -- with minimize_good_signatures="below" or
            # minimize_selected_count="below" that is no longer cost[-1] (see
            # cost_utils.feature_cost), so it must be extracted the same way here.
            self.max_cost = (
                feature_cost(
                    cost,
                    self.config.get("minimize_good_signatures", "none"),
                    self.config.get("minimize_selected_count", "none"),
                )
                - 1
            )
            if full_feature_pool and optimal:
                # clingo minimizes the feature cost, so the extracted value is optimal for this
                # pool: nothing at this complexity beats the new `max_cost`. That refutes the level
                # just as an UNSAT would, and keeps the `max_cost < complexity` short circuit
                # in `iterative_solver.solve` available for the rounds that follow. A model
                # that was merely the best found before the time budget ran out proves no such
                # thing, which is why `optimal` guards this and not `max_cost` above.
                self.refuted_complexity = max(self.refuted_complexity, self.complexity)
            self.succ_complexity = self.complexity
            # self.solved = {
            #     problem.name: True if problem in self.active_problems else False
            #     for problem in self.problems
            # }
            self.new_states.clear()
            for problem in self.problems:
                if problem.name in self.selected_states and problem not in self.active_problems:
                    del self.selected_states[problem.name]

    def frontier_budget_left(self) -> bool:
        """Whether another frontier expansion may still be spent on this run."""
        return self.frontier_expansions <= self.config["max_frontier_expansions"]

    def set_solved(self, problem: Problem, solved: bool = True):
        self.solved[problem.name] = solved

    def get_unsolved_problems(self) -> list[Problem]:
        return [problem for problem in self.problems if not self.solved[problem.name]]

    def _next_addable_problem(self) -> Optional[Problem]:
        """The next unsolved problem that is not already in the training set, if any."""
        return next(
            (
                problem
                for problem in self.problems
                if not self.solved[problem.name] and problem not in self.active_problems
            ),
            None,
        )

    def _add_next_problem(self) -> None:
        """Add the next unsolved problem to the training set and reset the sweep for it.

        Shared by the default escalation ladder (added only once complexity has climbed back
        to max_cost <= complexity or max_complexity) and `add_problem_after_success` (added
        right after a success, skipping the climb). Everything below must stay identical
        between the two call sites.
        """
        self.all_features = False
        self.max_cost = MAX_COST
        self.active_problems_solved = False
        self.complexity = self.succ_complexity
        self.sweep_target = self.complexity
        self.last_step = LastStep.START
        next_problem = self._next_addable_problem()
        assert next_problem is not None
        if (
            self.config["unselect_problems"]
            and self.active_problems
            and self.problems.index(next_problem)
            > max([self.problems.index(problem) for problem in self.active_problems])
        ):
            self.active_problems = [next_problem]
            # The training set is replaced, not extended, so nothing carries over.
            self.refuted_complexity = self.config["min_complexity"] - 1
        else:
            self.active_problems.append(next_problem)
            if self.plans_added_since_success:
                # Policy-conformant plans were added *after* the success that established
                # `succ_complexity`, so that refutation was made over a smaller state space
                # than the next round will be solved over. The monotonicity argument below
                # only covers adding an instance, not enlarging an existing one's state
                # space, so the bound does not survive -- same reasoning as
                # `_invalidate_refutations`.
                self.refuted_complexity = self.config["min_complexity"] - 1
            else:
                # Adding an instance is monotone: a selection that solves the larger set also
                # solves every subset, so "no solution below `succ_complexity`" carries over
                # and the sweep resumes there instead of at `min_complexity`. That bound is
                # unconditional -- it was established before the success tightened `max_cost`,
                # which is reset here anyway.
                self.refuted_complexity = self.succ_complexity - 1
        if self.plan_iterators:
            # setdefault, not `= []`: a problem may already hold policy-conformant plans
            # recorded while it was still outside the training set (`record_policy_plans`),
            # and those are the whole point of the mechanism. Without any, this is `= []`.
            self.active_plans.setdefault(next_problem.name, [])
            # min_number_of_plans is a deliberate floor, not runaway growth, so it is exempt
            # from max_plans_per_problem: the cap only stops *further* growth from INC_PLANS or
            # frontier expansion, via _plan_cap_reached there.
            # Policy-conformant plans are excluded from the floor too: they are not planner
            # samples, and letting them suppress the planner's own diversity would defeat
            # min_number_of_plans.
            while (
                len(self.active_plans[next_problem.name]) - self.policy_plan_counts.get(next_problem.name, 0)
                < self.config["min_number_of_plans"]
            ):
                next_plan = next(self.plan_iterators[next_problem.name], None)
                if next_plan is None:
                    break
                # This initial batch is not deduped by state coverage -- min_number_of_plans is
                # a deliberate floor, not runaway growth -- but the coverage tracker still needs
                # to know about these plans so later INC_PLANS/frontier additions are compared
                # against the true baseline.
                if self.plan_coverage is not None:
                    self.plan_coverage.add(next_problem.name, next_plan)
                self.active_plans[next_problem.name].append(next_plan)
        # Last, so the prefix plans are deduped against the floor batch that was just drawn (a
        # plan reaching no state those already cover cannot change the state space) and so the
        # refutation bound this method just set is dropped again if any plan is actually added.
        self._seed_prefix_plans(next_problem)

    def __next__(self) -> Mapping[str, Any]:
        assert self.last_result != Result.UNKNOWN, "You must set the result of the last problem before calling next"
        log.debug(
            f"last result: {self.last_result.name}, all features: {self.all_features}, complexity: {self.complexity}"
        )
        if self.last_result == Result.FRONTIER and self.frontier_progress and self.frontier_budget_left():
            # Retry the exact same configuration; only the plan and dead-end sets grew.
            self.last_step = LastStep.EXPAND_FRONTIER
        elif (
            self.config["add_problem_after_success"]
            and self.active_problems_solved
            and self._next_addable_problem() is not None
        ):
            # Skip the complexity/plan/feature climb entirely and add the next problem right
            # after a success. See the config comment on `add_problem_after_success`.
            self._add_next_problem()
        elif (
            (self.last_step == LastStep.INC_COMPLEXITY or self.complexity == self.config["max_complexity"])
            # A restarted sweep has to climb *past* the level the last plan was added at
            # before the next one is added. With ">=" the restart and the addition alternate
            # at one fixed level and the search never reaches the higher complexities at all;
            # this way each plan is added one level deeper, as it is without the restart. The
            # guard is vacuous when the sweep is not restarted, since `sweep_target` then
            # stays at `min_complexity` and `last_step == INC_COMPLEXITY` implies a higher
            # complexity. At `max_complexity` there is no deeper level to reach, so plans are
            # added there on every round, exactly as before.
            and (self.complexity > self.sweep_target or self.complexity == self.config["max_complexity"])
            and self.active_problems
            and not self.active_problems_solved
            and self.last_result != Result.OUT_OF_RESOURCES
            and self.plan_iterators
            and (
                # Find the next plan for an active problem that is not yet solved, is not
                # capped at max_plans_per_problem, and whose drawn plan actually adds a state
                # beyond what this problem's existing example plans already cover.
                found := next(
                    (
                        (k.name, v)
                        for k in self.active_problems
                        if not self.solved[k.name]
                        and not self._plan_cap_reached(k.name)
                        and (v := next(self.plan_iterators[k.name], None)) is not None
                        and self._accept_plan(k.name, v)
                    ),
                    None,
                )
            )
        ):
            problem, plan = found
            self.active_plans.setdefault(problem, []).append(plan)
            self.all_features = False
            self.max_cost = MAX_COST
            self._invalidate_refutations()
            self.last_step = LastStep.INC_PLANS
        elif (
            self.active_problems
            and self.last_result != Result.OUT_OF_RESOURCES
            and not self.all_features
            and self.config["use_unrestricted_features"]
        ):
            self.all_features = True
        elif (
            self.active_problems
            and self.last_result != Result.OUT_OF_RESOURCES
            and (self.all_features or not self.config["use_unrestricted_features"])
            and self.complexity < self.config["max_complexity"]
            and self.max_cost > self.complexity
        ):
            self.all_features = False
            self.complexity += 1
            self.last_step = LastStep.INC_COMPLEXITY
        elif (
            self.active_problems_solved
            and any(not solved for solved in self.solved.values())
            and self._next_addable_problem() is not None
        ):
            self._add_next_problem()
        else:
            raise StopIteration
        log.debug(
            f'Next set: {", ".join([p.name for p in self.active_problems])},'
            f" complexity={self.complexity},"
            f" all_features={self.all_features},"
            f' max_cost={self.max_cost if self.max_cost < MAX_COST else "MAX_COST"},'
            f" |selected_states|={len(self.selected_states)} states,"
            f" enforce_highest_complexity={self.enforce_highest_complexity()},"
            f" last_step={self.last_step.name}"
        )
        return {
            "active_problems": self.active_problems,
            "complexity": self.complexity,
            "all_features": self.all_features,
            "max_cost": self.max_cost,
            "example_plans": self.active_plans,
            "dead_states": self.dead_states,
            "allow_frontier": self.frontier_budget_left(),
            "enforce_highest_complexity": self.enforce_highest_complexity(),
        }


class OneShotProblemIterator(ProblemIterator):

    def __iter__(self, *args, **kwargs) -> "OneShotProblemIterator":
        super().__iter__(*args, **kwargs)
        self.active_problems = list(self.problems)
        self.called = False
        self.all_features = self.config["use_unrestricted_features"]
        self.complexity = self.config["max_complexity"]
        if self.config["use_selected_states"]:
            self.new_states = {problem.name: {problem.init} for problem in self.active_problems}
        return self

    def __next__(self) -> Mapping[str, Any]:
        if self.called:
            raise StopIteration
        self.called = True
        # Must match ProblemIterator.__next__: solve_step takes example_plans, not
        # selected_states, so returning the latter used to make --one-shot raise TypeError.
        return {
            "active_problems": self.active_problems,
            "complexity": self.complexity,
            "all_features": self.all_features,
            "max_cost": self.max_cost,
            "example_plans": self.active_plans,
            "dead_states": self.dead_states,
            "allow_frontier": self.frontier_budget_left(),
            # A single round at max complexity refutes nothing beforehand.
            "enforce_highest_complexity": False,
        }
