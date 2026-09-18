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

# How many plans beyond the number actually wanted `resample_planner_plans` may pull from a
# problem's plan stream before giving up on finding new ones. Pulling a plan runs real search, so
# the resample must not turn into an unbounded planner loop when the stream keeps handing back
# plans the problem already has.
RESAMPLE_DRAW_LIMIT_FACTOR = 4
RESAMPLE_DRAW_LIMIT_MARGIN = 8


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

    def reset(self, problem_name: str) -> None:
        """Forget everything covered for `problem_name`.

        The coverage set only ever grows, which is right as long as plans are only ever added.
        `ProblemIterator.resample_planner_plans` *removes* plans, and the states only those plans
        reached are then no longer covered by anything -- leaving them in would make the tracker
        reject every replacement plan as redundant. The caller re-adds the plans it kept.
        """
        self.covered.pop(problem_name, None)


class MaxComplexityContinuation:
    """What `ProblemIterator` needs from its caller to keep going past an exhausted sweep (H33).

    The iterator itself decides *whether* the run may continue once the complexity sweep is
    exhausted and *how* (resample, or add the next unsolved problem). The two things it cannot
    know on its own are whether the run still has wall-clock budget and whether H32's resample
    is still available -- both live in `iterative_solver`. They arrive here as plain callables
    so the iterator stays testable without the solver: the default instance says "time left, no
    resample", which is exactly what a bare iterator should assume.
    """

    def __init__(
        self,
        time_left: Optional[Callable[[], bool]] = None,
        resample_available: Optional[Callable[[], bool]] = None,
        resample: Optional[Callable[[], tuple[int, int]]] = None,
    ):
        self._time_left = time_left
        self._resample_available = resample_available
        self._resample = resample

    def time_left(self) -> bool:
        """Whether starting another round is still worthwhile (no deadline hit, no stop pending)."""
        return self._time_left() if self._time_left is not None else True

    def resample_available(self) -> bool:
        """Whether a resample can still be spent (`resample_on_stall` on, budget left)."""
        return self._resample is not None and (self._resample_available is None or self._resample_available())

    def resample(self) -> tuple[int, int]:
        """Draw a new sample of planner plans; returns `(plans replaced, problems affected)`."""
        assert self._resample is not None, "resample() is only called when resample_available() said so"
        return self._resample()


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
        # A MutableMapping rather than a Mapping because `resample_planner_plans` replaces a
        # problem's exhausted stream with a fresh one; every caller already passes a dict.
        plans: Optional[MutableMapping[str, Iterator[Plan]]] = None,
        plan_coverage: Optional[PlanStateCoverage] = None,
        on_problem_added: Optional[Callable[[Problem], "PrefixPlans"]] = None,
        continuation: Optional[MaxComplexityContinuation] = None,
    ):
        self.problems = problems
        self.config = config
        self.plan_iterators = plans
        # What `_continue_past_max_complexity` (H33) may reach for once the complexity sweep is
        # exhausted. The default instance grants unlimited time and no resample, so an iterator
        # built without one continues only by adding the next unsolved problem.
        self.continuation = continuation if continuation is not None else MaxComplexityContinuation()
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
        # How often the run was kept alive past an exhausted complexity sweep (H33). Read out by
        # `iterative_solver` as the `maxComplexityContinuations` stat.
        self.max_complexity_continuations = 0
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

    def resample_planner_plans(
        self, fresh_plan_iterator: Optional[Callable[[Problem], Iterator[Plan]]] = None
    ) -> tuple[int, int]:
        """Throw away the *planner's* example plans of every training problem and draw new ones.

        Which policy the loop converges to is decided by the example plans it happens to have
        sampled: on blocks3ops with `--type datalog-sig`, an unlucky early sample sends 2 of 6
        seeds into a patchwork policy (feature cost 13-16) whose coverage then never improves
        again, while the other 4 reach 95/95 within minutes. This is the in-loop escape hatch --
        when `solve_iteratively` sees the best coverage stall for `stall_rounds` rounds it calls
        this and the search continues over a different sample.

        What is kept and what is replaced:

        - The plans in *front* of a problem's list, counted in `policy_plan_counts`, are the
          policy-conformant trajectories of H27 and the policy-prefix plans of H30 (and are the
          plans that carry information about the best policy so far). They are never discarded:
          they are not samples, and the whole point of both mechanisms is that they stay in the
          state space.
        - Everything behind them was drawn from the planner -- the `min_number_of_plans` floor,
          INC_PLANS additions, frontier plans. Those are the sample, and they are what is
          replaced, one for one: a problem ends up with exactly as many planner plans as it had,
          so neither `max_plans_per_problem` nor the `min_number_of_plans` floor can be violated
          by a resample.

        Where the replacements come from: first from the problem's own plan stream, simply
        continued -- the planner yields lazily and dedupes within a stream, so the next plans it
        has to offer are new ones that cost nothing extra to reach. Only when that stream runs
        dry is `fresh_plan_iterator` asked for a new one (reseeded by the caller; see
        `iterative_solver._fresh_plan_iterator` for why a fresh seed is not enough on its own),
        and the problem's stream is then replaced by it so later INC_PLANS draws continue there.
        A problem whose stream can offer nothing new keeps the plans it has.

        Bookkeeping, exactly what a plan-set change requires and nothing more:

        - `_invalidate_refutations`: the state space changed, so no complexity level is refuted
          any more -- the same reasoning as INC_PLANS, `record_policy_plans` and the frontier
          expansion. `resample_reset_complexity` additionally restarts the sweep at
          `min_complexity`, mirroring `reset_complexity_on_state_space_change`.
        - `plans_added_since_success`: the plan set changed after the success that established
          `succ_complexity`, so `_add_next_problem` must not carry that bound over.
        - `max_cost` is deliberately **left alone**. A resample is not permission to accept a
          worse policy: the best policy so far still stands, and the next round is still asked to
          beat its cost.
        - No problem is added, and `last_step` is untouched: the next round runs the same
          training set at the current complexity over the new sample.

        Returns `(plans replaced, problems affected)`.
        """
        if not self.plan_iterators:
            return 0, 0
        replaced = 0
        problems_affected = 0
        for problem in self.active_problems:
            name = problem.name
            stream = self.plan_iterators.get(name)
            if stream is None:
                continue
            existing = list(self.active_plans.get(name, []))
            kept_count = self.policy_plan_counts.get(name, 0)
            kept, discarded = existing[:kept_count], existing[kept_count:]
            if not discarded:
                # Nothing was sampled for this problem (only policy/prefix plans, or no plans at
                # all), so there is nothing to resample.
                continue
            wanted = len(discarded)
            # A replacement must differ from what the problem keeps *and* from what is being
            # thrown away -- redrawing the discarded plans is not a resample.
            excluded = {plan_key(plan) for plan in kept} | {plan_key(plan) for plan in discarded}
            drawn: list[Plan] = []
            fresh_used = False
            pulls_left = RESAMPLE_DRAW_LIMIT_FACTOR * wanted + RESAMPLE_DRAW_LIMIT_MARGIN
            while len(drawn) < wanted and pulls_left > 0:
                plan = next(stream, None)
                if plan is None:
                    if fresh_used or fresh_plan_iterator is None:
                        break
                    fresh_used = True
                    stream = fresh_plan_iterator(problem)
                    continue
                pulls_left -= 1
                key = plan_key(plan)
                if key in excluded:
                    continue
                excluded.add(key)
                drawn.append(plan)
            if fresh_used:
                self.plan_iterators[name] = stream
            if not drawn:
                log.info("Resampling found no new example plan for %s; keeping its %d existing one(s)", name, wanted)
                continue
            self.active_plans[name] = kept + drawn
            if self.plan_coverage is not None:
                # The discarded plans' states are no longer covered by anything, so the tracker
                # has to be rebuilt from the plans that remain rather than merely extended.
                self.plan_coverage.reset(name)
                for plan in self.active_plans[name]:
                    self.plan_coverage.add(name, plan)
            # The problem may have dropped below max_plans_per_problem again, so let the cap be
            # reported afresh if it is hit once more.
            self._plan_cap_logged.discard(name)
            replaced += len(drawn)
            problems_affected += 1
        if replaced:
            self._invalidate_refutations()
            if self.config.get("resample_reset_complexity", False):
                self.sweep_target = max(self.sweep_target, self.complexity)
                self.complexity = self.config["min_complexity"]
            self.plans_added_since_success = True
        return replaced, problems_affected

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

    def _restart_sweep(self) -> None:
        """Send the complexity sweep back to `min_complexity` after a max-complexity continuation.

        Restarting is sound, never merely cheap: the feature pool at complexity `c` contains
        every feature of complexity `< c`, so every policy reachable from the level the sweep had
        climbed to is also reachable from `min_complexity` on the way back up -- the restart can
        only cost time, never a policy. What it buys is size: the low-complexity instances are
        far smaller (roles alone contribute `n*m^2` grounded values per state), and a continuation
        exists precisely because the big ones at the top of the sweep did not produce a policy --
        several of them because they could not be grounded at all.

        `sweep_target` is raised to the level the sweep had reached, the same bookkeeping
        `_invalidate_refutations` does for `reset_complexity_on_state_space_change`: without it a
        restarted sweep would add an example plan at `min_complexity` on the very next round
        (`last_step == INC_COMPLEXITY` still holds from the climb) and the two would alternate at
        one fixed level instead of the sweep ever climbing again. `last_step` goes back to START
        for the same reason.

        Refutations are deliberately *not* touched here. Each caller has already done what its
        own state-space change requires: the resample invalidated them (`resample_planner_plans`
        -> `_invalidate_refutations`), and `_add_next_problem` set the monotone bound it is
        entitled to keep.
        """
        self.sweep_target = max(self.sweep_target, self.complexity)
        self.complexity = self.config["min_complexity"]
        self.last_step = LastStep.START

    def _continue_past_max_complexity(self) -> bool:
        """Keep the run alive when the complexity sweep is exhausted (H33). Returns whether it is.

        Reaching `max_complexity` on the current training set is not the end of what the run can
        try, it is only the end of one sweep. Measured on logistics_dp (47 problems,
        `--type datalog-sig`, 4 h graceful budget): the loop exhausted the sweep after 28 rounds /
        84 min with 15/47 solved and stopped with `failureReason=maxcomplexity`, leaving 2.6 h
        unused, while slower variants that never reached the top of the sweep got 23/47 in the
        same budget. Fast rounds make this the common ending, not a rare one.

        In order:

        1. **Resample** (H32), if one is still available. This changes the example plans, i.e.
           the state space, which is the one escalation that can make a policy possible that the
           exhausted sweep could not express at any complexity.
        2. **Add the next unsolved problem** that is not in the training set yet, via the normal
           `_add_next_problem` bookkeeping. Note what this cannot do: if the exhausted sweep
           consisted purely of full-pool `NO_SOLUTION` rounds, every one of those levels is
           refuted for the enlarged training set too (adding an instance is monotone), so the
           re-climb can only pay off where the sweep was *not* cleanly refuted -- rounds that
           timed out, ran out of resources or ran over the restricted generators. Those are
           exactly the rounds a big instance produces, which is why this is worth trying rather
           than stopping.
        3. Neither available -> the caller stops with the `maxcomplexity` reason it always used.

        Both continuations restart the sweep at `min_complexity` (`_restart_sweep`) and neither
        loosens `max_cost`: the resample leaves it alone by design, and in this branch
        `_add_next_problem`'s reset to `MAX_COST` is a no-op, because reaching here with a problem
        left to add means the last elif of `__next__` failed on `active_problems_solved`, and a
        training set that has not been solved since the last problem joined it still carries the
        `MAX_COST` that `_add_next_problem`/INC_PLANS set back then.

        Termination: each continuation either spends one of the finitely many resamples or moves
        one problem into the training set, so the run cannot circle here forever.
        """
        if not self.config.get("continue_after_max_complexity", True):
            return False
        if self.complexity < self.config["max_complexity"] or not self.active_problems:
            # A stop for some other reason (nothing to escalate below the top of the sweep, or a
            # training set that never got started); this mechanism is about the exhausted sweep.
            return False
        unsolved = self.get_unsolved_problems()
        if not unsolved:
            return False
        if not self.continuation.time_left():
            log.info(
                "Max complexity %d reached with %d unsolved problem(s); not continuing, the"
                " wall-clock budget is spent",
                self.complexity,
                len(unsolved),
            )
            return False
        if self.continuation.resample_available():
            replaced, affected = self.continuation.resample()
            if replaced:
                log.info(
                    "Max complexity %d reached with %d unsolved problem(s); continuing by"
                    " resample (%d plan(s) for %d problem(s))",
                    self.complexity,
                    len(unsolved),
                    replaced,
                    affected,
                )
                self._restart_sweep()
                self.max_complexity_continuations += 1
                return True
        next_problem = self._next_addable_problem()
        if next_problem is not None:
            log.info(
                "Max complexity %d reached with %d unsolved problem(s); continuing by adding problem %s",
                self.complexity,
                len(unsolved),
                next_problem.name,
            )
            self._add_next_problem()
            self._restart_sweep()
            self.max_complexity_continuations += 1
            return True
        return False

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
        elif not self._continue_past_max_complexity():
            # Nothing left to escalate and the complexity sweep cannot be restarted on anything
            # new (H33): this is the end of the run.
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
