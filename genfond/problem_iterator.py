import enum
import logging
import sys
from typing import Any, Collection, Iterator, Mapping, MutableMapping, Optional

from pddl.core import Plan, Problem

from .ground import state_string
from .state_space_generator import State

log = logging.getLogger("genfond.problem_iterator")

MAX_COST = sys.maxsize


def plan_key(plan: Plan) -> tuple:
    """A hashable identity for a plan. `Plan` defines __eq__ but no __hash__."""
    return tuple((str(name), tuple(str(arg) for arg in args)) for name, args in plan.actions)


class Result(enum.Enum):
    UNKNOWN = 0
    SUCCESS = 1
    NO_SOLUTION = 2
    OUT_OF_RESOURCES = 3
    # A model was found, but it relies on transitions into unexpanded frontier states, so it
    # is not a policy yet. The caller expands those states and the same configuration is
    # retried with the enlarged plan set.
    FRONTIER = 4


class LastStep(enum.Enum):
    START = 0
    INC_PLANS = 1
    INC_COMPLEXITY = 2
    EXPAND_FRONTIER = 3


class ProblemIterator:

    def __init__(self, problems: list[Problem], config: Mapping, plans: Optional[Mapping[str, Iterator[Plan]]] = None):
        self.problems = problems
        self.config = config
        self.plan_iterators = plans

    def __iter__(self) -> "ProblemIterator":
        self.active_problems: list[Problem] = []
        self.active_plans: MutableMapping[str, Plan] = dict()
        self.selected_states: dict[str, set[State]] = dict()
        self.new_states: dict[str, set[State]] = dict()
        self.dead_states: dict[str, set[State]] = dict()
        self.frontier_expansions = 0
        self.frontier_progress = False
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
            for plan in new_plans:
                if plan_key(plan) in known:
                    # The planner is deterministic, so a frontier state that the new plan
                    # fails to expand yields the same plan every round. Re-adding it is not
                    # progress; counting it as such spins this loop until the expansion
                    # budget runs out.
                    log.debug("Frontier plan for %s is already an example plan, ignoring it", problem_name)
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

    def set_last_result(self, result: Result, cost: Optional[tuple[int]] = None) -> None:
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
            self.max_cost = cost[-1] - 1
            if full_feature_pool:
                # clingo minimizes the feature cost, so `cost[-1]` is optimal for this pool:
                # nothing at this complexity beats the new `max_cost`. That refutes the level
                # just as an UNSAT would, and keeps the `max_cost < complexity` short circuit
                # in `iterative_solver.solve` available for the rounds that follow.
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

    def __next__(self) -> Mapping[str, Any]:
        assert self.last_result != Result.UNKNOWN, "You must set the result of the last problem before calling next"
        log.debug(
            f"last result: {self.last_result.name}, all features: {self.all_features}, complexity: {self.complexity}"
        )
        if self.last_result == Result.FRONTIER and self.frontier_progress and self.frontier_budget_left():
            # Retry the exact same configuration; only the plan and dead-end sets grew.
            self.last_step = LastStep.EXPAND_FRONTIER
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
                # Find the next plan for an active problem that is not yet solved
                found := next(
                    (
                        (k.name, v)
                        for k in self.active_problems
                        if not self.solved[k.name] and (v := next(self.plan_iterators[k.name], None)) is not None
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
        elif self.active_problems_solved and any(not solved for solved in self.solved.values()):
            self.all_features = False
            self.max_cost = MAX_COST
            self.active_problems_solved = False
            self.complexity = self.succ_complexity
            self.sweep_target = self.complexity
            self.last_step = LastStep.START
            next_problem = next(
                problem
                for problem in self.problems
                if not self.solved[problem.name] and problem not in self.active_problems
            )
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
                # Adding an instance is monotone: a selection that solves the larger set also
                # solves every subset, so "no solution below `succ_complexity`" carries over
                # and the sweep resumes there instead of at `min_complexity`. That bound is
                # unconditional -- it was established before the success tightened `max_cost`,
                # which is reset here anyway.
                self.refuted_complexity = self.succ_complexity - 1
            if self.plan_iterators:
                self.active_plans[next_problem.name] = []
                while len(self.active_plans[next_problem.name]) < self.config["min_number_of_plans"]:
                    next_plan = next(self.plan_iterators[next_problem.name], None)
                    if next_plan is None:
                        break
                    self.active_plans[next_problem.name].append(next_plan)
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
