import enum
import sys
import logging
from .state_space_generator import state_to_string

log = logging.getLogger('genfond.problem_iterator')

MAX_COST = sys.maxsize


class Result(enum.Enum):
    UNKNOWN = 0
    SUCCESS = 1
    NO_SOLUTION = 2
    OUT_OF_RESOURCES = 3


class ProblemIterator:

    def __init__(self, problems, min_complexity, max_complexity, use_all_features):
        self.problems = problems
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
        self.use_all_features = use_all_features

    def __iter__(self):
        self.active_problems = []
        self.selected_states = dict()
        self.new_states = dict()
        self.all_features = False
        self.complexity = self.min_complexity
        self.last_result = Result.SUCCESS
        self.succ_complexity = self.complexity
        self.active_problems_solved = True
        self.max_cost = MAX_COST
        self.solved = {problem.name: False for problem in self.problems}
        return self

    def set_last_result(self, result, cost=MAX_COST):
        self.last_result = result
        if result == Result.SUCCESS:
            self.active_problems_solved = True
            self.max_cost = cost - 1
            self.succ_complexity = self.complexity
            # self.solved = {
            #     problem.name: True if problem in self.active_problems else False
            #     for problem in self.problems
            # }
            self.new_states.clear()
            for problem in self.problems:
                if problem.name in self.selected_states and problem not in self.active_problems:
                    del self.selected_states[problem.name]

    def set_new_state(self, problem_name, state):
        log.debug(f'Adding new state for {problem_name}: {state_to_string(state)}')
        self.new_states.setdefault(problem_name, set()).add(state)

    def _update_selected_states(self):
        for problem, states in self.new_states.items():
            self.selected_states.setdefault(problem, set()).update(states)
        self.new_states.clear()

    def set_solved(self, problem):
        self.solved[problem.name] = True

    def get_unsolved_problems(self):
        return [problem for problem in self.problems if not self.solved[problem.name]]

    def __next__(self):
        assert self.last_result != Result.UNKNOWN, "You must set the result of the last problem before calling next"
        log.debug(f'new states: {", ".join([f"{k}: {len(v)}" for k, v in self.new_states.items()])}')
        log.debug(f'active problems: {", ".join([p.name for p in self.active_problems])}')
        log.debug(
            f'last result: {self.last_result.name}, all features: {self.all_features}, complexity: {self.complexity}')
        if any(problem.name in self.new_states for problem in self.active_problems):
            self.all_features = False
            self.max_cost = MAX_COST
            self.active_problems_solved = False
            self.complexity = self.succ_complexity
            self._update_selected_states()
        elif (self.active_problems and self.last_result != Result.OUT_OF_RESOURCES and not self.all_features
              and self.use_all_features):
            self.all_features = True
        elif (self.active_problems and self.all_features and self.complexity < self.max_complexity
              and self.max_cost > self.complexity):
            self.all_features = False
            self.complexity += 1
        elif self.active_problems_solved and any(not solved for solved in self.solved.values()):
            self.all_features = False
            self.max_cost = MAX_COST
            self.active_problems_solved = False
            self.complexity = self.succ_complexity
            next_problem = next(problem for problem in self.problems if not self.solved[problem.name])
            if not next_problem.name in self.new_states:
                self.new_states[next_problem.name] = {next_problem.init}
            self._update_selected_states()
            if False and self.active_problems and self.problems.index(next_problem) > max(
                [self.problems.index(problem) for problem in self.active_problems]):
                self.active_problems = [next_problem]
            else:
                self.active_problems.append(next_problem)
        else:
            raise StopIteration
        log.debug(f'Next set: {", ".join([p.name for p in self.active_problems])},'
                  f' complexity={self.complexity},'
                  f' all_features={self.all_features},'
                  f' max_cost={self.max_cost if self.max_cost < MAX_COST else "MAX_COST"},'
                  f' |selected_states|={len(self.selected_states)} states')
        return (self.active_problems, self.complexity, self.all_features, self.max_cost, self.selected_states)
