import enum
import sys
import logging

log = logging.getLogger(__name__)

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
        self.last_result = Result.SUCCESS
        self.active_problems_solved = True
        self.max_cost = MAX_COST
        self.solved = {problem.name: False for problem in problems}

    def __iter__(self):
        self.active_problems = []
        self.all_features = False
        self.complexity = self.min_complexity
        return self

    def set_last_result(self, result, cost=MAX_COST):
        self.last_result = result
        if result == Result.SUCCESS:
            self.active_problems_solved = True
            self.max_cost = cost - 1
            self.solved = {
                problem.name: True if problem in self.active_problems else False
                for problem in self.problems
            }

    def set_solved(self, problem):
        self.solved[problem.name] = True

    def get_unsolved_problems(self):
        return [problem for problem in self.problems if not self.solved[problem.name]]

    def __next__(self):
        assert self.last_result != Result.UNKNOWN, "You must set the result of the last problem before calling next"
        if (self.active_problems and self.last_result != Result.OUT_OF_RESOURCES and not self.all_features
                and self.use_all_features):
            self.all_features = True
        elif (self.active_problems and self.last_result != Result.OUT_OF_RESOURCES
              and self.complexity < self.max_complexity and self.max_cost > self.complexity):
            self.all_features = False
            self.complexity += 1
        elif self.active_problems_solved and any(not solved for solved in self.solved.values()):
            self.all_features = False
            self.max_cost = MAX_COST
            self.active_problems_solved = False
            next_problem = next(problem for problem in self.problems if not self.solved[problem.name])
            if self.active_problems and self.problems.index(next_problem) > max(
                [self.problems.index(problem) for problem in self.active_problems]):
                self.active_problems = [next_problem]
            else:
                self.active_problems.append(next_problem)
        else:
            raise StopIteration
        log.debug(
            f'Next set: {", ".join([p.name for p in self.active_problems])}, {self.complexity}, {self.all_features}, {self.max_cost}'
        )
        return (self.active_problems, self.complexity, self.all_features, self.max_cost)
