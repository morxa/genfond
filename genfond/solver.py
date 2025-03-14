import logging
import os
import os.path
from typing import Optional

import clingo

from .problem_iterator import MAX_COST

log = logging.getLogger(__name__)


def convert_arg(symbol: clingo.Symbol) -> str | int:
    if symbol.type == clingo.SymbolType.Number:
        return symbol.number
    elif symbol.type == clingo.SymbolType.String:
        return symbol.string
    elif symbol.type == clingo.SymbolType.Function:
        return str(symbol)
    else:
        raise ValueError(f"Unknown symbol type: {symbol.type} for {symbol}")


class Solver:

    def __init__(
        self,
        asp_code: str,
        num_threads: Optional[int] = None,
        solve_prog: str = "solve.lp",
        max_cost: int = MAX_COST,
        max_prune_cost: int = MAX_COST,
        min_feature_complexity: Optional[int] = None,
    ):
        self.asp_code = asp_code
        self.control = clingo.Control()
        self.control.load(os.path.join(os.path.dirname(__file__), solve_prog))
        self.control.add("instances", [], asp_code)
        parts: list[tuple[str, list[clingo.Symbol]]] = [("base", []), ("instances", [])]
        if max_cost < MAX_COST:
            parts.append(("limit_feature_cost", [clingo.Number(max_cost)]))
        if max_prune_cost < MAX_COST:
            parts.append(("limit_prune_cost", [clingo.Number(max_prune_cost)]))
        if min_feature_complexity:
            parts.append(("min_feature_complexity", [clingo.Number(min_feature_complexity)]))
        self.control.ground(parts)
        assert isinstance(self.control.configuration.solve, clingo.Configuration)
        self.control.configuration.solve.parallel_mode = num_threads or os.cpu_count()
        self.solution: dict = dict()
        self.cost: list[int] = []
        self.statistics: dict = dict()

    def on_model(self, model: clingo.Model) -> None:
        if not self.solution:
            log.info("Found first solution")
        self.solution = dict()
        for symbol in model.symbols(shown=True):
            args = [convert_arg(arg) for arg in symbol.arguments]
            if len(args) == 0:
                self.solution.setdefault(symbol.name, set()).add(True)
            elif len(args) == 1:
                self.solution.setdefault(symbol.name, set()).add(args[0])
            else:
                self.solution.setdefault(symbol.name, set()).add(tuple([convert_arg(arg) for arg in symbol.arguments]))
        self.solution["cost"] = model.cost
        self.cost = model.cost

    def solve(self) -> bool:
        res = self.control.solve(on_model=self.on_model)
        self.statistics = self.control.statistics
        assert res.satisfiable is not None
        return res.satisfiable
