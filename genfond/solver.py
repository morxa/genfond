import clingo
import os.path
import os
import logging

log = logging.getLogger(__name__)


def convert_arg(symbol):
    if symbol.type == clingo.SymbolType.Number:
        return symbol.number
    elif symbol.type == clingo.SymbolType.String:
        return symbol.string
    elif symbol.type == clingo.SymbolType.Function:
        return str(symbol)
    else:
        raise ValueError(f'Unknown symbol type: {symbol.type} for {symbol}')


class Solver:

    def __init__(self, asp_code, num_threads=None, solve_prog='solve.lp'):
        self.asp_code = asp_code
        self.control = clingo.Control()
        self.control.load(os.path.join(os.path.dirname(__file__), solve_prog))
        self.control.add("instances", [], asp_code)
        self.control.ground([("base", []), ("instances", [])])
        self.control.configuration.solve.parallel_mode = num_threads or os.cpu_count()
        self.solution = None
        self.cost = None

    def on_model(self, model):
        if not self.solution:
            log.info('Found first solution')
        self.solution = dict()
        for symbol in model.symbols(shown=True):
            args = [convert_arg(arg) for arg in symbol.arguments]
            if len(args) == 0:
                self.solution.setdefault(symbol.name, set()).add(True)
            elif len(args) == 1:
                self.solution.setdefault(symbol.name, set()).add(args[0])
            else:
                self.solution.setdefault(symbol.name, set()).add(tuple([convert_arg(arg) for arg in symbol.arguments]))
        self.solution['cost'] = model.cost
        self.cost = model.cost

    def solve(self):
        return self.control.solve(on_model=self.on_model).satisfiable
