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

    def __init__(self, asp_code, num_threads=None):
        self.asp_code = asp_code
        self.control = clingo.Control()
        s = clingo.Symbol
        s.type
        progfile = open(os.path.dirname(__file__) + '/solve.lp', 'r')
        self.prog = progfile.read() + '\n' + asp_code
        self.control.add("base", [], self.prog)
        self.control.ground([("base", [])])
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
        self.cost = model.cost

    def solve(self):
        return self.control.solve(on_model=self.on_model).satisfiable
