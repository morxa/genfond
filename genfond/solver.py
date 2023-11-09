import clingo
import os.path


def convert_arg(symbol):
    if symbol.type == clingo.SymbolType.Number:
        return symbol.number
    elif symbol.type == clingo.SymbolType.String:
        return symbol.string
    elif symbol.type == clingo.SymbolType.Function:
        if len(symbol.arguments) == 0:
            return symbol.name
        else:
            return tuple([convert_arg(arg) for arg in symbol.arguments])
    else:
        raise ValueError(f'Unknown symbol type: {symbol.type} for {symbol}')


class Solver:

    def __init__(self, asp_code):
        self.asp_code = asp_code
        self.control = clingo.Control()
        s = clingo.Symbol
        s.type
        progfile = open(os.path.dirname(__file__) + '/solve.lp', 'r')
        prog = progfile.read() + '\n' + asp_code
        self.control.add("base", [], prog)
        self.control.ground([("base", [])])
        self.solution = None

    def on_model(self, model):
        self.solution = dict()
        for symbol in model.symbols(shown=True):
            self.solution.setdefault(symbol.name, set()).add(
                tuple([convert_arg(arg) for arg in symbol.arguments]))

    def solve(self):
        return self.control.solve(on_model=self.on_model).satisfiable
