from genfond.solver import Solver


def test_solver_choose_good_trans():
    program = """
        feature(f).
        feature_complexity(f, 1).

        state(0, 0).
        eval(0, 0, f, 0).

        state(0, 1).
        eval(0, 1, f, 1).

        state(0, 2).
        eval(0, 2, f, 2).
        goal(0, 2).

        trans(0, 0, a, 1).
        trans(0, 0, b, 2).
    """
    solver = Solver(program)
    assert solver.solve()
    assert solver.solution['good_trans'] == {(0, 0, 2)}


def test_solver_no_good_trans():
    program = """
        feature(f).
        feature_complexity(f, 1).

        state(0, 0).
        eval(0, 0, f, 0).

        state(0, 1).
        eval(0, 1, f, 1).

        state(0, 2).
        eval(0, 2, f, 2).
        goal(0, 2).

        trans(0, 0, a, 1).
        trans(0, 0, a, 2).
    """
    solver = Solver(program)
    assert solver.solve() is False, f'Unexpected solution {solver.solution}'


def test_solver_two_conflicting_actions():
    program = """
        feature(f).
        feature_complexity(f, 1).

        state(0, 0).
        eval(0, 0, f, 0).

        state(0, 1).
        eval(0, 1, f, 1).

        state(0, 2).
        eval(0, 2, f, 2).

        state(0, 3).
        eval(0, 3, f, 3).

        state(0, 4).
        eval(0, 4, f, 4).
        goal(0, 4).

        trans(0, 0, a, 1).
        trans(0, 0, a, 0).
        trans(0, 0, b, 1).
        trans(0, 0, b, 2).
        trans(0, 1, c, 3).
        trans(0, 3, d, 4).
    """
    solver = Solver(program)
    assert solver.solve()
    assert solver.solution['good_trans'] == {(0, 0, 0), (0, 0, 1), (0, 1, 3),
                                             (0, 3, 4)}
