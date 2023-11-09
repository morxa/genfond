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
