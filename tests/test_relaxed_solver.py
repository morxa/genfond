from genfond.solver import Solver


def test_relaxed_solver_choose_good_trans():
    program = """
        feature(f).
        feature_complexity(f, 1).
        feature(g).
        feature_complexity(g, 1).

        state(0, 0).
        alive(0, 0).
        eval(0, 0, f, 0).
        eval(0, 0, g, 0).

        state(0, 1).
        eval(0, 1, f, 1).
        eval(0, 1, g, 0).

        state(0, 2).
        alive(0, 2).
        eval(0, 2, f, 1).
        eval(0, 2, g, 1).
        goal(0, 2).

        trans(0, 0, a, 1).
        trans(0, 0, b, 2).
    """
    solver = Solver(program, solve_prog='solve_relax.lp')
    assert solver.solve()
    assert solver.solution['good_trans'] == {(0, 0, 2)}
    assert solver.solution['bad_trans'] == {(0, 0, 1)}
    assert solver.solution['selected'] == {'g'}
