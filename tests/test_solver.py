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


def test_solver_feature_selection():
    program = """
        feature(f).
        feature_complexity(f, 1).
        feature(g).
        feature_complexity(g, 2).
        state(0, 0).
        eval(0, 0, f, 0).
        eval(0, 0, g, 0).

        state(0, 1).
        eval(0, 1, f, 1).
        eval(0, 1, g, 1).
        state(0, 2).
        eval(0, 2, f, 2).
        eval(0, 2, g, 1).
        state(0, 3).
        eval(0, 3, f, 1).
        eval(0, 3, g, 2).
        goal(0, 3).
    """
    solver = Solver(program)
    assert solver.solve()
    assert solver.solution['selected'] == {'g'}


def test_solver_equiv():
    program = """
        feature(f).
        feature_complexity(f, 2).
        feature(g).
        feature_complexity(g, 1).

        state(0, 0).
        eval(0, 0, f, 0).
        eval(0, 0, g, 0).

        state(1, 0).
        eval(1, 0, f, 0).
        eval(1, 0, g, 1).

        state(0, 1).
        eval(0, 1, f, 1).
        eval(0, 1, g, 0).
        goal(0, 1).

        state(1, 1).
        eval(1, 1, f, 1).
        eval(1, 1, g, 1).
        goal(1, 1).

        trans(0, 0, a, 1).
        trans(1, 0, b, 1).
    """
    solver = Solver(program)
    assert solver.solve()
    assert solver.solution['selected'] == {'f'}


def test_solver_equiv2():
    program = """
        feature(f).
        feature_complexity(f, 1).
        feature(g).
        feature_complexity(g, 2).
        feature(h).
        feature_complexity(h, 3).
        state(0, 0).
        eval(0, 0, f, 0).
        eval(0, 0, g, 0).
        eval(0, 0, h, 0).

        state(0, 1).
        eval(0, 1, f, 1).
        eval(0, 1, g, 0).
        eval(0, 1, h, 0).

        state(0, 2).
        eval(0, 2, f, 1).
        eval(0, 2, g, 1).
        eval(0, 2, h, 0).

        state(0, 3).
        eval(0, 3, f, 3).
        eval(0, 3, g, 0).
        eval(0, 3, h, 0).

        state(0, 5).
        eval(0, 5, f, 3).
        eval(0, 5, g, 0).
        eval(0, 5, h, 1).

        state(0, 4).
        eval(0, 4, f, 4).
        eval(0, 4, g, 0).
        eval(0, 4, h, 0).
        goal(0, 4).

        state(0, 6).
        eval(0, 6, f, 4).
        eval(0, 6, g, 0).
        eval(0, 6, h, 1).
        goal(0, 6).

        trans(0, 0, a, 1).
        trans(0, 0, a, 0).
        trans(0, 0, b, 1).
        trans(0, 0, b, 2).
        trans(0, 1, c, 3).
        trans(0, 1, c, 5).
        trans(0, 3, d, 1).
        trans(0, 5, d, 1).
        trans(0, 3, d, 4).
        trans(0, 5, e, 6).
    """
    solver = Solver(program)
    assert solver.solve()
    assert solver.solution['selected'] == {'f', 'g'}
    assert solver.solution['good_trans'] == {(0, 0, 0), (0, 0, 1), (0, 1, 3),
                                             (0, 3, 1), (0, 3, 4), (0, 1, 5),
                                             (0, 5, 1), (0, 5, 6)}


def test_solver_rank():
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
        goal(0, 3).

        trans(0, 0, b, 1).
        trans(0, 1, a, 2).
        trans(0, 2, a, 0).
        trans(0, 0, a, 3).

        % Fix some good transitions.
        good_trans(0, 0, 1).
        good_trans(0, 1, 2).
        good_trans(0, 2, 0).
    """
    solver = Solver(program)
    assert solver.solve()
    assert (0, 0, 3) in solver.solution['good_trans']
