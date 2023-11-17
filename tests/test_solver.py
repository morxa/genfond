from genfond.solver import Solver
from genfond.policy import Cond, Effect, PolicyRule, generate_policy


def test_solver_choose_good_trans():
    program = """
        feature(f).
        feature_complexity(f, 1).
        feature(g).
        feature_complexity(g, 1).

        state(0, 0).
        eval(0, 0, f, 0).
        eval(0, 0, g, 0).

        state(0, 1).
        eval(0, 1, f, 1).
        eval(0, 1, g, 0).

        state(0, 2).
        eval(0, 2, f, 1).
        eval(0, 2, g, 1).
        goal(0, 2).

        trans(0, 0, a, 1).
        trans(0, 0, b, 2).
    """
    solver = Solver(program)
    assert solver.solve()
    assert solver.solution['good_trans'] == {(0, 0, 2)}
    assert solver.solution['selected'] == {'f', 'g'}
    assert solver.solution['trans_delta'] == {(0, 0, 1, 'f', 1), (0, 0, 1, 'g', 0), (0, 0, 2, 'f', 1),
                                              (0, 0, 2, 'g', 1)}


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
        feature(g).
        feature_complexity(g, 1).
        feature(dead).
        feature_complexity(dead, 1).

        state(0, 0).
        eval(0, 0, f, 0).
        eval(0, 0, g, 0).
        eval(0, 0, dead, 0).

        state(0, 1).
        eval(0, 1, f, 1).
        eval(0, 1, g, 0).
        eval(0, 1, dead, 0).

        state(0, 2).
        eval(0, 2, f, 2).
        eval(0, 2, g, 0).
        eval(0, 2, dead, 1).

        state(0, 3).
        eval(0, 3, f, 3).
        eval(0, 3, g, 0).
        eval(0, 3, dead, 0).

        state(0, 4).
        eval(0, 4, f, 4).
        eval(0, 4, g, 1).
        goal(0, 4).
        eval(0, 4, dead, 0).

        trans(0, 0, a, 1).
        trans(0, 0, a, 0).
        trans(0, 0, b, 1).
        trans(0, 0, b, 2).
        trans(0, 1, c, 3).
        trans(0, 3, d, 4).
    """
    solver = Solver(program)
    assert solver.solve()
    assert solver.solution['good_trans'] == {(0, 0, 0), (0, 0, 1), (0, 1, 3), (0, 3, 4)}
    assert solver.solution['selected'] == {'f', 'dead'}
    assert solver.solution['trans_delta'] == {(0, 0, 0, 'f', 0), (0, 0, 0, 'dead', 0), (0, 0, 1, 'f', 1),
                                              (0, 0, 1, 'dead', 0), (0, 0, 2, 'f', 1), (0, 0, 2, 'dead', 1),
                                              (0, 1, 3, 'f', 1), (0, 1, 3, 'dead', 0), (0, 3, 4, 'f', 1),
                                              (0, 3, 4, 'dead', 0)}


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
    assert solver.solution['good_trans'] == {(0, 0, 1), (1, 0, 1)}
    assert solver.solution['trans_delta'] == {(0, 0, 1, 'f', 1), (1, 0, 1, 'f', 1)}


def test_solver_equiv2(program_with_nontriv_equiv):
    program = program_with_nontriv_equiv
    solver = Solver(program)
    assert solver.solve()
    assert solver.solution['selected'] == {'n_f', 'b_g'}


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


def test_bool_equiv():
    program = """
        feature(f).
        feature_complexity(f, 1).
        feature(g).
        feature_complexity(g, 2).

        state(0, 0).
        eval(0, 0, f, 3).
        eval(0, 0, g, 0).

        state(0, 1).
        eval(0, 1, f, 2).
        eval(0, 1, g, 0).

        state(0, 2).
        eval(0, 2, f, 1).
        eval(0, 2, g, 0).

        state(0, 3).
        eval(0, 3, f, 1).
        eval(0, 3, g, 1).
        goal(0, 3).

        % This transition cannot be chosen because it cannot be
        % boolean-distinguished from the next transition (which is a bad
        % transition).
        trans(0, 0, a, 1).
        trans(0, 1, a, 2).
        trans(0, 1, b, 3).
    """
    solver = Solver(program)
    assert solver.solve() is False, f'Unexpected solution {solver.solution}'


def test_solver_bool_equiv2(simple_program):
    solver = Solver(simple_program)
    assert solver.solve()
    # 'g' must also be chosen to bool-distinguish states 2 and 3.
    assert solver.solution['selected'] == {'n_f', 'b_g', 'b_h'}
    assert solver.solution['bool_repr'] == {(0, 0), (0, 2), (0, 4)}


def test_solver_generate_policy(simple_program):
    solver = Solver(simple_program)
    assert solver.solve()
    assert solver.solution['good_trans_delta'] == {(0, 0, 1, 'n_f', -1), (0, 0, 1, 'b_g', 0), (0, 0, 1, 'b_h', 0),
                                                   (0, 1, 3, 'n_f', -1), (0, 1, 3, 'b_g', 0), (0, 1, 3, 'b_h', 0),
                                                   (0, 3, 4, 'n_f', 0), (0, 3, 4, 'b_g', 0), (0, 3, 4, 'b_h', 1)}
    policy = generate_policy(solver.solution)
    print(policy)
    assert policy.rules == {
        PolicyRule([('n_f', Cond.POSITIVE), ('b_g', Cond.FALSE), ('b_h', Cond.FALSE)], [[('n_f', Effect.DECREASE)]]),
        PolicyRule([('n_f', Cond.POSITIVE), ('b_g', Cond.FALSE), ('b_h', Cond.FALSE)], [[('b_h', Effect.SET)]])
    }


def test_solver_policy_nontriv_equiv(program_with_nontriv_equiv):
    solver = Solver(program_with_nontriv_equiv)
    assert solver.solve()
    assert solver.solution['good_trans'] == {(0, 0, 0), (0, 0, 1), (0, 1, 3), (0, 3, 1), (0, 3, 4), (0, 1, 5),
                                             (0, 5, 1), (0, 5, 6)}
    assert solver.solution['trans_delta'] == {(0, 0, 0, 'n_f', 0), (0, 0, 0, 'b_g', 0), (0, 0, 1, 'n_f', 1),
                                              (0, 0, 1, 'b_g', 0), (0, 0, 2, 'n_f', 1), (0, 0, 2, 'b_g', 1),
                                              (0, 1, 3, 'n_f', 1), (0, 1, 3, 'b_g', 0), (0, 1, 5, 'n_f', 1),
                                              (0, 1, 5, 'b_g', 0), (0, 3, 1, 'n_f', -1), (0, 3, 1, 'b_g', 0),
                                              (0, 5, 1, 'n_f', -1), (0, 5, 1, 'b_g', 0), (0, 3, 4, 'n_f', 1),
                                              (0, 3, 4, 'b_g', 0), (0, 5, 6, 'n_f', 1), (0, 5, 6, 'b_g', 0)}
    policy = generate_policy(solver.solution)
    assert policy.rules == {
        PolicyRule([('n_f', Cond.POSITIVE), ('b_g', Cond.FALSE)],
                   [[('n_f', Effect.DECREASE)], [('n_f', Effect.INCREASE)]]),
        PolicyRule([('n_f', Cond.POSITIVE), ('b_g', Cond.FALSE)], [[('n_f', Effect.INCREASE)]]),
        PolicyRule([('n_f', Cond.ZERO), ('b_g', Cond.FALSE)], [[], ([('n_f', Effect.INCREASE)])]),
    }
