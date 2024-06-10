from genfond.solver import Solver
from genfond.policy import PolicyType, PolicyRule, Cond, Effect
from genfond.generate_policy import generate_policy


def test_trans_constraint_solver_choose_good_trans():
    program = """
        feature(n_f).
        feature_complexity(n_f, 1).
        feature(n_g).
        feature_complexity(n_g, 1).

        state(0, 0).
        alive(0, 0).
        eval(0, 0, n_f, 0).
        eval(0, 0, n_g, 0).

        state(0, 1).
        eval(0, 1, n_f, 1).
        eval(0, 1, n_g, 0).

        state(0, 2).
        alive(0, 2).
        eval(0, 2, n_f, 1).
        eval(0, 2, n_g, 1).
        goal(0, 2).

        trans(0, 0, a, 1).
        trans(0, 0, b, 2).
    """
    solver = Solver(program, solve_prog='solve_trans_constraints.lp')
    assert solver.solve()
    assert solver.solution['good_trans'] == {(0, 0, 2)}
    assert solver.solution['bad_trans'] == {(0, 0, 1)}
    assert solver.solution['selected'] == {'n_g'}
    policy = generate_policy(solver.solution, policy_type=PolicyType.CONSTRAINED)
    assert policy.rules == {PolicyRule({'n_g': Cond.ZERO}, [[('n_g', Effect.INCREASE)]])}
    assert policy.constraints == {PolicyRule({'n_g': Cond.ZERO}, [[]])}
