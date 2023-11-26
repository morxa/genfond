from genfond.policy import Policy, PolicyRule, Cond, Effect


def test_policy_simplify():
    policy = Policy({'b_f', 'b_g', 'n_h'}, {
        PolicyRule({
            'b_f': Cond.TRUE,
            'b_g': Cond.FALSE
        }, [[('n_h', Effect.INCREASE)]]),
        PolicyRule({
            'b_f': Cond.TRUE,
            'b_g': Cond.TRUE
        }, [[('n_h', Effect.INCREASE)]])
    })
    policy.simplify()
    assert policy.rules == {PolicyRule({'b_f': Cond.TRUE}, [[('n_h', Effect.INCREASE)]])}
