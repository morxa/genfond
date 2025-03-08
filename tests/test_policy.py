import copy

from genfond.rule_policy import Cond, Effect, Policy, PolicyRule


def test_policy_simplify():
    policy = Policy(
        {"b_f", "b_g", "n_h"},
        {
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.FALSE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.TRUE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
        },
    )
    policy.simplify()
    assert policy.rules == {PolicyRule({"b_f": Cond.TRUE}, [[("n_h", Effect.INCREASE)]])}


def test_policy_simplify_two_different_conditions():
    policy = Policy(
        {"b_f", "b_g", "n_h"},
        {
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.FALSE,
                    "n_h": Cond.POSITIVE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.TRUE,
                    "n_h": Cond.ZERO,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
        },
    )
    orig_policy = copy.deepcopy(policy)
    policy.simplify()
    assert policy.rules == orig_policy.rules


def test_policy_simplify_subset():
    policy = Policy(
        {"b_f", "b_g", "n_h"},
        {
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.TRUE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
        },
    )
    policy.simplify()
    assert policy.rules == {PolicyRule({"b_f": Cond.TRUE}, [[("n_h", Effect.INCREASE)]])}


def test_policy_simplify_subset2():
    policy = Policy(
        {"b_f", "b_g", "n_h"},
        {
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.TRUE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
        },
    )
    policy.simplify()
    assert policy.rules == {PolicyRule({"b_f": Cond.TRUE}, [[("n_h", Effect.INCREASE)]])}


def test_policy_simplify_mismatching_subset():
    policy = Policy(
        {"b_f", "b_g", "n_h"},
        {
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.TRUE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
            PolicyRule(
                {
                    "b_f": Cond.FALSE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
        },
    )
    policy.simplify()
    assert policy.rules == {
        PolicyRule(
            {
                "b_f": Cond.TRUE,
                "b_g": Cond.TRUE,
            },
            [[("n_h", Effect.INCREASE)]],
        ),
        PolicyRule(
            {
                "b_f": Cond.FALSE,
            },
            [[("n_h", Effect.INCREASE)]],
        ),
    }


def test_policy_simplify_mismatching_keys():
    policy = Policy(
        {"b_f", "b_g", "n_h"},
        {
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.FALSE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "n_h": Cond.ZERO,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
        },
    )
    orig_policy = copy.deepcopy(policy)
    policy.simplify()
    assert policy.rules == orig_policy.rules


def test_policy_simplify_recurse():
    policy = Policy(
        {"b_f", "b_g", "n_h"},
        {
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.FALSE,
                    "n_h": Cond.ZERO,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.FALSE,
                    "n_h": Cond.POSITIVE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.TRUE,
                    "n_h": Cond.ZERO,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
            PolicyRule(
                {
                    "b_f": Cond.TRUE,
                    "b_g": Cond.TRUE,
                    "n_h": Cond.POSITIVE,
                },
                [[("n_h", Effect.INCREASE)]],
            ),
        },
    )
    policy.simplify()
    assert policy.rules == {PolicyRule({"b_f": Cond.TRUE}, [[("n_h", Effect.INCREASE)]])}
