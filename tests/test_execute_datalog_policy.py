import random

import pddl
import pytest

from genfond.config_handler import ConfigHandler
from genfond.datalog_policy import Cond, DatalogPolicy, DatalogPolicyRule
from genfond.execute_datalog_policy import execute_datalog_policy
from genfond.execute_rule_policy import ExecutionTimeout
from genfond.ground import action_string


def test_block_clear_all(blocks_clear):
    domain, problem = blocks_clear

    policy = DatalogPolicy(
        [
            DatalogPolicyRule(
                "unstack(X, Y)",
                concepts=[
                    ("X", "c_primitive(clear, 0)"),
                ],
            ),
            DatalogPolicyRule(
                "putdown(X)",
                concepts=[
                    ("X", "c_primitive(clear, 0)"),
                ],
            ),
        ]
    )
    config = ConfigHandler()
    execute_datalog_policy(domain, problem, policy, config)


def test_execute_datalog_policy_time_limit_triggers_timeout(blocks_clear):
    """The real (non-mocked) `validation_time_limit` mechanism: a vanishingly small time_limit
    must be exceeded before the very first step (goal not already satisfied by problem.init, so
    the loop body runs at least once), raising ExecutionTimeout rather than running to
    completion or hanging. See test_in_loop_validation.py for the higher-level
    `_test_policy_on_problems` behaviour this enables (mocked, since that test is about the
    control flow around execute_policy rather than the deadline check itself)."""
    domain, problem = blocks_clear
    policy = DatalogPolicy(
        [
            DatalogPolicyRule(
                "unstack(X, Y)",
                concepts=[
                    ("X", "c_primitive(clear, 0)"),
                ],
            ),
            DatalogPolicyRule(
                "putdown(X)",
                concepts=[
                    ("X", "c_primitive(clear, 0)"),
                ],
            ),
        ]
    )
    config = ConfigHandler()
    with pytest.raises(ExecutionTimeout):
        execute_datalog_policy(domain, problem, policy, config, time_limit=1e-9)


def test_fond_blocks(fond_blocks):
    domain, problem = fond_blocks

    policy = DatalogPolicy(
        [
            DatalogPolicyRule(
                "puton(X, Y, Z)",
                concepts=[
                    ("Y", "c_one_of(Table)"),
                    (
                        "X",
                        "c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),c_projection(r_primitive(on_g,0,1),0))",
                    ),
                ],
            ),
            DatalogPolicyRule(
                "puton(X, Y, Z)",
                concepts=[
                    ("Y", "c_one_of(Table)"),
                    (
                        "X",
                        "c_some(r_transitive_reflexive_closure(r_primitive(on,0,1)),c_projection(r_primitive(on_g,0,1),1))",
                    ),
                ],
            ),
            DatalogPolicyRule(
                "puton(X, Y, Z)",
                concepts=[
                    ("X", "c_projection(r_primitive(on_g,0,1),0)"),
                    ("Y", "c_projection(r_primitive(on_g,0,1),1)"),
                ],
            ),
        ]
    )
    config = ConfigHandler()
    execute_datalog_policy(domain, problem, policy, config)


def test_datalog_policy_with_conds(doors):
    domain, problem = doors
    policy = DatalogPolicy(
        [
            DatalogPolicyRule("move-forward-door-open(P1, P2, P3, P4)", conds={"b_nullary(hold-key)": Cond.TRUE}),
            DatalogPolicyRule("move-forward-door-closed(P1, P2, P3, P4)", conds={"b_nullary(hold-key)": Cond.TRUE}),
            DatalogPolicyRule("move-forward-last-door-open(P1, P2, P3)", conds={"b_nullary(hold-key)": Cond.TRUE}),
            DatalogPolicyRule("move-forward-last-door-closed(P1, P2, P3)", conds={"b_nullary(hold-key)": Cond.TRUE}),
            DatalogPolicyRule("pick-key(X)", conds={"b_nullary(hold-key)": Cond.FALSE}),
        ]
    )
    config = ConfigHandler()
    config["policy_steps"] = 10
    for _ in range(10):
        execute_datalog_policy(domain, problem, policy, config)


def test_datalog_policy_with_roles(fond_blocks):
    domain, problem = fond_blocks
    policy = DatalogPolicy(
        [
            DatalogPolicyRule(
                "puton(X, Y, Z)",
                roles=[
                    ("X", "Y", "r_primitive(on_g,0,1)"),
                ],
            ),
            DatalogPolicyRule(
                "puton(X, Y, Z)",
                concepts=[("Y", "c_one_of(Table)")],
            ),
        ]
    )
    config = ConfigHandler()
    config["policy_steps"] = 10
    for _ in range(10):
        execute_datalog_policy(domain, problem, policy, config)


def test_blocks3ops(blocks3ops):
    domain, problem = blocks3ops

    cond1 = (
        "b_empty(r_restrict(r_primitive(on, 0, 1), c_and(c_primitive(ontable, 0), c_not(c_primitive(ontable_g, 0)))))"
    )
    cond2 = "b_empty(r_and(r_primitive(on, 0, 1), r_not(r_primitive(on_g, 0, 1))))"

    policy = DatalogPolicy(
        [
            DatalogPolicyRule(
                "newtower(X, Y)",
                conds={
                    cond1: Cond.FALSE,
                },
            ),
            DatalogPolicyRule(
                "newtower(X, Y)",
                conds={
                    cond2: Cond.FALSE,
                },
            ),
            DatalogPolicyRule(
                "stack(X, Y)",
                concepts=[
                    ("Y", "c_primitive(ontable, 0)"),
                    ("Y", "c_primitive(ontable_g, 0)"),
                ],
                roles=[("X", "Y", "r_primitive(on_g, 0, 1)")],
                conds={
                    cond1: Cond.TRUE,
                    cond2: Cond.TRUE,
                },
            ),
            DatalogPolicyRule(
                "stack(X, Y)",
                concepts=[
                    ("Y", "c_some(r_and(r_primitive(on, 0, 1), r_inverse(r_primitive(on, 1, 0))), c_top)"),
                ],
                roles=[("X", "Y", "r_primitive(on_g, 0, 1)")],
                conds={
                    cond1: Cond.TRUE,
                    cond2: Cond.TRUE,
                },
            ),
        ]
    )
    config = ConfigHandler()
    config["policy_steps"] = 10
    execute_datalog_policy(domain, problem, policy, config)


def test_execute_datalog_policy_with_augmented_states(blocks_clear):
    domain, problem = blocks_clear
    policy = DatalogPolicy(
        [
            DatalogPolicyRule(
                "unstack(X, Y)",
                state_aug_conds={"b_empty(c_and(c_primitive(clear_g,0),c_primitive(aparam1,0)))": Cond.FALSE},
            )
        ]
    )
    config = ConfigHandler()
    config["include_actions"] = True
    config["policy_steps"] = 1
    execute_datalog_policy(domain, problem, policy, config)
    policy = DatalogPolicy(
        [
            DatalogPolicyRule(
                "unstack(X, Y)",
                state_aug_conds={"b_empty(c_and(c_primitive(clear_g,0),c_primitive(aparam0,0)))": Cond.FALSE},
            )
        ]
    )
    with pytest.raises(RuntimeError):
        execute_datalog_policy(domain, problem, policy, config)


FLAT_PROBLEM = """(define (problem blocks-flat)
    (:domain blocks3ops)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b0) (clear b1) (clear b2) (clear b3) (clear b4)
           (ontable b0) (ontable b1) (ontable b2) (ontable b3) (ontable b4))
    (:goal (and (on b0 b1) (on b1 b2) (on b2 b3) (on b3 b4)))
)
"""


def _flat_blocks(tmp_path, blocks3ops):
    """The blocks3ops domain over five loose blocks, so every `stack` binding is applicable."""
    domain, _ = blocks3ops
    path = tmp_path / "flat.pddl"
    path.write_text(FLAT_PROBLEM)
    return domain, pddl.parse_problem(path)


def _pinned_trajectory(domain, problem, policy, seed, steps):
    """The trajectory `policy` takes from `problem.init` under `seed`, as action strings."""
    config = ConfigHandler()
    config["policy_steps"] = steps
    config["abort_on_cycle"] = False
    random.seed(seed)
    actions: list = []
    try:
        execute_datalog_policy(domain, problem, policy, config, out_actions=actions)
    except RuntimeError:  # a stub policy need not reach the goal; only the prefix is of interest
        pass
    return [action_string(action) for action in actions]


@pytest.mark.parametrize(
    "seed, expected",
    [
        (0, ["stack(b4,b0)", "stack(b3,b2)", "stack(b1,b3)"]),
        (7, ["stack(b4,b2)", "stack(b1,b3)", "stack(b0,b4)"]),
    ],
)
def test_datalog_execution_binding_order_is_pinned(tmp_path, blocks3ops, seed, expected):
    """A seeded execution must keep picking the *same* binding out of the applicable ones.

    The rule is unconditioned and every block is loose, so all twenty ground `stack` actions are
    legal choices and the trajectory is decided purely by the order the executor enumerates
    candidate bindings in: one `random.shuffle` per rule parameter, then the object product in
    the order those shuffles produced. `execute_datalog_policy` prunes that product hard -- it
    only visits tuples that name a ground action applicable in the current state -- and this
    pins that the pruning is order-preserving, i.e. that it skips exactly the tuples the
    conditions would have rejected anyway. A change to the draw sequence or to the enumeration
    order breaks this test, and with it the reproducibility of every recorded run.
    """
    domain, problem = _flat_blocks(tmp_path, blocks3ops)
    policy = DatalogPolicy([DatalogPolicyRule("stack(X, Y)")])
    assert _pinned_trajectory(domain, problem, policy, seed, len(expected)) == expected


def test_datalog_execution_binding_order_is_pinned_with_concepts(tmp_path, blocks3ops):
    """As above, but with one parameter already narrowed by a concept.

    The concept filter shortens the list *before* the shuffle and the applicability pruning
    shortens it after, so this pins that the two compose without disturbing the draw sequence.
    """
    domain, problem = _flat_blocks(tmp_path, blocks3ops)
    policy = DatalogPolicy([DatalogPolicyRule("stack(X, Y)", concepts=[("Y", "c_some(r_primitive(on_g,0,1),c_top)")])])
    assert _pinned_trajectory(domain, problem, policy, 0, 3) == [
        "stack(b4,b0)",
        "stack(b2,b3)",
        "stack(b1,b2)",
    ]


def test_execute_datalog_policy_with_parameter_augmented_conditions(blocks_clear):
    """A `param_aug_conds` rule: the feature is evaluated on the state augmented with one marked
    action parameter (`aparam0`), not with the whole action.

    The executor memoises that augmented dlplan state per marked object within a step, so this
    also covers the cache: the conditions of one rule are checked against the same state the
    unmemoised code rebuilt for each of them.
    """
    domain, problem = blocks_clear
    feature = "b_empty(c_and(c_primitive(clear_g,0),c_primitive(aparam0,0)))"
    config = ConfigHandler(None, "datalog-action-params")
    config["policy_steps"] = 3
    random.seed(0)
    actions: list = []
    execute_datalog_policy(
        domain,
        problem,
        DatalogPolicy([DatalogPolicyRule("unstack(X, Y)", param_aug_conds={feature: (0, Cond.TRUE)})]),
        config,
        out_actions=actions,
    )
    assert [action_string(action) for action in actions] == ["unstack(b1,b0)"]
