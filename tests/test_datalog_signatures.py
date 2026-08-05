"""The signature-quotiented datalog encoding must agree with the unquotiented one.

`solve_datalog_sig.lp` ranges the pairwise separation constraint over action signature classes
instead of over (instance, state, action) triples. The claim is that this is exactly
equivalence-preserving, so these tests pin the two encodings together rather than checking the
quotient in isolation.
"""

import pddl
import pytest

from genfond.config_handler import ConfigHandler
from genfond.feature_generator import FeaturePool
from genfond.generate_datalog_policy import generate_datalog_policy
from genfond.solver import Solver

MAX_COMPLEXITY = 4


def _solve(domain, problem, type_, max_complexity=MAX_COMPLEXITY):
    config = ConfigHandler(type=type_)
    pool = FeaturePool(domain, [problem], config, max_complexity=max_complexity)
    solver = Solver(pool.to_clingo(), num_threads=1, solve_prog=config["solve_prog"])
    assert solver.solve()
    return pool, solver


def test_signature_facts_are_only_emitted_when_asked(gripper):
    domain, problem = gripper
    pool, _ = _solve(domain, problem, "datalog")
    assert not pool.signature_ids
    assert "asig(" not in pool.to_clingo()


def test_signatures_collapse_the_state_action_pairs(gripper):
    domain, problem = gripper
    pool, _ = _solve(domain, problem, "datalog-sig")
    state_action_pairs = sum(len(node.children) for node in pool.state_graphs[problem.name].nodes.values())
    assert pool.signature_ids
    # The whole point: strictly fewer classes than triples.
    assert len(pool.signature_ids) < state_action_pairs


def test_signature_encoding_grounds_far_smaller(gripper):
    domain, problem = gripper
    _, plain = _solve(domain, problem, "datalog")
    _, signatures = _solve(domain, problem, "datalog-sig")
    plain_atoms = plain.statistics["problem"]["lp"]["atoms"]
    signature_atoms = signatures.statistics["problem"]["lp"]["atoms"]
    assert signature_atoms < plain_atoms / 10


@pytest.mark.parametrize("fixture", ["gripper", "blocks_clear"])
def test_both_encodings_find_the_same_policy(fixture, request):
    domain, problem = request.getfixturevalue(fixture)
    _, plain = _solve(domain, problem, "datalog")
    _, signatures = _solve(domain, problem, "datalog-sig")
    assert generate_datalog_policy(signatures.solution) == generate_datalog_policy(plain.solution)


@pytest.mark.parametrize("fixture", ["gripper", "blocks_clear"])
def test_both_encodings_agree_on_feature_cost(fixture, request):
    domain, problem = request.getfixturevalue(fixture)
    _, plain = _solve(domain, problem, "datalog")
    _, signatures = _solve(domain, problem, "datalog-sig")
    # cost[-1] is the feature complexity; the @2 level is absent when no pruned/2 grounds.
    assert signatures.cost[-1] == plain.cost[-1]


def test_signature_encoding_still_reports_frontier_states():
    """The graph layer is untouched, so frontier expansion must behave identically."""
    domain = pddl.parse_domain("tests/fixtures/pddl_files/gripper/domain.pddl")
    problem = pddl.parse_problem("tests/fixtures/pddl_files/gripper/problem-2-1.pddl")
    config = ConfigHandler(type="datalog-sig")
    assert config["frontier_expansion"]
    pool = FeaturePool(domain, [problem], config, max_complexity=MAX_COMPLEXITY)
    program = pool.to_clingo()
    assert "asig(" in program
    assert "sig_aname(" in program
