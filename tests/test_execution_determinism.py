"""Policy execution must be reproducible from `--seed` alone.

`pddl`'s `Constant.__hash__` is `hash((Constant, name))`: it mixes in the hash of the *class
object*, which is the default identity hash, i.e. an address. `Predicate.__hash__` hashes its
terms, so the address leaks into `State` (`frozenset[Predicate]`) too. Two runs of the same
code with the same `PYTHONHASHSEED` therefore iterate `problem.objects`, `domain.constants` and
a `set[State]` in *different* orders, because ASLR moves the class between processes.

That order used to reach the policy executor: `construct_instance_info` hands out dlplan object
indices in set order, and `execute_datalog_policy` turns those indices back into object names
before shuffling them, so the same RNG draws bound different objects and the policy took a
different trajectory on the same problem with the same seed.

It also used to reach the *solver*: `construct_vocabulary_info` registers `domain.predicates` in
set order, DLPlan enumerates primitive concepts and roles in vocabulary order, and the feature
pool serialises them to the ASP instance in enumeration order. Reordering facts does not change
what the program means, but it does change which of several equally optimal models clingo
reports first -- so an identically seeded run learned a different policy.

These tests pin the orderings down. The address layout is simulated by re-salting
`Constant.__hash__` and re-parsing, which changes exactly the set orders ASLR changes -- without
the flakiness of hoping two real subprocesses happen to land on different layouts.
"""

import os.path
import random
from contextlib import contextmanager

import pddl
import pytest
from pddl.logic.terms import Constant, Variable

from genfond.config_handler import ConfigHandler
from genfond.datalog_policy import DatalogPolicy, DatalogPolicyRule
from genfond.execute_datalog_policy import execute_datalog_policy
from genfond.feature_generator import construct_instance_info, construct_vocabulary_info
from genfond.ground import action_string

FLAT_PROBLEM = """(define (problem blocks-flat)
    (:domain blocksworld)
    (:objects b0 b1 b2 b3 b4)
    (:init (clear b0) (clear b1) (clear b2) (clear b3) (clear b4)
           (ontable b0) (ontable b1) (ontable b2) (ontable b3) (ontable b4))
    (:goal (and (on b0 b1) (on b1 b2) (on b2 b3) (on b3 b4)))
)
"""

DOMAIN_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "pddl_files", "blocks3ops", "domain.pddl")


@contextmanager
def address_layout(salt):
    """Re-hash the `pddl` terms under `salt`, the way a different process address layout would.

    Both terms are needed: `Constant` drives the order of `problem.objects`, `Variable` (through
    `Predicate.__hash__`, which hashes its terms) the order of `domain.predicates`.

    Everything hashed inside the block must also be *built* inside it, so the parsing happens
    here rather than in a fixture: a frozenset built under one hash and read under another would
    be corrupt rather than merely reordered.
    """
    originals = (Constant.__hash__, Variable.__hash__)
    Constant.__hash__ = lambda self, _s=salt: hash((_s, self.name))  # type: ignore[method-assign]
    Variable.__hash__ = lambda self, _s=salt: hash((_s, self.name))  # type: ignore[method-assign]
    try:
        yield
    finally:
        Constant.__hash__, Variable.__hash__ = originals  # type: ignore[method-assign]


def _flat_problem(tmp_path):
    path = tmp_path / "flat.pddl"
    path.write_text(FLAT_PROBLEM)
    return pddl.parse_domain(DOMAIN_PATH), pddl.parse_problem(path)


def _trajectory(tmp_path, salt):
    with address_layout(salt):
        domain, problem = _flat_problem(tmp_path)
        # A single unconditioned rule, so every ground `stack` is a legal choice and the
        # trajectory is decided purely by the binding order the executor draws from.
        policy = DatalogPolicy([DatalogPolicyRule("stack(X, Y)")])
        config = ConfigHandler()
        config["policy_steps"] = 4
        config["abort_on_cycle"] = False
        random.seed(1)
        actions: list = []
        try:
            execute_datalog_policy(domain, problem, policy, config, out_actions=actions)
        except Exception:  # the policy is a stub; only the prefix it took is of interest
            pass
        return [action_string(action) for action in actions]


@pytest.mark.parametrize("salt", [0, 1, 2, 3])
def test_vocabulary_predicates_are_ordered_by_name(tmp_path, salt):
    """The DLPlan vocabulary, and with it the order of the emitted ASP facts, must not depend on
    the address layout."""
    with address_layout(salt):
        domain, _ = _flat_problem(tmp_path)
        vocabulary = construct_vocabulary_info(domain, ConfigHandler())
        names = [p.get_name() for p in vocabulary.get_predicates()]
        assert names == ["clear", "clear_g", "on", "on_g", "ontable", "ontable_g"]


@pytest.mark.parametrize("salt", [0, 1, 2, 3])
def test_instance_objects_are_ordered_by_name(tmp_path, salt):
    """dlplan object indices must not depend on the address layout."""
    with address_layout(salt):
        domain, problem = _flat_problem(tmp_path)
        config = ConfigHandler()
        vocabulary = construct_vocabulary_info(domain, config)
        instance, _ = construct_instance_info(vocabulary, domain, problem, 0, config)
        assert [o.get_name() for o in instance.get_objects()] == ["b0", "b1", "b2", "b3", "b4"]


def test_datalog_execution_is_reproducible_across_address_layouts(tmp_path):
    """Same seed, different address layout, same trajectory."""
    trajectories = [_trajectory(tmp_path, salt) for salt in range(4)]
    assert trajectories[0], "the stub policy should have applied at least one action"
    assert all(trajectory == trajectories[0] for trajectory in trajectories), trajectories
