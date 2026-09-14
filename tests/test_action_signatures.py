"""The precomputed distinguishing sets must agree with the rules they replace.

`iter_dist_set_facts` moves the `sig_c_dist`/`sig_r_dist`/`sig_f_dist` derivation out of the ASP
program and into Python. These tests pin the emitted `sig_pair/3` + `dist/2` relation against a
naive transcription of the rules it replaced.
"""

import re

from genfond.action_signatures import ActionSignature, iter_dist_set_facts


def _parse(signatures):
    pairs = dict()
    dist_sets: dict[int, set[str]] = dict()
    for line in "".join(iter_dist_set_facts(signatures)).splitlines():
        match = re.fullmatch(r"sig_pair\((\d+),(\d+),(\d+)\)\.", line)
        if match:
            pairs[(int(match.group(1)), int(match.group(2)))] = int(match.group(3))
            continue
        match = re.fullmatch(r"dist\((\d+),(.*)\)\.", line)
        assert match, line
        dist_sets.setdefault(int(match.group(1)), set()).add(match.group(2))
    return pairs, dist_sets


def _naive(first: ActionSignature, second: ActionSignature) -> set[str]:
    """The old ASP rules, transcribed."""
    values1 = dict(first.bools)
    values2 = dict(second.bools)
    elements = {f'f("{f}")' for f in values1.keys() & values2.keys() if values1[f] != values2[f]}
    elements |= {f'c("{c}")' for c, n in first.concepts ^ second.concepts if n < min(first.arity, second.arity)}
    elements |= {
        f'r("{r}")' for r, n1, n2 in first.roles ^ second.roles if max(n1, n2) < min(first.arity, second.arity)
    }
    return elements


def _signature(name, arity, concepts, roles, bools):
    return ActionSignature(name, arity, frozenset(concepts), frozenset(roles), tuple(bools))


SIGNATURES = [
    _signature("pick", 2, [("c_top", 0), ("c_clear", 1)], [("r_on", 0, 1)], [("b_a", 1), ("b_b", 0)]),
    _signature("pick", 2, [("c_top", 0)], [("r_on", 0, 1)], [("b_a", 1), ("b_b", 0)]),
    _signature("pick", 2, [("c_top", 0), ("c_clear", 1)], [], [("b_a", 0), ("b_b", 0)]),
    # Same difference as the first pair, so it must share a distinguishing-set id.
    _signature("pick", 2, [("c_top", 0), ("c_clear", 1), ("c_held", 1)], [("r_on", 0, 1)], [("b_a", 1), ("b_b", 0)]),
    _signature("pick", 2, [("c_top", 0), ("c_held", 1)], [("r_on", 0, 1)], [("b_a", 1), ("b_b", 0)]),
    _signature("drop", 1, [("c_top", 0)], [("r_on", 0, 0)], [("b_a", 1), ("b_b", 1)]),
    _signature("drop", 1, [], [], [("b_a", 1), ("b_b", 1)]),
]


def test_pairs_cover_exactly_the_same_action_name():
    pairs, _ = _parse(SIGNATURES)
    expected = {
        (i, j)
        for i in range(len(SIGNATURES))
        for j in range(i + 1, len(SIGNATURES))
        if SIGNATURES[i].name == SIGNATURES[j].name
    }
    assert set(pairs) == expected


def test_dist_sets_match_the_rules_they_replace():
    pairs, dist_sets = _parse(SIGNATURES)
    for (i, j), dist_set_id in pairs.items():
        assert dist_sets.get(dist_set_id, set()) == _naive(SIGNATURES[i], SIGNATURES[j])


def test_identical_dist_sets_are_shared():
    pairs, dist_sets = _parse(SIGNATURES)
    assert pairs[(0, 1)] == pairs[(3, 4)]
    # Every emitted set is distinct, i.e. the deduplication is exact.
    assert len({frozenset(elements) for elements in dist_sets.values()}) == len(dist_sets)
